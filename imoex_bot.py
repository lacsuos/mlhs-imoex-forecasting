import logging
from datetime import datetime
from datetime import timedelta
import pandas as pd
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import torch 
from io import BytesIO
from utils.data_processing import DatasetCreator
import boto3
import os

CLASSIFIER_PATH = 'imoex_project/model_weights/LSTMBinaryClassificator.pth'
REGRESSOR_PATH = 'imoex_project/model_weights/LSTMRegressor.pth'

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Состояния для ConversationHandler
DATE_INPUT = 1

from utils.data_processing import DatasetCreator
from io import BytesIO
import torch
from torch import nn
from utils.data_processing import DatasetCreator
import pandas as pd
import numpy as np
from importlib import reload
import boto3
import pickle
FEATURES = ['close', 'brent', 'usdrub', 'cpi', 'sentiment']


class MoexPredictor:
    def __init__(self, classifier, regressor, boto3_client):
        self.classifier = classifier
        self.regressor = regressor
        self.s3=boto3_client
        self.bucket_name = 'admoskalenko-mlops'

    def predict(self, date):
        object_key = 'imoex_project/datasets/stock_data.pkl'
        response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
        file_content = response['Body']
        stock_df = pd.read_pickle(file_content)
        
        object_key ='imoex_project/datasets/news_df_processed.pkl'
        response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
        file_content = response['Body']
        news_df = pd.read_pickle(file_content)
        
        agg_df = news_df.groupby('date').apply(lambda x: (x['label']*x['score']).sum()).reset_index(name='sentiment')
        agg_df['date'] = agg_df['date'].apply(pd.Timestamp)
        stock_df['date'] = stock_df['date'].apply(pd.Timestamp)
        prepared_df_w_sentiment = pd.merge(stock_df, agg_df, on='date', how='left')
        prepared_df_w_sentiment['sentiment'].fillna(0, inplace=True)
        prepared_df_w_sentiment.dropna(inplace=True)
        prepared_df_w_sentiment.sort_values('date', inplace=True)
        prepared_df_w_sentiment = prepared_df_w_sentiment[prepared_df_w_sentiment['date'] < date].tail(60)
        
        response = s3.get_object(Bucket=bucket_name, Key='imoex_project/model_weights/scaler.pkl')
        model_bytes = response['Body'].read()
        scaler = pickle.loads(model_bytes)
        creator = DatasetCreator(train=False, scaler=scaler)
        X = creator.create_datasets(prepared_df_w_sentiment, FEATURES, 'target')
        with torch.no_grad():
            class_predict = self.classifier(X)
            regr_predict = self.regressor(X)
        return {'value': np.exp(float(regr_predict[-1])), 'up_probability': float(class_predict[-1])}


class MoexPredictorBot:
    def __init__(self, token: str, model):
        self.token = token
        self.model = model  # Ваша обученная модель
        self.app = Application.builder().token(token).build()
        
        # Настройка обработчиков
        self._setup_handlers()
        
    def _setup_handlers(self):
        conv_handler = ConversationHandler(
            entry_points=[
                CommandHandler('start', self.start),
                CommandHandler('predict_tomorrow', self.predict_tomorrow),
                CommandHandler('predict_date', self.predict_date)
            ],
            states={
                DATE_INPUT: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_date_input)]
            },
            fallbacks=[CommandHandler('cancel', self.cancel)]
        )
        
        self.app.add_handler(conv_handler)
        self.app.add_error_handler(self.error_handler)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        user = update.effective_user
        reply_markup = ReplyKeyboardMarkup(
            [["/predict_tomorrow", "/predict_date"]],
            resize_keyboard=True
        )
        
        await update.message.reply_html(
            rf"Привет {user.mention_html()}! Я бот для прогнозирования индекса Мосбиржи.",
            reply_markup=reply_markup
        )

    async def predict_tomorrow(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Прогноз на завтра"""
        tomorrow = datetime.now().date() + timedelta(days=1)
        prediction = self.model.predict(date=tomorrow.strftime('%Y-%m-%d'))
        
        await update.message.reply_text(
            f"Прогноз индекса Мосбиржи на {tomorrow.strftime('%d.%m.%Y')}:\n"
            f"📈 {prediction['value']:.2f} пунктов\n"
            f"Вероятность роста: {prediction['up_probability']:.1%}"
        )

    async def predict_date(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Запрос даты для прогноза"""
        await update.message.reply_text(
            "Введите дату в формате ГГГГ-ММ-ДД (например, 2024-08-15):"
        )
        return DATE_INPUT

    async def handle_date_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка введенной даты"""
        try:
            input_date = datetime.strptime(update.message.text, "%Y-%m-%d").date()
            if input_date > datetime.now().date():
                await update.message.reply_text("Дата не должна быть в будущем!")
                return DATE_INPUT
                
            prediction = self.model.predict(date=update.message.text)
            
            await update.message.reply_text(
                f"Прогноз на {input_date.strftime('%d.%m.%Y')}:\n"
                f"🔹 Ожидаемое значение: {prediction['value']:.2f}\n"
                f"Вероятность роста: {prediction['up_probability']:.1%}"
            )
            return ConversationHandler.END
            
        except ValueError:
            await update.message.reply_text("Неверный формат даты! Попробуйте снова.")
            return DATE_INPUT

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Отмена операции"""
        await update.message.reply_text(
            "Операция отменена.",
            reply_markup=ReplyKeyboardMarkup([["/predict_tomorrow", "/predict_date"]], resize_keyboard=True)
        )
        return ConversationHandler.END

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка ошибок"""
        logger.error("Exception while handling update:", exc_info=context.error)
        
        await update.message.reply_text(
            "Произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже."
        )

    def run(self):
        """Запуск бота"""
        self.app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION")
    )
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    response = s3.get_object(Bucket=bucket_name, Key=CLASSIFIER_PATH)
    model_bytes = response['Body'].read()
    buffer_ = BytesIO(model_bytes)
    classifier = torch.load(buffer_, map_location="cpu", weights_only=False)

    response = s3.get_object(Bucket=bucket_name, Key=REGRESSOR_PATH)
    model_bytes = response['Body'].read()
    buffer_ = BytesIO(model_bytes)
    regressor = torch.load(buffer_, map_location="cpu", weights_only=False)

    model = MoexPredictor(classifier=classifier, regressor=regressor, boto3_client=s3)

    bot = MoexPredictorBot(
        token=token,
        model=model
    )
    bot.run()