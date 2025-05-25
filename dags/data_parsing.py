from airflow.models import DAG, Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import sys
from utils.parsers.news_parser import lentaRu_parser
from utils.parsers.stock_parser import StockMarketParser
from utils.models.lstm_models import SentimentModel
import pickle
import json
import pandas as pd
from typing import Any, Dict, Literal
import logging
import io
from transformers import AutoTokenizer, AutoModel
import torch
from datetime import datetime, timedelta

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())


KEYWORDS = ['ЦБ РФ', 
            'нефть', 
            'IMOEX', 
            'санкции', 
            'дивиденды', 
            'президент', 
            'сша', 
            'трамп', 
            'путин', 
            'лавров', 
            'цб', 
            'сбербанк', 
            'яндекс', 
            'лукойл',
            'война',
            'ближний восток',
            'нато',
            'курс',
            'опек',
            'минфин',
            'правительство',
            'одкб',
            'макрон',
            'меркель',
            'германия',
            'франция',
            'великобритания',
            'испания',
            'италия',
            'газ',
            'северный поток',
            'мосбиржа',
            'урегулирование',
            'мир',
            'война',
            'сво',
            'лукашенко',
            'силуанов',
            'Набиуллина',
            'нарышкин',
            'шойгу',
            'белоусов',
            'экономика',
            'риски',
            'застройщики',
            'ключевая ставка',
            'ставка',
            'инфляция',
            'ввп',
            'медведев',
           'газпром',
           'украина',
           'байден',
           'обама',
           'порошенко',
           'зеленский',
           'Система',
           'Аэрофлот',
           'Алроса',
           'Астра',
           'Московский кредитный банк',
           'мкб'
            'Северсталь',
            'санкция',
            'санкционный список',
            'En+ Group',
          'ФСК-Россети',
            'специальная военная операция',
            'спецоперация',
          'Совкомфлот',
            'Газпром',
            'Норникель',
            'Хэдхантер',
            'РусГидро',
            'Интер РАО',
            'Европлан',
            'Лукойл',
            'MAGN',
            'Магнит',
            'Московская биржа',
            'мосбиржа',
            'Мосэнерго',
            'Мечел',
            'МТС',
            'НЛМК',
            'НОВАТЭК',
            'ФосАгро',
            'ГК ПИК',
            'Полюс',
            'Роснефть',
            'европа',
            'евросоюз',
            'ес',
            'Ростелеком',
            'РУСАЛ',
            'Сбер',
            'Селигдар',
            'ГК Самолет',
            'Сургутнефтегаз',
            'Совкомбанк',
            'ТКС Холдинг',
            'Татнефть',
            'Транснефть',
            'ЮГК',
            'Юнипро',
            'ВК',
            'ВТБ',
            'Яндекс',
]
BUCKET = Variable.get("S3_BUCKET")
FIRST_DATE = datetime(2013, 1, 1).date()
SECOND_DATE = datetime.now().date()

DEFAULT_ARGS = {
    "owner": "admoskalenko",
    "retry": 3,
    "retry_delay": timedelta(minutes=1)
}
BERT_MODEL = "deepvk/bert-base-uncased"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_bert_embedding(tokenizer, bert, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state.cpu().mean(dim=1).squeeze().numpy()

def process_news(news_df, sentiment_model):
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    bert = AutoModel.from_pretrained(BERT_MODEL)
    bert.to(DEVICE)
    news_df["embedding"] = news_df["title"].apply(lambda x: get_bert_embedding(tokenizer, bert, x))
    news_df['sentiment_prediction'] = news_df['title'].apply(sentiment_model)
    news_df['label'] = news_df['sentiment_prediction'].apply(lambda x: x[0]['label']).map({'neutral': 0, 'positive': 1, 'negative': -1})
    news_df['score'] = news_df['sentiment_prediction'].apply(lambda x: x[0]['score'])
    return news_df[['title', 'date', 'embedding', 'label', 'score']]


def create_dag(dag_id: str):

    ####### DAG STEPS #######

    def init() -> Dict[str, Any]:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _LOG.info(f'Pipeline started: {date}')
        metrics = dict()
        metrics['pipeline_start_date'] = date
        return metrics

    def get_stock_data(**kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='init')
        metrics['download_stock_date_begin'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        stock = StockMarketParser()
        df = stock.get_data(FIRST_DATE, SECOND_DATE)
        df.dropna(inplace=True)
        metrics['download_stock_date_end'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics['stock_data_shape'] = list(df.shape)

        s3_hook = S3Hook("s3_connection")
        filebuffer = io.BytesIO()
        df.to_pickle(filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"imoex_project/datasets/stock_data.pkl",
            bucket_name=BUCKET,
            replace=True,
        )
        _LOG.info(f'Stock market data has been parsed and uploaded. {df.shape[0]} shape of dataset')
        return metrics



    def get_news_data(**kwargs) -> Dict[str, Any]:
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='init')
        metrics['news_parsing_date_begin'] = date

        s3_hook = S3Hook("s3_connection")
        file = s3_hook.download_file(
            key=f"imoex_project/datasets/news_df_processed.pkl",
            bucket_name=BUCKET)
        df = pd.read_pickle(file)
        query = ''
        offset = 0
        size = 500
        sort = "3"
        title_only = "0"
        domain = "1"
        material = "0"
        bloc = "4"
        dateFrom = df['date'].astype(str).max()
        dateTo = min(SECOND_DATE.strftime('%Y-%m-%d'), dateFrom)
        param_dict = {'query'     : query, 
                'from'      : str(offset),
                'size'      : str(size),
                'dateFrom'  : dateFrom,
                'dateTo'    : dateTo,
                'sort'      : sort,
                'title_only': title_only,
                'type'      : material, 
                'bloc'      : bloc,
                'domain'    : domain}
        parser = lentaRu_parser()
        tbl = parser.get_articles(param_dict=param_dict,
                         time_step = 4,
                         save_every = 5, 
                         save_excel = False)
        _LOG.info(f'Parsed {tbl.shape[0]} news')
        _LOG.info(f'Parsed {tbl.columns} news')
        tbl['is_relevant'] = tbl['title'].str.contains('\\b(' + '|'.join(KEYWORDS) + ')\\b', case=False, regex=True, na=False)
        tbl['date'] = pd.to_datetime(tbl['pubdate'],unit='s').dt.date.apply(pd.to_datetime)
        df['date'] = df['date'].apply(pd.to_datetime)
        tbl = tbl[tbl['is_relevant']].copy()
        sentiment_model = SentimentModel()
        tbl = process_news(tbl, sentiment_model)
        updated_df = pd.concat((tbl, df)).drop_duplicates(subset=['title', 'date', 'score', 'label'])
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filebuffer = io.BytesIO()
        updated_df.to_pickle(filebuffer)
        updated_df = updated_df['date'].astype(str)
        filebuffer.seek(0)
        s3_hook.load_file_obj(
            file_obj=filebuffer,
            key=f"imoex_project/datasets/news_df_processed.pkl",
            bucket_name=BUCKET,
            replace=True,
        )
        metrics['news_parsing_date_end'] = date 
        _LOG.info('Data has been prepared.')
        return metrics 


    def save_results(**kwargs) -> None:
        ti = kwargs["ti"]
        stock_metrics = ti.xcom_pull(task_ids='get_stock_data')
        news_metrics = ti.xcom_pull(task_ids='get_news_data')
        merged = stock_metrics | news_metrics
        s3_hook = S3Hook("s3_connection")
        buff = io.BytesIO()
        buff.write(json.dumps(merged, indent=2).encode())
        buff.seek(0)
        s3_hook.load_file_obj(file_obj = buff, key=f"imoex_project/metrics/metrics.json", bucket_name=BUCKET, replace=True)

    dag = DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",
        start_date=days_ago(2),
        catchup=False,
        default_args=DEFAULT_ARGS
        )
    with dag:
        task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)
        task_get_data = [PythonOperator(task_id=f"get_stock_data", python_callable=get_stock_data, dag=dag), PythonOperator(task_id=f"get_news_data", python_callable=get_news_data, dag=dag)]
        task_save_results = PythonOperator(task_id="save_result", python_callable=save_results, dag=dag)
        task_init >> task_get_data >> task_save_results

create_dag(f"get_training_data")