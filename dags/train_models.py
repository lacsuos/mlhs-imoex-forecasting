from airflow.models import DAG, Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from airflow.models.baseoperator import chain
import mlflow
import sys
import numpy as np
from utils.trainers import train_classifier, train_regressor
from utils.data_processing import DatasetCreator
import torch
from torch import nn
import os
import pickle
import json
import pandas as pd
from typing import Any, Dict, Literal, List
from torch.utils.data import Dataset, DataLoader
import logging
import io
from datetime import datetime, timedelta
_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())


BATCH_SIZE = 32
BUCKET = Variable.get("S3_BUCKET")

DEFAULT_ARGS = {
    "owner": "admoskalenko",
    "retry": 3,
    "retry_delay": timedelta(minutes=1)
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEATURES = ['close', 'brent', 'usdrub', 'cpi', 'sentiment']


def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)

from utils.models.lstm_models import LSTMBinaryClassifier, AttentionLSTMClassifier, AttentionLSTMRegressor, LSTMRegressor
model_names = ["LSTMBinaryClassificator", "AttentionLSTMClassifier", "LSTMRegressor", "AttentionLSTMRegressor"]
models = dict(
    zip(model_names, [
        LSTMBinaryClassifier(len(FEATURES), 32, 1),
        AttentionLSTMClassifier(len(FEATURES), 64, 1, 1, 0.3),
        LSTMRegressor(len(FEATURES), 32, 1),
        AttentionLSTMRegressor(len(FEATURES), 64, 1, 1, 0.3),
    ]))


def create_dag(dag_id: str, m_names: List, exp_name: str):

    ####### DAG STEPS #######

    def init() -> Dict[str, Any]:
        exps = mlflow.search_experiments(filter_string=f"name = '{exp_name}'")

        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _LOG.info(f'Pipeline started: {date}')

        if len(exps) > 0:
            experiment_id = exps[0].experiment_id
        else:
            experiment_id = mlflow.create_experiment(exp_name)

        import psutil
        mem = psutil.virtual_memory()
        _LOG.info(f"Available RAM: {mem.available / 1024**3:.2f} GB")
        if mem.available < 2 * 1024**3:  # Менее 2 GB
            raise Exception("Not enough RAM!")
        mlflow.start_run(run_name="tau_ceti_pn", experiment_id = experiment_id, description = "parent")
        run = mlflow.active_run()
        metrics = dict()
        metrics['pipeline_start_date'] = date
        metrics['experiment_id'] = experiment_id
        metrics['run_id'] = run.info.run_id
        return metrics

    def train_model(model_name, task, **kwargs) -> Dict[str, Any]:
        ti = kwargs['ti']
        metrics = ti.xcom_pull(task_ids='init')
        s3_hook = S3Hook("s3_connection")
        metrics['download_date_begin'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        file = s3_hook.download_file(
            key=f"imoex_project/datasets/stock_data.pkl",
            bucket_name=BUCKET)
        stock_df = pd.read_pickle(file)

        file = s3_hook.download_file(
            key=f"imoex_project/datasets/news_df_processed.pkl",
            bucket_name=BUCKET)
        news_df = pd.read_pickle(file)
        agg_df = news_df.groupby('date').apply(lambda x: (x['label']*x['score']).sum()).reset_index(name='sentiment')
        agg_df['date'] = agg_df['date'].apply(pd.Timestamp)
        stock_df['date'] = stock_df['date'].apply(pd.Timestamp)
        prepared_df_w_sentiment = pd.merge(stock_df, agg_df, on='date', how='left')
        prepared_df_w_sentiment['sentiment'].fillna(0, inplace=True)
        prepared_df_w_sentiment.dropna(inplace=True)
        prepared_df_w_sentiment.sort_values('date', inplace=True)
        creator = DatasetCreator(train=True)
        if task == 'regression':
            trainer = train_regressor
            prepared_df_w_sentiment['target'] = np.log(prepared_df_w_sentiment['close'])
            criterion = nn.MSELoss()
        elif task == 'classification':
            trainer = train_classifier
            prepared_df_w_sentiment['target'] = prepared_df_w_sentiment['close'].diff() > 0
            criterion = nn.BCELoss()
        _LOG.info(f'{task}')
        train_dataset, test_dataset = creator.create_datasets(prepared_df_w_sentiment, FEATURES, 'target')
        buffer = io.BytesIO()
        pickle.dump(creator.scaler, buffer)
        buffer.seek(0)   
        s3_hook.load_file_obj(
            file_obj=buffer,
            key=f"imoex_project/model_weights/scaler.pkl",
            bucket_name=BUCKET,
            replace=True,
            )

        _LOG.info(f'{train_dataset.X.size()}, {test_dataset.X.size()} train-test sizes')
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        experiment_id = metrics['experiment_id']
        parent_run = metrics['run_id']
        _LOG.info(f'{prepared_df_w_sentiment.shape[0]} shape of prepared dataset | {stock_df.shape[0]} shape of stock dataset | {agg_df.shape[0]} shape of news dataset')
        _LOG.info('MLFlow exp started')
        with mlflow.start_run(run_name=model_name, experiment_id=experiment_id, parent_run_id=parent_run, nested=True) as child_run:
            timestamps = {}
            date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            timestamps['train_date_begin'] = date 
            model = models[model_name]
            model.to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            _LOG.info(f'{model_name} training began')
            model_weights, model_metrics = trainer(model=model, train_loader=train_loader, val_loader=test_loader, optimizer=optimizer, criterion=criterion, epochs=1000, patience=200)
            model.load_state_dict(model_weights)

            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=model_name,
                registered_model_name=model_name
            )
            filebuffer = io.BytesIO()
            torch.save(model, filebuffer)
            filebuffer.seek(0)
            s3_hook.load_file_obj(
                file_obj=filebuffer,
                key=f"imoex_project/model_weights/{model_name}.pth",
                bucket_name=BUCKET,
                replace=True,
            )
            for m, val in model_metrics.items():    
                mlflow.log_metric(m, val)
            metrics['train_date_end'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        _LOG.info(f'{model_name} has been trained')
        return metrics





    def save_results(**kwargs) -> None:
        ti = kwargs["ti"]
        metrics = ti.xcom_pull(task_ids='init')
        for m_name in m_names:
            model_metrics = ti.xcom_pull(task_ids = f"train_{m_name}")
            metrics[f'{m_name}'] = model_metrics
        s3_hook = S3Hook("s3_connection")
        buff = io.BytesIO()
        buff.write(json.dumps(metrics, indent=2).encode())
        buff.seek(0)
        s3_hook.load_file_obj(file_obj = buff, key=f"imoex_project/metrics/metrics_train.json", bucket_name=BUCKET, replace=True)

    ####### INIT DAG #######

    dag = DAG(
        dag_id=dag_id,
        schedule_interval="0 1 * * *",
        start_date=days_ago(2),
        catchup=False,
        default_args=DEFAULT_ARGS
    )
    try:
        with dag:
            task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)
            tasks_train = [
                PythonOperator(
                    task_id=f"train_{m_name}", 
                    python_callable=train_model, 
                    dag=dag, 
                    op_kwargs={'model_name': m_name, 'task': 'regression' if 'regressor' in m_name.lower() else 'classification'}
                    ) for m_name in m_names
                    ]
            task_save_results = PythonOperator(task_id="save_result", python_callable=save_results, dag=dag)
            chain(task_init, *tasks_train, task_save_results)
    finally:
        mlflow.end_run()

configure_mlflow()
create_dag(f"train_models", model_names, 'lstms')