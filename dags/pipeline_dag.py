from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import sys

# Добавим путь к проекту
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from etl.load_data import save_dataset
from etl.preprocess import preprocess
from etl.train_model import train
from etl.evaluate_model import evaluate

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='ml_pipeline_breast_cancer',
    default_args=default_args,
    description='ML pipeline with scikit-learn and Airflow',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=['ml']
) as dag:

    task_load_data = PythonOperator(
        task_id='load_data',
        python_callable=save_dataset
    )

    task_preprocess = PythonOperator(
        task_id='preprocess',
        python_callable=preprocess
    )

    task_train_model = PythonOperator(
        task_id='train_model',
        python_callable=train
    )

    task_evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate
    )

    task_load_data >> task_preprocess >> task_train_model >> task_evaluate
