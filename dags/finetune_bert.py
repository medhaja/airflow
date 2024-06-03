from airflow.models import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import numpy as np
import wandb
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AdamW
from datasets import load_dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import os
os.environ['WANDB_API_KEY'] = ''
# Define the BertFineTuner class as you had before
class BertFineTuner:
    def __init__(self, project_name, entity_name, model_name, dataset_name, dataset_split, learning_rate):
        self.project_name = project_name
        self.entity_name = entity_name
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.learning_rate = learning_rate
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Initialize W&B
        wandb.init(project=self.project_name, entity=self.entity_name)

    def load_and_tokenize_dataset(self):
        dataset = load_dataset(self.dataset_name, self.dataset_split)
        dataset = dataset["train"]
        tokenized_data = self.tokenizer(dataset["sentence"], return_tensors="np", padding=True, truncation=True)
        self.tokenized_data = dict(tokenized_data)
        self.labels = np.array(dataset["label"])

    def compile_model(self):
        self.model.compile(optimizer=AdamW(learning_rate=self.learning_rate))

    def train_model(self, epochs, batch_size):
        class WandbCallback(Callback):
            def on_epoch_end(self, epoch, logs=None):
                wandb.log(logs)

        self.model.fit(
            self.tokenized_data, 
            self.labels, 
            epochs=epochs, 
            batch_size=batch_size, 
            callbacks=[WandbCallback()]
        )

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

fine_tuner = BertFineTuner(
    project_name="bert_finetuning",
    entity_name="medhaja",
    model_name="bert-base-cased",
    dataset_name="glue",
    dataset_split="cola",
    learning_rate=3e-5
)

# Define Airflow tasks
def load_and_tokenize_dataset():
    fine_tuner.load_and_tokenize_dataset()

def compile_model():
    fine_tuner.compile_model()

def train_model():
    fine_tuner.train_model(epochs=3, batch_size=32)

def save_model():
    fine_tuner.save_model(save_path='model')

# Define the DAG
with DAG(
    dag_id='bert_finetuning_dag',
    schedule_interval='@daily',  # Run every day
    start_date=datetime(2024, 6, 1),
    catchup=False
) as dag:

    task_load_and_tokenize_dataset = PythonOperator(
        task_id='load_and_tokenize_dataset',
        python_callable=load_and_tokenize_dataset,
        execution_timeout=timedelta(minutes=60)
    )

    task_compile_model = PythonOperator(
        task_id='compile_model',
        python_callable=compile_model,
        execution_timeout=timedelta(minutes=60)
    )

    task_train_model = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        execution_timeout=timedelta(minutes=60)
    )

    task_save_model = PythonOperator(
        task_id='save_model',
        python_callable=save_model,
        execution_timeout=timedelta(minutes=60)
    )

    task_load_and_tokenize_dataset >> task_compile_model >> task_train_model >> task_save_model
