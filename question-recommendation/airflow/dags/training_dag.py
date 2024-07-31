from datetime import timedelta, datetime
from textwrap import dedent

from airflow import DAG

from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id="training_dag",
    # schedule_interval = "0 0 1 * *",
    schedule_interval="@weekly",
    default_args=default_args,
    start_date=datetime(2021, 1, 16),
    catchup=False
)

### TASK DEFINITIONS ###

# t1
t1 = BashOperator(
    task_id="pull_raw_data",
    bash_command="python3 /opt/smartjen_recsys/src/data/pull_raw_data.py",
    dag=dag
)

# t2
t2 = BashOperator(
    task_id="make_dataset",
    depends_on_past=False,
    bash_command="python3 /opt/smartjen_recsys/src/data/make_dataset.py",
    dag=dag
)

# t3
t3 = BashOperator(
    task_id="filter",
    depends_on_past=False,
    bash_command="python3 /opt/smartjen_recsys/src/data/filter.py",
    dag=dag
)

# t4
t4 = BashOperator(
    task_id="interaction_features",
    depends_on_past=False,
    bash_command="python3 /opt/smartjen_recsys/src/features/interaction_features.py",
    dag=dag
)

# t5
t5 = BashOperator(
    task_id="student_features",
    depends_on_past=False,
    bash_command="python3 /opt/smartjen_recsys/src/features/student_features.py",
    dag=dag
)

# t6
t6 = BashOperator(
    task_id="item_features",
    depends_on_past=False,
    bash_command="python3 /opt/smartjen_recsys/src/features/item_features.py",
    dag=dag
)

# t7
t7 = BashOperator(
    task_id="train_default_model",
    depends_on_past=False,
    bash_command="python3 /opt/smartjen_recsys/src/models/train_model.py --tuning=false",
    dag=dag
)

# t8
t8 = BashOperator(
    task_id="tune_params",
    depends_on_past=False,
    bash_command="python3 /opt/smartjen_recsys/src/models/tune_params.py",
    dag=dag
)

# t9
t9 = BashOperator(
    task_id="train_tuned_model",
    depends_on_past=False,
    bash_command="python3 /opt/smartjen_recsys/src/models/train_model.py --tuning=true",
    dag=dag
)

# t10
t10 = BashOperator(
    task_id="init_model_scoring",
    depends_on_past=False,
    bash_command="python3 /opt/smartjen_recsys/src/models/predict_model.py --training=true",
    dag=dag
)

t11 = BashOperator(
    task_id="upload_weakest_kps",
    depends_on_past=False,
    bash_command="python3 /opt/smartjen_recsys/src/db_integration/upload_results.py",
    dag=dag
)

### SET UP WORKFLOW DEPENDENCIES ###
t1 >> t2 >> t3
t3 >> [t4, t5, t6]
[t4, t5, t6] >> t7
t7 >> t8 >> t9 >> t10 >> t11
