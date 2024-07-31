mkdir airflow
export AIRFLOW_HOME=$(pwd)/airflow
pip3 install apache-airflow
airflow db init