import requests
import json 
import pandas as pd
import os
import pymysql as mysql
import pandas as pd
import torch
import boto3
from botocore import UNSIGNED
from botocore.client import Config as AWSConfig
URL_QUERY = ""

db_config = {
    "pals": {
        "hostname": "",
        # "hostname": "127.0.0.1",
        "user": "admin",
        "password": "qllESKF7SZu9rK7C499G",
        "schema_name": "saas_main",
        "port": 3306
        # "port": 3308
    },
    "alias4urDB": {
        "hostname": "<new_host>",
        "user": "user",
        "password": "password",
        "schema_name": "saas_main",
        "port": "port" #3307
    }
}

def call_db(db_name, query, url_query=URL_QUERY):
    df = None
    try:
        if db_name == "original": db_name = 'pals'
        df = query2db(db_name, query)
        return df
    except:
        try:
            response = requests.get(url_query, params={"db":db_name, "query":query}, headers={'Accept': 'application/json'})
            output = json.loads(response.text)
            return pd.DataFrame(output['body'])
        except Exception as e: 
            print("[ERR] raise from db_adapter.call_db",str(e))
            return None


def query2db(db, query):
    #run query
    if db == "pals" or db=="original" or db=="smartjen":    
        my_db = mysql.connect(host=db_config['pals']['hostname'],
                                        user=db_config['pals']['user'], 
                                        password=db_config['pals']['password'],
                                        db=db_config['pals']['schema_name'],
                                        port=db_config['pals']['port'],
                                        connect_timeout=10)
       

    elif db == 'alias4urDB': #add new DB source here
        my_db = mysql.connect(host=db_config['alias4urDB']['hostname'],
                        user=db_config['alias4urDB']['user'], 
                        password=db_config['alias4urDB']['password'],
                        db=db_config['alias4urDB']['schema_name'],
                        port=db_config['alias4urDB']['port'],
                        connect_timeout=30)
    
    else:            
       raise "body['db'] must be either ['smartjen', 'rnd',  'pals'] with smartjen is student data, rnd is rnd data"
    
    #convert data to dict
    if query.upper().strip().find("SELECT") == 0:
        results = pd.read_sql_query(query, my_db)
    else:
        conn = my_db.cursor()
        conn.execute(query)
        results = pd.DataFrame()
        my_db.commit()
    return results


def get_model_and_config(db_id:int, model_name, device=torch.device("cpu"), root="/tmp"):
    """
    The function `get_model_and_config` downloads a model and its corresponding configuration file from
    an S3 bucket if they do not already exist locally, and returns the model state dictionary and
    dataset configuration.
    
    :param model_name: The `model_name` parameter is the name of the model you want to retrieve. It is
    used to construct the file paths for the model and configuration files
    :param device: The `device` parameter specifies the device on which the model will be loaded. By
    default, it is set to `torch.device("cpu")`, which means the model will be loaded on the CPU.
    However, you can pass a different device, such as `torch.device("cuda")`, to
    :param root: The `root` parameter is the root directory where the model and config files will be
    saved. By default, it is set to `TMPROOT`
    :return: The function `get_model_and_config` returns two values: `model_state_dict` and
    `dataset_cfg`.
    """
    BUCKET_NAME = 'pdf-digitalization' # replace with your bucket name
    s3 = boto3.client('s3', config=AWSConfig(signature_version=UNSIGNED))
    model_state_dicts, dataset_cfgs = [], []
    query = f"SELECT link, model_name  FROM pals_model_config_link WHERE db_id={db_id} and category='kt-model' and model_name like '{model_name}_%' ORDER BY model_name"
    df = call_db("pals", query )

    df['extention'] = df['link'].apply(lambda x: x.split(".")[-1].strip())
    model_links = df[df['extention'] == "pth"].reset_index(drop=True)
    config_links = df[df['extention'] == "json"].reset_index(drop=True)
    for i in range(len(model_links)):
        model_name = model_links['model_name'][i]
        path_model = os.path.join(root, f"{model_name}.pth")
        path_config = os.path.join(root, f"{model_name}.json")

        if not os.path.exists(path_model):
            model_link = model_links['link'][i].split(".com/")[-1]
            config_link = config_links['link'][i].split(".com/")[-1]
            with open(path_model, "wb") as data, open(path_config, "wb") as cfg:
                s3.download_fileobj(BUCKET_NAME, model_link, data)
                s3.download_fileobj(BUCKET_NAME, config_link, cfg)

        model_state_dict = torch.load(path_model,  map_location=device)['model']
        with open(path_config) as f: dataset_cfg = json.load(f) #config4dataset
        model_state_dicts.append(model_state_dict)
        dataset_cfgs.append(dataset_cfg)
    return model_state_dicts, dataset_cfgs