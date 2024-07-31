import os
import yaml
import requests
import boto3
import json
from botocore import UNSIGNED
from botocore.client import Config as AWSConfig
import logging
import datetime
logger=logging.getLogger()
import concurrent.futures
URL_QUERY = "https://spld938daj.execute-api.ap-southeast-1.amazonaws.com/recsys-db-query"
DB = "rnd"
BUCKER_NAME = "pdf-digitalization"

def request2s3(file_name, file_path, s3name=BUCKER_NAME):
        try:
            file_name = f"model/{file_name}"
            s3 = boto3.client('s3', config=AWSConfig(signature_version=UNSIGNED))
            s3.upload_file(file_path, s3name, file_name) #upload to s3.
            text = "https://{}.s3.amazonaws.com/{}".format(s3name, file_name)
            return text
        except Exception as e:
            print("S3 error", e)
            return None  

import pymysql as mysql
import pandas as pd
import db_integration.query_rnd as QueryRnDBank
import db_integration.query as QueryBank

def call_db(db_name, query, cfg, to_json=False):
    df = None
    try:
        if db_name == "original": db_name = 'pals'
        df = query2db(db_name, query, cfg)
        if df is None : raise Exception("Cannot query to DB with query : \n{}".format(query))
        if to_json: return {str(k):v for k, v in df.to_dict().items()}
        return df
    except Exception as e:
        print("\t===>[ERR] raise from db_integration.call_db",str(e))
        df = None



def query2db(db, query, cfg):
    #run query
    if db == "pals" or db=="original" or db=="smartjen":    
        my_db = mysql.connect(host=cfg['indb']['hostname'],
                                        user=cfg['indb']['user'], 
                                        password=cfg['indb']['password'],
                                        db=cfg['indb']['schema_name'],
                                        port=cfg['indb']['port'],
                                        connect_timeout=10)
    
    else:            
       raise "body['db'] must be either ['smartjen', 'rnd', 'hwdb'] with smartjen is student data, rnd is rnd data"
    
    #convert data to dict
    if query.upper().strip().find("SELECT") == 0:
        results = pd.read_sql_query(query, my_db)
    else:
        conn = my_db.cursor()
        conn.execute(query)
        results = pd.DataFrame()
        my_db.commit()
    return results


def subprocess_request(file_path, model_name, cfg):
    try:
        file_name = os.path.basename(file_path)
        if not 'ktmodel' in file_name: return
        # extention = file_name.split(".")[-1]
        link = request2s3(file_name=file_name, file_path=file_path ) 
        # updated =  datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
        updated =  call_db('pals',"SELECT  NOW() as now FROM pals_model_config_link LIMIT 1", cfg)['now']['0']
        print("updated at :", updated)
        db_id=file_name.split("-")[0].replace("dbid", "")#dbid{args.dbid}-
        file_name=file_name.split("-")[1]#.replace("_ktmodel")#dbid{args.dbid}-
        if link is None: raise Exception("ERROR in uploading to S3")
        call_db('pals', f"INSERT INTO  pals_model_config_link (db_id, link, category, model_name, updated) VALUES ('{db_id}, {link}', 'kt-model', '{file_name}', '{updated}')",
                cfg=cfg)
        call_db(f"""DELETE  FROM  pals_model_config_link 
                    WHERE   category='kt-model' and model_name='{file_name}' and db_id={db_id}
                        and (updated is NULL or updated < DATE_SUB(NOW(), INTERVAL 48 HOUR) ) """)
    except Exception as e:
        logger.exception(str(e))
        print(f"Cannot upload file-->{file_path} because of {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Finetune Training")
    parser.add_argument('-s', '--dbid', type=str)
    args = parser.parse_args()

    dir_path = os.getcwd()
    with open("cfg/db.yml", "r") as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)
    with open("cfg/config.yml", "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    modelfiles = sorted([p.path for p in os.scandir(config['data']['models_folder']) if p.is_file()])

    for p in modelfiles:
        model_name = os.path.basename(p).split(".")[0]
        print("update file-->", model_name)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(subprocess_request, p, model_name, cfg)

    # Log
    print("New model in db :")
    df  = call_db('pals', f"""SELECT * FROM  pals_model_config_link  WHERE   category='kt-model'""", cfg)
    print(df)