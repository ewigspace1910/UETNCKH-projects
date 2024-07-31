import pandas as pd
import json
import os
import datetime
import requests
import numpy as np
import boto3
import json
from botocore import UNSIGNED
from botocore.client import Config as AWSConfig
import logging
logging.basicConfig(level=logging.ERROR,format='%(levelname) -10s %(asctime)s %(module)s:%(lineno)s %(funcName)s %(message)s')
LOGGER = logging.getLogger()
from const import MIN_QUESTS_IN_TOPIC, BUCKER_NAME
from db_integration import call_db, QueryBank, QueryRnDBank

def request2s3(file_name, file_path, s3name=BUCKER_NAME):
        try:
            file_name = f"recsys/{file_name}"
            s3 = boto3.client('s3', config=AWSConfig(signature_version=UNSIGNED))
            s3.upload_file(file_path, s3name, file_name) #upload to s3.
            text = "https://{}.s3.amazonaws.com/{}".format(s3name, file_name)
            return text
        except Exception as e:
            print("S3 error", e)
            return None

def submit_to_link_table(file_path, model_name, category, db_id:int):
    try:
        file_name = os.path.basename(file_path)
        extention = file_name.split(".")[-1]
        link = request2s3(file_name=file_name, file_path=file_path )
        updated =  datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
        if link is None: raise Exception("ERROR in uploading to S3")
        call_db('pals', f"INSERT INTO  pals_model_config_link (db_id, link, category, model_name, extention, updated) VALUES ({db_id}, '{link}', '{category}', '{model_name}', '{extention}', '{updated}')")
        call_db('pals', f"DELETE  FROM  pals_model_config_link WHERE db_id={db_id} and category='{category}' and model_name='{model_name}' and extention='{extention}' and (updated is NULL or updated < '{updated}')")
    except Exception as e:
        LOGGER.exception(str(e))
        print(f"Cannot upload file-->{file_path} because of {str(e)}")

############## CHECK INPUT 4 RECSYS ##############
def check_model_config(model_name, db_id):
    query = QueryRnDBank.get_model_config_link(db_id=db_id, category='kt-model', model_name=model_name, use_like=True)
    df = call_db("pals", query )
    # model_link = df[df['extention'] == "pth"]['link'][0]
    config_link = df[df['extention'] == "json"]['link'][0]

    dataset_cfg = requests.get(config_link)
    dataset_cfg = json.loads(dataset_cfg.text)

    return dataset_cfg

def check_kt_ingredient(subject_id:int, student_id:int=None, topic_id:int=None, substrand_id:int=None, 
                        student_level:int=None, filter_pendingmark:bool=False, from_db:int=-1):
    """
    return by code in ERROR_DEFFINITION
    """
    try:
        dataset_cfg = check_model_config(f"subject_{subject_id}_%", from_db)
        valid_topic = [float(k) for k in dataset_cfg['pid2idx'].keys()]
        valid_substrand = [float(k) for k in dataset_cfg['q2idx']]
    except Exception as e:
        LOGGER.error("[ERR] utils.check_kt_ingredient : "+str(e))
        return 400
    
    #check student
    if not student_id is None:
        topic= call_db("pals", QueryBank.get_topic_id_from_performance_records(from_db, student_id, subject_id))['topic_id'].to_list()
        if len(set(valid_topic) & set(topic)) == 0:
            return 200
    #check topic_id
    if not topic_id  is None:
        if not topic_id in valid_topic: return 301
        elif not student_level is None:
            query = QueryBank.get_num_quests_in_topics(db_id=from_db, topic_list=[topic_id], student_level=student_level)
            question_bank = call_db("pals", query, to_json=True)
            if question_bank is None or list(question_bank['num'].values())[0] < MIN_QUESTS_IN_TOPIC:return 303
    #check substrand
    # if not substrand_id is None:
    #     if not substrand_id in valid_substrand: return 301
        # if not student_level is None:
        #     query = QueryBank.get_topic_id_under_substrand(subject_id=subject_id, substrand_id=substrand_id, student_level=student_level)
        #     topic_levels = call_db("pals", query)['topic_id'].to_list()
        #     topic_levels = [x for x in topic_levels if x in valid_topic]
        #     if len(topic_levels) == 0 : return 304
    return -1
