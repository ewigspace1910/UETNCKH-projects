import uvicorn
from mangum import Mangum
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request,Query
from typing import List
import os
import time
import datetime
import pandas as pd
import logging
logger=logging.getLogger()
import torch
from torch.utils.data import DataLoader
import warnings
import random
import threading
warnings.filterwarnings('ignore')


from utils import process_data_4infer, Config
from monacobert import MonaCoBERT_CTT, ASSIST2017_PID_DIFF, pid_diff_collate_fn,  model_infer
from db_adapter import call_db, get_model_and_config

app=FastAPI()
TMPROOT = "." if os.getenv("root") is None else os.getenv("root") 

def save2RNDdb(mastery_df, student_id, subject_id, from_db:int, from_sys=0):
    """
    Mastery_df is DF containing 3 columns: topic_id, substrand_id, mastery 
    """
    insert_query = ""
    time_created1 = datetime.datetime.now().strftime("%y-%m-%d %H:%M:00")
    time_created2 = datetime.datetime.now().strftime("%y-%m-%d %H:%M:30")
    topic_sent = "-99"
    print(time_created2)
    for idx, row in enumerate(mastery_df.iterrows()):
        _, (topic_id,  substrand_id, mastery) = row
        # strand_id = substrand2strand[substrand_id] if substrand_id in substrand2strand.keys() else -1
        strand_id=-1
        value = f" ({student_id}, {topic_id}, {strand_id}, {substrand_id}, -1, {subject_id}, {round(mastery, 2)}, '{time_created2}', {from_db}, {from_sys}),"
        insert_query +=  value
        if (idx + 1) % 50 == 0 or (idx+1) == len(mastery_df):
            insert_query = """INSERT INTO pals_student_mastery 
                                    (user_id, topic_id, strand_id, substrand_id, difficulty_level, subject_id, mastery, date, pals_dbid, sys_id) 
                            VALUES  """ + insert_query[:-1] + """;"""
            print(insert_query)
            call_db("pals", insert_query)
            insert_query = ""
        topic_sent +=  f", {topic_id}"
    
    call_db("pals", f"DELETE  FROM  pals_student_mastery WHERE user_id={student_id} and topic_id in ({topic_sent}) and (date < '{time_created1}') and pals_dbid={from_db} and sys_id={from_sys} ")
    if student_id % 5 ==0 :
        print(f"DELETE  FROM  pals_student_mastery WHERE user_id={student_id} and topic_id in ({topic_sent}) and ( date < '{time_created1}') and pals_dbid={from_db} and sys_id={from_sys} ")
        print(f"SAVED! PROFILE student = {student_id}, subject={subject_id}!")
###############################################################################
#   Routers configuration                                                     #
###############################################################################

@app.get("/")
async def root(request: Request):
    return {"message":  f"experiment API in: {str(request.url)}docs" }


##### STUDENT SIDE ########
@app.get("/update_user", summary="get the student mastery on each topic of paticular subject. Result then will be updated to DB. Assume that, your data is checked by API check_support")
async def update2db(student_id:int, subject_id:int, student_level:int=None, from_db:int=1, from_sys:int=0):
    tstart = time.time()
    body, code = None, None
    try:
        #extract list topic from config
        device = torch.device('cpu')
        model_state_dicts, dataset_cfgs = get_model_and_config(from_db, f"subject_{subject_id}", device=device, root=TMPROOT) 
        subject_id = 2 if subject_id == 1111 else subject_id #for test
        if len(model_state_dicts) != len(dataset_cfgs) or len(model_state_dicts)==0: raise IOError("Len of model is incorrect, pls check get_model_link functions") 
        ## 1. create df
        df_src = __query_history__(subject_id, student_id, from_db=from_db)
        ## Process df and infer ##
        results = pd.DataFrame()
        for idx, (model_state_dict, dataset_config) in enumerate(zip(model_state_dicts, dataset_cfgs)):
            ## 4. predict
            y_scores, items, skills, diffs = __infer__(student_id=student_id, subject_id=subject_id,
                                                    dataset_config=dataset_config, model_state_dict=model_state_dict,
                                                    df_src=df_src, student_level=student_level, device=device, from_db=from_db)
            ## 5. save
            idx2items = {dataset_config['pid2idx'][k]:float(k) for k in dataset_config['pid2idx']}
            idx2skills = {dataset_config['q2idx'][k]:float(k) for k in dataset_config['q2idx']}
            skills = [idx2skills[i] for i in skills]
            items = [idx2items[i] for i in items]
            result = pd.DataFrame({
                "topic_id"  : items,
                "substrand" : skills,
                "difficulty_level": diffs,
                "mastery" : y_scores,
                "model"   : [idx] * len(y_scores)
            })
            results = pd.concat([results, result], axis=0)
        results    = results.groupby(['topic_id', "substrand", 'difficulty_level'])['mastery'].mean().reset_index()
        # hamornic mean 
        results['mastery'] = results['mastery'].apply(lambda x: 1/(x+0.00001))
        mastery_df = results.groupby(['topic_id',"substrand"])['mastery'].sum().reset_index()
        count_df = results.groupby(['topic_id',"substrand"])['mastery'].count().reset_index()
        mastery_df['mastery'] = count_df['mastery'] / mastery_df['mastery'] 
            
        ###Save to DB
        # insert_query = ""
        # time_created = datetime.datetime.now().strftime("%y-%m-%d %H:%M:30")
        # query = f"""SELECT org_substrand_id as substrand_id, org_strand_id as strand_id FROM pals_topics 
        #             WHERE db_id={from_db} and org_subject_id={subject_id}"""
        # substrand2strand = call_db("pals", query).set_index("substrand_id").to_dict()['strand_id']
        # for idx, row in enumerate(mastery_df.iterrows()):
        #     _, (topic_id,  substrand_id, mastery) = row
        #     strand_id = substrand2strand[substrand_id] if substrand_id in substrand2strand.keys() else -1
        #     value = f" ({student_id}, {topic_id}, {strand_id}, {substrand_id}, -1, {subject_id}, {round(mastery, 2)}, '{time_created}', {from_db}, {from_sys}),"
        #     insert_query +=  value
        #     if (idx + 1) % 50 == 0 or (idx+1) == len(results):
        #         insert_query = """INSERT INTO pals_student_mastery 
        #                                 (user_id, topic_id, strand_id, substrand_id, difficulty_level, subject_id, mastery, date, from_db, from_sys) 
        #                         VALUES  """ + insert_query[:-1] + """;"""
        #         call_db("pals", insert_query)
        #         insert_query = ""

        print(mastery_df)
        try:
            save2RNDdb(mastery_df, student_id, subject_id, from_db, from_sys)
            curtime=datetime.datetime.now().strftime("%y-%m-%d %H:%M:30")
            body, code = f"Updated data of student {student_id} to RnD DB at {curtime}", True
        except:  raise "===>!!![ERR] cannot save data into DB"

    except Exception as e:
        e = logger.exception(str(e))
        body, code = f"GET ERROR in processing", False
    
    return {
        "complete" :code,
        "body": body,
        "excution time" : time.time() - tstart
    }


@app.get("/infer_user_topics", summary="get the student mastery on paticular topic or on paticular substrand. Result will be return immediately. Assume that, your data is checked by API check_support")
async def infer(student_id:int, subject_id:int, topic_id:int=None, substrand_id:int=None, topic_list:List[int]=Query([]),
            student_level:int=None, from_db:int=1, from_sys:int=0):
    tstart = time.time()
    body, code = None, None
    try:
        #extract list topic from config
        device = torch.device('cpu')
        model_state_dicts, dataset_cfgs = get_model_and_config(from_db, f"subject_{subject_id}", device=device, root=TMPROOT)
        if len(model_state_dicts) != len(dataset_cfgs) or len(model_state_dicts)==0: raise IOError("Len of model is incorrect, pls check get_model_link functions")
        ## 1. create df
        df_src = __query_history__(subject_id, student_id, from_db=from_db)
        
        results = pd.DataFrame()
        for idx, (model_state_dict, dataset_config) in enumerate(zip(model_state_dicts, dataset_cfgs)):
            ## 4. predict
            y_scores, items, skills, diffs = __infer__(student_id=student_id, subject_id=subject_id, dataset_config=dataset_config, 
                                                model_state_dict=model_state_dict, df_src=df_src, 
                                                topic_id=topic_id, substrand_id=substrand_id, student_level=student_level, topic_list=topic_list, 
                                                device=device, from_db=from_db)
            ## 5. save 
            idx2items = {dataset_config['pid2idx'][k]:float(k) for k in dataset_config['pid2idx']}
            idx2skills = {dataset_config['q2idx'][k]:float(k) for k in dataset_config['q2idx']}
            result = pd.DataFrame({
                "topic_id"  : [idx2items[i] for i in items],
                "substrand_id": [idx2skills[i] for i in skills],
                "difficulty_level": diffs,
                "mastery" : y_scores,
                "model"   : [idx] * len(y_scores)
            })
            results = pd.concat([results, result], axis=0)
        results    = results.groupby(['topic_id', 'difficulty_level', 'substrand_id'])['mastery'].mean().reset_index()
        # hamornic mean 
        results['mastery'] = results['mastery'].apply(lambda x: 1/(x+0.00001))
        mastery_df = results.groupby(['topic_id',"substrand_id"])['mastery'].sum().reset_index()
        count_df = results.groupby(['topic_id',"substrand_id"])['mastery'].count().reset_index()
        mastery_df['mastery'] = count_df['mastery'] / mastery_df['mastery'] 
        #############
        try:
            save2RNDdb(mastery_df, student_id, subject_id, from_db, from_sys)
        except:
            print("===>!!![ERR] cannot save data into DB")
        code = True
        body = mastery_df[['topic_id', 'mastery']].to_dict()
    except Exception as e:
        e = logger.exception(str(e))
        body, code = f"GET ERROR in processing", False
    return {
        "complete" :code,
        "body": body,
        "excution time" : time.time() - tstart
    }


def __query_history__(subject_id, student_id, from_db:int):
    query = f"""SELECT sc.topic_id as item_id, sc.question_id, st.substrand_id as skill_id,
                        CASE WHEN sq.facility_index > 0 THEN Round(sq.facility_index ,2) * 100
                            WHEN sq.difficulty_level = 'hard' THEN 90 
                            WHEN sq.difficulty_level = 'easy' THEN 30
                            ELSE 60  
                        END as difficulty, sc.subject_id as subject, sc.created_at as timestamp, 
                        CASE WHEN sc.score / (sc.fullmark + 0.001) is NULL THEN 0
                             WHEN sc.score / (sc.fullmark + 0.001) < 0.7   THEN 0
                             ELSE 1
                        END as correct
                FROM pals_performance_score  as sc
                JOIN pals_questions as sq ON sq.question_id = sc.question_id AND sq.pals_dbid = sc.pals_dbid
                JOIN pals_topics as st ON st.id = sc.topic_id AND st.pals_dbid = sc.pals_dbid
                WHERE sc.pals_dbid = {from_db}  and sc.user_id ={student_id} and sq.subject_type={subject_id} ORDER BY timestamp desc LIMIT 250;"""
    df_src = call_db("pals", query)

    return df_src

def __infer__(student_id, subject_id, dataset_config, model_state_dict, df_src:pd.DataFrame, 
            topic_id:int=None, substrand_id:int=None , student_level=None, topic_list:list=None, device=torch.device('cpu'), from_db:int=1):
    config = Config(dataset_config)  
    valid_skill = [float(k) for k in dataset_config['q2idx'].keys()  ]
    topic2strand = call_db("pals", f"""SELECT  distinct(id) as topic_id, substrand_id as skill_id   
                                    FROM pals_topics  
                                    WHERE pals_dbid={from_db} and subject_type={subject_id};""").set_index('topic_id').to_dict()['skill_id']       
    
    topic2strand = {k:v for k,v in topic2strand.items() if v in valid_skill}
    valid_topic = [float(k) for k in dataset_config['pid2idx'].keys() if float(k) in topic2strand.keys()]
    df_src["valid_topic"] = df_src['item_id'].apply(lambda x: True if x in valid_topic else False)
    df = df_src[df_src['valid_topic']].drop(labels=['valid_topic'], axis=1)

    if not topic_id is None: #topic
        df = process_data_4infer(df, valid_topic=valid_topic, valid_skill=valid_skill, 
                    topic2strand=topic2strand, max_seq_len=config.max_seq_len, topic_id=topic_id)
    elif not substrand_id is None: #by substrand
        query = f"""SELECT distinct(id) as topic_id FROM pals_topics  WHERE pals_dbid={from_db} and substrand_id={substrand_id}"""
        student_topic = call_db("pals", query)
        student_topic = [t for t in student_topic['topic_id'].to_list() if t in valid_topic]
        df = process_data_4infer(df, valid_topic=student_topic, valid_skill=valid_skill, topic2strand=topic2strand, max_seq_len=config.max_seq_len)
    elif not topic_list is None and len(topic_list)> 0:
        student_topic = [t for t in topic_list if t in valid_topic]
        df = process_data_4infer(df, valid_topic=student_topic, valid_skill=valid_skill, topic2strand=topic2strand, max_seq_len=config.max_seq_len)
    else: # all topic student learnt
        query = f"""SELECT distinct(s.topic_id) FROM pals_performance_score as s WHERE s.pals_dbid={from_db} and s.user_id={student_id} and s.subject_id={subject_id}"""
        student_topic = call_db("pals", query)
        student_topic = [t for t in student_topic['topic_id'].to_list() if t in valid_topic]
        df = process_data_4infer(df, valid_topic=student_topic, valid_skill=valid_skill, topic2strand=topic2strand, max_seq_len=config.max_seq_len)
    
    
    ## 2. create dataloader
    test_dataset = ASSIST2017_PID_DIFF(config.max_seq_len, config=config, dataset_dir=df,
                                    q2idx=dataset_config['q2idx'], pid2idx=dataset_config['pid2idx'])
    num_q = test_dataset.num_q
    num_r = test_dataset.num_r
    num_pid = test_dataset.num_pid
    num_diff = test_dataset.num_diff
    test_loader = DataLoader(test_dataset,batch_size = config.batch_size, shuffle = False, collate_fn = pid_diff_collate_fn)

    ## 3. load model
    model = MonaCoBERT_CTT(
        num_q=num_q, num_r=num_r, num_pid=num_pid, num_diff=num_diff,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        num_head=config.num_head, num_encoder=config.num_encoder,
        max_seq_len=config.max_seq_len, device=device,
        use_leakyrelu=config.use_leakyrelu,
        dropout_p=config.dropout_p
    ).to(device)
    model.load_state_dict(model_state_dict)  
    
    ## 4. predict
    y_scores, items, skills, diffs = model_infer(model, test_loader, device)
    return y_scores, items, skills, diffs

###############################################################################
#   Handler for AWS Lambda                                                    #
###############################################################################

handler = Mangum(app)

###############################################################################
#   Run the self contained application                                        #
###############################################################################


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=9000)
