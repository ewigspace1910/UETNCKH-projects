from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from typing import List
import json
import pandas as pd
import time
import requests
import const
from utils import LOGGER, check_kt_ingredient
from db_integration import call_db, QueryBank, QueryRnDBank
from learningpath_gen.baserule  import lp_from_list_of_topicids_by_kg, lp_from_list_of_topicname
from routers.api_kg_interaction import filter_topic_api
import concurrent.futures
import multiprocessing as mp
router = APIRouter(
    prefix="/kt",
    tags=['(1.x) Knowledge tracing']
)

@router.get("/update-student", summary="update mastery state of student on all topics in a subject")
async def kt_update_student(student_id:int, subject_id:int, from_db:int, from_sys:int=0):
    tstart = time.time()
    try:
        r = requests.get(const.URL_KTmodel['infer_user_topic'], params ={"student_id":student_id, "subject_id":subject_id, "from_db":from_db, 'from_sys':from_sys})
        r = json.loads(r.text)
        if  r['complete']: output = f"success = {str(r['complete'])}"
        else: raise Exception(r['body'])
        return JSONResponse({"complete" : True, "msg": output, "execution time" : time.time() - tstart})
    except Exception as e:
        e = LOGGER.exception(str(e))
        return JSONResponse({"complete" : False, "msg":str(e),  "execution time" : time.time() - tstart})


@router.get("/get-mastery", summary="get mastery of student on each topic in DB")
async def kt_get_mastery(student_ids:List[int] = Query(), topic_ids:List[int] = Query(), subject_id:int=2, from_db:int=1, from_sys:int=0):
    tstart = time.time()
    output = {}
    for student_id in student_ids:
        #summary each topic
        query = QueryRnDBank.get_mastery_records(db_id=from_db, student_id=student_id, subject_id=subject_id, from_sys=from_sys)
        response = call_db("pals", query)
        if response is None:
            output[student_id]={'mastery' : {k:"unknow"  for k in topic_ids}} 
        else: 
            student_profile = response.set_index('topic_id').to_dict()['mastery']     
            output[student_id]={
                'mastery' : {k:"unknow" if k not in student_profile else student_profile[k] for k in topic_ids}
            } 
    return JSONResponse({
        "complete" : True,
        "msg" : output,
        "execution time" : time.time() - tstart
    })



@router.get("/find-weakest-topics", summary="find top-k weakest topics under substrand of a student")
async def kt_get_weakest_topics(from_db:int, student_id:int, substrand_id:int, subject_id:int, student_level:int,  top_k:int=50, branch_names:List[str]=Query([]), from_sys:int=0):
    tstart = time.time()
    top_k = 50
    output = None
    try:
        #check substrand valid
        r = requests.get(const.URL_KTmodel['infer_user_topic'], params ={"student_id":student_id, "subject_id":subject_id, 'substrand_id':substrand_id, "from_db":from_db, 'from_sys':from_sys})
        r = json.loads(r.text)
        if not r['complete']:  
            LOGGER.error(f"[ERR] KT model raise error in kt_get_weakest_topics with student_id={student_id}, subject_id={subject_id}, substrand_id={substrand_id}")
            tmp_df = {}
        else:
            tmp_df = pd.DataFrame(r['body'])
            if "topic_id" in tmp_df.columns: tmp_df = tmp_df.set_index("topic_id")
            tmp_df = {int(k) : v for k,v in tmp_df.to_dict()['mastery'].items()}

        response = call_db("pals", QueryBank.get_available_topic_ids(from_db, subject_id, substrand_id, student_level, get_name=True))
        topic_id_list = response['topic_id'].to_list()

        available_topics = []
        for t in topic_id_list:
            tmp_res = call_db("pals", QueryBank.get_num_quests_in_topics(from_db, [t], student_level=student_level, branch_names=branch_names))['num'].to_list()
            if max(tmp_res) >= const.MIN_QUESTS_IN_TOPIC: available_topics+=[t]
        
        topic_id2name = {topic_id_list[i]:name for i, name in enumerate(response['topic_name'].to_list())}
        topic_id_list = [i for i in topic_id_list if i in available_topics]
        
        _x = { k:"unknown" if not k in tmp_df.keys() else tmp_df[k] for k in topic_id_list }
        _y = {k: 2 if v is None or type(v) == str else 1 if v < 0.50 else 3 if v < 0.79  else 5 for k, v in _x.items()}
        _z = {k:"bad" if v < 2 else "unknown" if v == 2 else "normal" if v==3 else "good"  for k,v in _y.items()}
        tmp_df = pd.DataFrame({"mastery":_x, "type":_y, "category" :_z}).sort_values(by=['type', "mastery"])
        #get top K
        topic_ids = list(tmp_df.index)[:top_k]
        if len(topic_ids) == 0:
            LOGGER.error(f"[ERR] kt_get_weakest_topics, Lenght of topic_ids = 0 with student_id={student_id}, subject_id={subject_id}, substrand_id={substrand_id}")
            raise Exception("Length topic_ids = 0! DB connection is gone!")
        mastery_scores = list(tmp_df['mastery'])[:top_k]
        categories = list(tmp_df["category"])[:top_k]
        output = {
            "weakest_topic_ids":topic_ids,
            "ranks": [i+1 for i in range(len(topic_ids))],
            "mastery": mastery_scores,
            "categories": categories,
            "confidence": 90 if r['complete'] else 0
        }

        #mics
        ####get learning path for student
        learning_path =  None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            parent_conn_default, child_conn_default = mp.Pipe()   
            if subject_id in const.KGsupported_subjects:
                executor.submit(lp_from_list_of_topicids_by_kg, topic_ids, student_level, subject_id, student_id, from_db, from_sys, child_conn_default)
            else:
                # topic_dict = {k:v for k,v in topic_id2name.items() if _z[k] != "good"}
                topic_dict = {k:v for k,v in topic_id2name.items()}
                if len(topic_dict) > 0:
                    executor.submit(lp_from_list_of_topicname, topic_dict, from_db, from_sys, False, child_conn_default)
            conn_ouput = parent_conn_default.recv()
            learning_path = conn_ouput['paths']
            
            #finetune learning path: remove good topic, unlock topic if student meets it's reqs
            if not learning_path is None and not learning_path['reqs'] is None:
                unlock_topics = set([k for k,v in tmp_df['category'].to_dict().items() if v == "good" or v== "normal"])
                all_unlockable_topics = unlock_topics.copy()
                for t in unlock_topics: 
                    all_unlockable_topics = all_unlockable_topics | set(learning_path['reqs'][t])
                # learning_path['unlockable_topics'] = {k:True if k in all_unlockable_topics or len(set(learning_path['reqs'][k]) & all_unlockable_topics)==len(learning_path['reqs'][k]) else False for k in learning_path['seq_path']}
                tmp_dict={}
                for k in topic_ids:
                    flag = False
                    if k in learning_path['pal_path'][0] or k==learning_path['seq_path'][0]: flag=True
                    elif len(learning_path['reqs'][k]) == 0: flag=True
                    elif k in all_unlockable_topics: flag = True
                    elif len(set(learning_path['reqs'][k]) & all_unlockable_topics) > len(learning_path['reqs'][k]) // 2: flag=True

                    tmp_dict[k] = flag
                learning_path['unlockable_topics'] = tmp_dict

            else: 
                LOGGER.error("[ERR] in api_kt_interaction cannot generate learning path by KG")
        
        output['learning_paths'] = learning_path
        return JSONResponse({
            "complete" : True,
            "msg" : output,
            "execution time" : time.time() - tstart
        })
    except Exception as e:
        LOGGER.error(e, "error in api_kt_interaction.kt_get_weakest_topics | inputs={}, {}, {}".format(student_id, substrand_id, student_level))
        return JSONResponse({
            "complete" : False,
            "msg" : output,
            "execution time" : time.time() - tstart
        })
