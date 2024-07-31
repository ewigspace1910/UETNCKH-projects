from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from typing import List
import json
import time
import requests
import random
import pandas as pd
from const import MASTERY_LEVEL, ERROR, URL_KTmodel
from utils import LOGGER, check_kt_ingredient, check_model_config
from logics.logic_wsgen import gen_questions_base_mastery, gen_ws_4_topic, filter_topics_by_gradelevel
from logics.logic_graph import find_relevant_topic
from db_integration import call_db, QueryBank, QueryRnDBank

router = APIRouter(
    prefix="/ss",
    tags=['(2.x) Practice mode (student side)']
)

##### STUDENT SIDE ########
@router.get("/genquests-topic", summary="generate list of question belonging to a particular topic")
async def ss_gen_new_quest_topic(from_db:int, student_id:int, student_level:int, topic_id:int, 
                                subject_id:int, num_question:int=10, round:int=0,
                                branch_names:List[str]=Query(["SmartJen"]), question_levels:List[int]=Query([]), 
                                question_type_ids:List[int]=Query([]), from_sys:int=0):
    tstart=time.time()
    try:
        #check whether subject_id is supported or not
        if round == 0:
            r = check_kt_ingredient(subject_id=subject_id, topic_id=topic_id, student_level=student_level, filter_pendingmark=True, from_db=from_db)
            if r > 299 : return    {"complete":False, "msg": ERROR[r]}
        #for each student
        output = {"complete":True, "msg":{student_id:{}}}
        ws = gen_ws_4_topic(student_id=student_id, student_level=student_level, topic_id=topic_id, subject_id=subject_id, num_question=num_question, 
                            is_teacher_side=False, branch_names=branch_names, question_levels=question_levels, question_type_ids=question_type_ids,
                            from_db=from_db, from_sys=from_sys)
        output['msg'][student_id] = ws[student_id]

        output['execution time'] = time.time() - tstart  
        return JSONResponse(output)

    except Exception as e:
        e = LOGGER.exception(str(e))
        return JSONResponse({"complete" : False, "msg": str(e),   "execution time" : time.time() - tstart})



@router.get("/genquests-substrand", summary="generate list of question belonging to a substrand, prioritize weakest topics")
async def ss_gen_new_quest_substrand(from_db:int, student_id:int, student_level:int, substrand_id:int, subject_id:int, 
                                    top_k:int=2, num_question:int=3, round:int=0, 
                                    branch_names:List[str]=Query(["SmartJen"]), question_levels:List[int]=Query([]), 
                                    question_type_ids:List[int]=Query([]), from_sys:int=0):
    tstart=time.time()
    # top_k = 50
    is_new_student = False
    output = {"complete":True, "msg":{student_id:{}}}
    top_k_weakest = []
    top_k_mastery = []
    try:
        #check whether subject_id is supported or not
        if round == 0:
            r = check_kt_ingredient(subject_id=subject_id, student_id=student_id, from_db=from_db)
            if r > 299 : return {"complete":False, "msg": "this subject is not supported by this version"}
            elif r==200 : is_new_student = True

        #find top-K topic:
        dataset_cfg = check_model_config(f"subject_{subject_id}_%", from_db)
        supported_topic = [float(k) for k in dataset_cfg['pid2idx'].keys()]
        topics_in_substrand = call_db(db_name="pals", query=QueryBank.get_available_topic_ids(db_id=from_db, subject_id=subject_id, 
                                                            substrand_id=substrand_id, student_level=student_level)) 
        unseen_topic, seen_topic = [], []  #important
        for x in topics_in_substrand['topic_id'].to_list():
            if x in supported_topic: seen_topic.append(x)
            else: unseen_topic.append(x)
        if len(seen_topic) == 0: is_new_student = True #KT model definately not support all topic in this substrand because no record in sj_performance_score
        
        excluded_quests = []
        if is_new_student:
            top_k_weakest = topics_in_substrand['topic_id'].to_list()
            random.shuffle(top_k_weakest)
            top_k_weakest = filter_topics_by_gradelevel(top_k_weakest, student_level=student_level, filter_pendingmark=True, from_db=from_db)
            
            if len(top_k_weakest) == 0: return {"complete":False, "msg": "This subtrand have no topic/question at same student level"}

            
            for topic_id in top_k_weakest[:top_k]:
                output["msg"][student_id][topic_id] ={}
                output["msg"][student_id][topic_id]['mastery-score'] = "unknown"
                output["msg"][student_id][topic_id]['ws'] = qchoices = gen_questions_base_mastery(student_id=-1, student_level=student_level, topic=topic_id, mastery=-1, 
                                                                                    num_question=num_question,is_teacher_side=True, from_db=from_db,
                                                                                    branch_names=branch_names, question_levels=question_levels, question_type_ids=question_type_ids,
                                                                                    prioritized_topics=top_k_weakest[:top_k], excluded_quests=excluded_quests
                                                                                )
                excluded_quests += qchoices["hard"] + qchoices["norm"] + qchoices["easy"] + qchoices["random"]

        else:
            #old student ---> Estimate mastery on all Topic in substrand --> choose top-k weakest topic student learnt and +1 new topic --> Categorize student/chose questions
            r = requests.get(URL_KTmodel['infer_user_topic'], params ={"student_id":student_id, "subject_id":subject_id, 'substrand_id':substrand_id, 'from_db':from_db, 'from_sys':from_sys})
            r = json.loads(r.text)
            if not r['complete']:  raise Exception("KT model raise error")
            tmp_df = pd.DataFrame(r['body'])
            if "topic_id" in tmp_df.columns: tmp_df = tmp_df.set_index("topic_id")
            grouped_df = tmp_df
            grouped_df.sort_values(by=['mastery'], inplace=True)
            
            #choôse weakest topic
            top_k_mastery, top_k_weakest = [], []
            for e in grouped_df['mastery'].index:
                top_k_weakest += [e]
                top_k_mastery += [grouped_df['mastery'][e]]

            valid_topic = filter_topics_by_gradelevel(top_k_weakest, student_level=student_level, filter_pendingmark=True, from_db=from_db)
            if len(valid_topic) == 0:    
                return {"complete":False, "msg": "This subtrand have no topic/question at same student level"}
            else:
                selected_mastery = [x for i, x in enumerate(top_k_mastery) if top_k_weakest[i] in valid_topic] # top_k_weak will have order same valid_topic: [1,2,3,4] & [1, 3, 4] 
            #choose top-k + 1 extra --> categorize --> choose question
            for i, topic_id in enumerate(valid_topic[:top_k]):
                output["msg"][student_id][topic_id] ={}
                output['msg'][student_id][topic_id]['mastery-score'] = selected_mastery[i]
                output['msg'][student_id][topic_id]['ws'] = qchoices = gen_questions_base_mastery(student_id, student_level, topic_id, mastery=top_k_mastery[i], 
                                                                                    num_question=num_question, is_teacher_side=False, from_db=from_db,
                                                                                    branch_names=branch_names, question_levels=question_levels, 
                                                                                    question_type_ids=question_type_ids,
                                                                                    prioritized_topics=top_k_weakest[:top_k], excluded_quests=excluded_quests
                                                                                )
                excluded_quests += qchoices["hard"] + qchoices["norm"] + qchoices["easy"] + qchoices["random"]

        #####################################     

        output['execution time'] = time.time() - tstart
        return JSONResponse(output)

    except Exception as e:
        e = LOGGER.exception(str(e))
        return JSONResponse({"complete" : False, "msg": str(e),   "execution time" : time.time() - tstart})

