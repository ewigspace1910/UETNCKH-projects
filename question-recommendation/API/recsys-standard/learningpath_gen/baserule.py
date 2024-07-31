from typing import List
import requests
import pandas as pd
import json

from const import TMPROOT, MASTERY_LEVEL, URL_KTmodel
from utils import  check_kt_ingredient, LOGGER
from logics.logic_graph import  extract_neighbor, topic_filters, find_relevant_topic
from db_integration import call_db, QueryBank, QueryRnDBank
from chatgpt.prompt import PROMPT_DICT


def lp_from_list_of_topicname(topic_dict:dict, from_db:int, from_sys:int=0, use_requirement_constrain=False, conn=None, **kwargs):
    try:
        advice = PROMPT_DICT.prompt_order_topics2sequence(topic_dict)
    except Exception as e:
        LOGGER.error(e, "error in api_learning_path.lp_from_list_of_topicname | topic_ids={}".format(";".join(list(topic_dict.keys()))))
        advice=None

    advice = {"name":"GPTpath", "paths":advice}
    if not conn is None:
        conn.send(advice)
        conn.close()
    return  advice



#use LLMxKG
def lp_from_list_of_topicids_by_llmkg(topic_ids:List[int], student_level:int, student_id:int=-1, from_db:int=-1, from_sys:int=0, conn=None):
    try:
        advice = None
        pass
    except Exception as e:
        LOGGER.error(e, "error in api_learning_path.lp_from_list_of_topicids | topic_ids={}".format(";".join(topic_ids)))
        advice = None
        
    advice = {"name":"AIpath", "paths":advice}
    if not conn is None:
        conn.send(advice)
        conn.close()
    return advice

def lp_from_list_of_topicids_by_kg(topic_ids:List[int], student_level:int, subject_id:int, student_id:int=-1, from_db:int=-1, from_sys:int=0, conn=None):
    try:
        topic_dict={}
        topic_ids_set = set(topic_ids)
        for topic_id in sorted(topic_ids):
            # reqs= find_relevant_topic(topic_id, subject_id=subject_id,student_id=student_level, student_level=student_level, rel_types=['require'], from_db=from_db)
            reqs= extract_neighbor(from_db, subject_id, topic_id, relation_types=['require'])
            reqs=reqs['topic_id'].tolist()
            topic_dict[topic_id] = set(reqs) & topic_ids_set ##just consider reqs belonging topics under the same
        for topic_id, reqs in topic_dict.items():
            for t in topic_dict[topic_id]:
                reqs = set(reqs) | set(topic_dict[t])
            topic_dict[topic_id] = reqs

        # if student_id > 0:
        #     #summary each topic
        #     query = QueryRnDBank.get_mastery_records(student_id, topic_list=topic_ids, from_db=from_db, from_sys=from_sys)
        #     response = call_db("rnd", query)
        #     groups = response.set_index('topic_id')
        #     mastery_scores= groups.mastery.to_dict()
        #     print(mastery_scores)
        reqs_dict  = {k:list(v) for k, v in topic_dict.items()}
        topic_dict = {k:len(v) for k, v in topic_dict.items()}
        topic_dict =  {k: v for k, v in sorted(topic_dict.items(), key=lambda item: item[1])}
        stage_dict =  {}
        for key, value in topic_dict.items():  stage_dict.setdefault(value, []).append(key)
        stage_dict = {idx:stage_dict[k] for idx, k in enumerate(stage_dict.keys())}
        advice = {"seq_path":list(topic_dict.keys()), 
                "pal_path": stage_dict,
                "reqs"    : reqs_dict}
    except Exception as e:
        LOGGER.error(e, "error in api_learning_path.lp_from_list_of_topicids | topic_ids={}".format(";".join(topic_ids)))
        try:
            flag, advice = False, {
                'seq_path' : topic_ids,
                'pal_path' : {idx:[tid]for idx, tid in enumerate(topic_ids)},
                'reqs'     : None
            }
        except : flag, advice = False, None        
    
    advice = {"name":"KGpath", "paths":advice}
    if not conn is None:
        conn.send(advice)
        conn.close()
    return advice