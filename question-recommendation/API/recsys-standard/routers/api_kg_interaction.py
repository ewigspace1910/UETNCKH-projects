from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import requests
import pandas as pd
import json
from typing import List
from const import TMPROOT, MASTERY_LEVEL, URL_KTmodel, MIN_QUESTS_IN_TOPIC
from utils import  check_kt_ingredient, LOGGER
from logics.logic_graph import  extract_neighbor, topic_filters
from db_integration import QueryBank, call_db
from itertools import combinations
router = APIRouter(
    prefix="/kg",
    tags=['(4.x) Knowledge graph']
)

EMPTY_TOPIC_LIST = JSONResponse({"topics":[], "ranks":[]})

@router.get('/find_next_topic_4student')
def find_next_of_topic(topic_id:int, subject_id:int, student_level:int, student_id:int, from_db:int, from_sys:int=0): 
    try:
        # graph = get_graph(subject_id, root=TMPROOT) 
        if subject_id not in [2,1]: raise Exception(f"Cannot get graph of subject {subject_id}")
        # check records
        r = check_kt_ingredient(subject_id=subject_id, student_id=student_id, topic_id=topic_id, from_db=from_db)
        mastery = "unknow"
        if r <= 0:
            # else get estimate
            r = requests.get(URL_KTmodel['infer_user_topic'], 
                            params ={"student_id":student_id, "subject_id":subject_id, 'topic_id':topic_id, 
                                     'from_db':from_db, 'from_sys':from_sys})
            r = json.loads(r.text)
            if r['complete']:  
                # Categorize the student into 3 group
                mastery = pd.DataFrame(r['body'])
                if "topic_id" in mastery.columns: mastery = mastery.set_index("topic_id")['mastery'][topic_id]
                else: mastery = mastery['mastery'][topic_id]

        if  not type(mastery) == str and mastery <= MASTERY_LEVEL["b1"]:
            edges_df = extract_neighbor(from_db, subject_id, topic_id, relation_types=['require', 'relevant'], min_weight=0, student_level=student_level)
            sorttype = 3
        else:
            edges_df = extract_neighbor(from_db, subject_id, topic_id, relation_types=['reverse of require', 'relevant'], min_weight=0, student_level=student_level)
            sorttype = 1
        
        edges_df = topic_filters(edges_df, student_id, subject_id, student_level, [student_level-1], sort_type=sorttype, from_db=from_db)
        
        return JSONResponse({"complete": True,
                            "topics": edges_df['topic_id'][:10].tolist(), 
                            "ranks": [i+1 for i in range(len(edges_df[:10]))]})
    
    except Exception as e:
        LOGGER.exception(e, "error in api_kg_interaction.find_next_topic | inputs={}, {}, {}".format(student_id, topic_id, student_level))
        return JSONResponse({"complete": False,
                            "topics": [], 
                            "ranks": []})
    

@router.get('/get-topic-reqs', summary="Inputs require a list of topics in the same substrand_id")
def get_req_of_topic(from_db:int, subject_id:int, substrand_id:int,  student_level:int, topic_ids:List[int]= Query(), branch_names:List[str]= Query()): 
    try:
        
        query = QueryBank.get_available_topic_ids(db_id=from_db, subject_id=subject_id, substrand_id=substrand_id, student_level=student_level)
        topics_under_substrand = set(call_db("pals", query)['topic_id'].to_list())
        available_topics=[]
        for t in topics_under_substrand:
            num = call_db("pals", QueryBank.get_num_quests_in_topics(db_id=from_db, topic_list=[t], student_level=student_level, branch_names=branch_names))['num']
            if max(num) >= MIN_QUESTS_IN_TOPIC: available_topics += [t]
        topics_under_substrand = [t for t in topics_under_substrand if t in available_topics]
        reqs = {}
        for tid in  topic_ids:
            req = extract_neighbor(db_id=from_db, subject_id=subject_id, topic_id=tid, student_level=student_level, relation_types=['require'])['topic_id'].to_list()
            req = set(req) & topics_under_substrand
            reqs[tid] = list(req)

        return JSONResponse({"complete": True,
                            "msg": reqs})

    except Exception as e:
        LOGGER.exception(e, "error in api_kg_interaction.get_topic_reqs | inputs={}, {}, {}".format(substrand_id, topic_ids, student_level))
        return JSONResponse({"complete": False,
                            "msg": None})



    

def order_substrand_under_strand(substrand_dict:dict):
    """
    @param substrand_dict: a dictionary with keys are substrand_ids and values are their topic dicts. 
                        The topic dict include topics and correspoding requirements
                        example: {substrand_1 : {
                                                    1 : {0},
                                                    2 : {1},
                                                    3 : {2, 1}
                                                }
                                substrand_2: {...}}
    @return : a dictionary contains the learning stages. Each stage contain topics the student can leanrt concurently. 
    """
    def __estimate_substrand_relationship(topics_reqs_ss1:dict, topics_reqs_ss2:dict):
        """ Example: keys = topic_ids, values = their topic requirements.
            topics_reqs_ss1 = {
                1 : {0},
                2 : {1},
                3 : {2, 1}
            }
            topics_reqs_ss2 = {
                4:  {2,3},
                5 : {4,2,6},
                6 : {2, 1}
            }"""
        out_dict = {1: topics_reqs_ss1, 2: topics_reqs_ss2}
        for i in topics_reqs_ss1.keys(): 
            out_dict[1][i] = len(topics_reqs_ss1[i] & set(topics_reqs_ss2.keys())) * 3  #3 is score of require relationship
        for i in topics_reqs_ss2.keys(): 
            out_dict[2][i] = len(topics_reqs_ss2[i] & set(topics_reqs_ss1.keys())) * 3


        ss1_coef = sum(out_dict[1].values())
        ss2_coef = sum(out_dict[2].values())
        relative_weight_ss1_ss2 = (1+ss1_coef) / (1+ss2_coef) 
        # print(ss1_coef / 3, "/", ss2_coef / 3)
        if  relative_weight_ss1_ss2  < 1/3:
            #print("ss2 require ss1" )
            return 'is required by'
        elif relative_weight_ss1_ss2 > 3/1:
            #print("ss1 require ss2")
            return 'require'
        else:
            # print("can learn in parallel")
            return None
        

    substrand_rel = {k:[] for k in substrand_dict}
    for (substrand_id1, substrand_id2) in list(combinations(substrand_dict.keys() , 2)):
        topics_reqs_ss1 = substrand_dict[substrand_id1].copy()
        topics_reqs_ss2 = substrand_dict[substrand_id2].copy()

        output = __estimate_substrand_relationship(topics_reqs_ss1, topics_reqs_ss2)
        # print("ratio of the number of cross-require relationship between  ", (substrand_id1, substrand_id2) , 'is = ', end='\t')
        if output is None: pass
        elif output == "is required by": substrand_rel[substrand_id2].append(substrand_id1) 
        elif output == "require": substrand_rel[substrand_id1].append(substrand_id2)

    #need to check
    # print(substrand_rel)
    substrand_dict = {k:len(v) for k, v in substrand_rel.items()}
    substrand_dict =  {k: v for k, v in sorted(substrand_dict.items(), key=lambda item: item[1])}
    stage_dict =  {}
    for key, value in substrand_dict.items():  stage_dict.setdefault(value, []).append(key)
    stage_dict = {idx:stage_dict[k] for idx, k in enumerate(stage_dict.keys())}
    return stage_dict, substrand_rel


@router.get('/order-substrands', summary="Inputs require a list of subtrands in KG")
def order_substrands_api(from_db:int, subject_id:int, student_level:int, substrand_ids:List[int]=Query()):
    try:
        topic_bank = call_db("pals", QueryBank.get_available_topic_ids(from_db=from_db, subject_id=subject_id, 
                                        student_level=student_level, substrand_id_list=substrand_ids, get_ssid=True))
        substrand_dict = {}
        topic_set = set(topic_bank['topic_id'].to_list())
        for idx, row in topic_bank.iterrows():
            tid = int(row['topic_id'])
            ssid = int(row['substrand_id'])
            reqs =  set(extract_neighbor(db_id=from_db, subject_id=subject_id, topic_id=tid, relation_types=['require'])['topic_id'].to_list()) & topic_set
            if not ssid in substrand_dict :substrand_dict[ssid] = {tid:reqs}
            else :substrand_dict[ssid][tid] = reqs 

        substrand_orders, substrand_rel = order_substrand_under_strand(substrand_dict)
        reqs = {
            "learning_paths": substrand_orders,
            "reqs"      : substrand_rel
        }
        return  JSONResponse({"complete": True,
                                "msg": reqs})
    except Exception as e:
        LOGGER.error(f"[ERR] error in api_kg_interation.order_substrands_api as {str(e)}")
        return  JSONResponse({"complete": True,
                        "msg": None})
    


@router.get('/filter-topics', summary="returns a list of topics-substrands with a sufficient number of questions")
def filter_topic_api(from_db:int, subject_id:int, student_level:int,  min_num_quest=MIN_QUESTS_IN_TOPIC, branch_names:List[str]=Query(['']), 
                     question_levels:List[int]=Query([]), question_type_ids:List[int]=Query([]), filter_pendingmark:bool=True):
    try:
        query = QueryBank.get_topics_with_filter_numquestion(from_db, subject_id, student_level, min_num_quest=min_num_quest, branch_names=branch_names,
                                                            question_levels=question_levels, question_type_ids=question_type_ids, filter_pendingmark=filter_pendingmark)
        topic_bank:pd.DataFrame = call_db("pals", query)
        # print(topic_bank)
        topic_dict = topic_bank.set_index('topic_id')['substrand_id']
        substrand_dict = {}
        for key, value in topic_dict.items():  substrand_dict.setdefault(value, []).append(key)
        substrand_dict = {k:v for k, v in substrand_dict.items() if len(v) > 0}
        return  JSONResponse({"complete": True,
                                "msg": substrand_dict})
    except Exception as e:
        LOGGER.error(f"[ERR] error in api_kg_interation.order_substrands_api as {str(e)}")
        return  JSONResponse({"complete": True,
                        "msg": None})