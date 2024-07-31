from fastapi import APIRouter
from fastapi.responses import JSONResponse
import time
import pandas as pd
from utils import LOGGER
from chatgpt.prompt import PROMPT_DICT
from db_integration import call_db, QueryRnDBank, QueryBank
import json
router = APIRouter(
    prefix="/analysis",
    tags=['(6.x) Performance analysis']
)


def __explore_fail_question_by_strategy__(student_id:int, topic_ids:list, from_db:int=-1, student_level=None, subject_id=None,):
    failed_quests = call_db("pals", QueryBank.get_strategy_failed_question(db_id=from_db, student_id=student_id, topic_ids=topic_ids))
    skills = {}
    def get_skill(value:int, skills:dict):
        if value < 1: return
        if value in skills.keys() : skills[value] +=1
        else: skills[value]=1
    for row in failed_quests.iterrows():
        get_skill(row[1]['substrategy_id'], skills)
        get_skill(row[1]['substrategy_id2'], skills)


    return skills


# def __explore_fail_question_by_content__(student_id:int, topic_ids:list, from_db:int=-1, student_level=None, subject_id=None, taxonomy=None):
#     resp = call_db("pals", QueryBank.get_content_failed_question(db_id=from_db, student_id=student_id,topic_ids=topic_ids))
#     # print(resp)
#     if len(resp) == 0: return []
#     failed_quests = resp['question_text'].to_list()
#     subject_name  = resp['subject_name'].to_list()[0]
#     return PROMPT_DICT.prompt_extract_concept_from_questions(question_texts=failed_quests, subject_name=subject_name, taxonomy_skills=taxonomy)


# @router.get("/missconcept/topic", summary="with an input topic, we will find concepts the student fail to solve at most")
# async def find_misconcepts_topic(student_id:int, student_level:int, topic_id:int, subject_id:int, from_db:int=-1):
#     tstart = time.time()
#     try:
#         if subject_id in call_db("pals", QueryBank.get_subject_id_having_strategy(db_id=from_db))['subject_id'].to_list():
#             # Sort the dictionary by values in descending order
#             skills = __explore_fail_question_by_strategy__(student_id=student_id, topic_ids=[topic_id], from_db=from_db, subject_id=subject_id, student_level=student_level)
#             if len(skills) == 0: return JSONResponse({"complete" : True, "msg": {},   "execution time" : time.time() - tstart})
#             output = {'substrategy_id':{}}
#             for substrategy in list(skills.keys())[:5]:
#                 # query = QueryBank.get_substrategy_info(substrategy=substrategy)
#                 # tmp_df = call_db("smartjen", query)
#                 # output['substrategy_id'][int(substrategy)] = {
#                 #     'name': str(tmp_df['substrategy_name'][0]),
#                 #     'strategy_id': int(tmp_df['strategy_id'][0]),
#                 #     'strategy_name': str(tmp_df['strategy_name'][0])
#                 # }
#                 output['substrategy_id'] += [substrategy]
#         elif subject_id in [1]:
#             with open("./static/Etaxonomy.json", 'r') as json_file:
#                 Etaxonomy = json.load(json_file)
#                 Etaxonomy = "\n".join([f"{k} : {v}" for k,v in Etaxonomy.items()])
#                 mistakes = __explore_fail_question_by_content__(student_id, topic_ids=[topic_id], from_db=from_db, subject_id=subject_id, student_level=student_level,
#                                                             taxonomy=Etaxonomy)
#                 output = {'type_of_question':mistakes}

#         else :
#             mistakes = __explore_fail_question_by_content__(student_id, topic_ids=[topic_id], from_db=from_db, subject_id=subject_id, student_level=student_level)
#             output = {'type_of_question':mistakes}
#         return JSONResponse({"complete" : True, "msg": output,   "execution time" : time.time() - tstart})
    
#     except Exception as e:
#         LOGGER.exception(e)
#         return JSONResponse({'complete':False, "msg":{} })
