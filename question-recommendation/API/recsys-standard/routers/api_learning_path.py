from fastapi import APIRouter, Query
from typing import List
from fastapi.responses import JSONResponse

from utils import   LOGGER
from db_integration import call_db, QueryBank, QueryRnDBank
from learningpath_gen.baserule import lp_from_list_of_topicids_by_kg, lp_from_list_of_topicname



router = APIRouter(
    prefix="/lp",
    tags=["(8.x) Learning path recommendation (non-official) "]
)


@router.get("/from-list-topic-name", summary="create learning path from set of topic name")
async def genfrom_topicnames(from_db:int, topic_ids:List[int]=Query(), topic_names:List[str]=Query()):
    flag = True
    try:
        assert len(topic_ids)==len(topic_names), "Len of topic_ids should be equal to length of topic_names" 
        topic_dict = {k:topic_names[v] for v, k in enumerate(topic_ids)}
        advice = lp_from_list_of_topicname(topic_dict, from_db)
    except Exception as e:
        LOGGER.error(e, "error in api_learning_path.lp_from_list_of_topicname | topic_ids={}".format(";".join(topic_ids)))
        flag = False

    advice = {"name":"normalpath", "paths":advice}
    return JSONResponse({
        "complete" : flag,
        "msg" : advice
    })




@router.get("/from_list_topic_id", summary="create learning path from set of topic id and use LLMxKG")
async def genfrom_topicids(from_db:int, student_level:int, topic_ids:List[int]=Query(), subject_id:int=-1, from_sys:int=0):
    flag = True
    try:
        # advice = PROMPT_DICT.prompt_order_topics2sequence(topic_ids)
        advice = lp_from_list_of_topicids_by_kg(topic_ids, student_level=student_level, subject_id=subject_id, 
                                                student_id=-1, from_db=from_db, from_sys=from_sys)
        pass
    except Exception as e:
        LOGGER.error(e, "error in api_learning_path.lp_from_list_of_topicids_by_kg | topic_ids={}".format(";".join(topic_ids)))
        flag, advice = False, None
        
    advice = {"name":"AIpath", "paths":advice}
    return JSONResponse({
        "complete" : flag,
        "msg" : advice
    })


@router.get("/4target-topic", summary="create learning path for a particular target topic based on student's profile")
async def lp_4targettopic(target_topic, student_id:int, student_level:int, subject_id:int=-1, from_db:str="original", from_sys:int=0):
    pass