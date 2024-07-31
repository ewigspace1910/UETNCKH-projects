from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from typing import List
import json
import time
import requests
import pandas as pd
from const import MASTERY_LEVEL, ERROR, URL_KTmodel
from utils import LOGGER, check_kt_ingredient, check_model_config
from logics.logic_wsgen import gen_ws_4_topic, gen_ws_4_substrand, gen_ws_4_topic, filter_topics_by_gradelevel, gen_ws_4_substrand_based_on_givenWS
from logics.logic_graph import find_relevant_topic
from db_integration import call_db, QueryBank, QueryRnDBank
import concurrent.futures
import multiprocessing as mp

router = APIRouter(
    prefix="/ts",
    tags=['(3.x) Practice mode (Teacher side)']
)

##### TEACHER SIDE ########
@router.post("/gen-ws-topic", summary="generate worksheet for one/group student with a paticular topic")
async def ts_gen_ws_topic(from_db:int, topic_id:int, subject_id:int, student_level:int, student_ids:List[int] = Query(None), num_question:int=10,
                        branch_names:List[str]=Query(["SmartJen"]), question_levels:List[int]=Query([]), question_type_ids:List[int]=Query([]), from_sys:int=0):
    tstart = time.time()
    try:
        #check whether subject_id is supported or not
        r = check_kt_ingredient(subject_id=subject_id, topic_id=topic_id, student_level=student_level, from_db=from_db)
        if r > 299 : return {"complete":False, "msg": ERROR[r]}

        #for each student
        output = {"complete":True, "msg":{}}
        parent_connections = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for student_id in student_ids:
                parent_conn, child_conn = mp.Pipe()
                parent_connections += [parent_conn]      
                executor.submit(gen_ws_4_topic, student_id, student_level, topic_id, subject_id, num_question, True, from_db, from_sys,
                                branch_names, question_levels, question_type_ids,
                                child_conn)
        for parent_connection in parent_connections:
            tmp_dict = parent_connection.recv()
            for name in tmp_dict:
                output["msg"][name]= tmp_dict[name]
            

        # print(output)
        output['execution time'] = time.time() - tstart
        return JSONResponse(output)

    except Exception as e:
        e = LOGGER.exception(str(e))
        return JSONResponse({"complete" : False, "msg": str(e),   "execution time" : time.time() - tstart})



@router.post("/gen-ws-substrand", summary="developing")
async def ts_gen_ws_substrand(from_db:int, substrand_id:int, subject_id:int, student_level:int, student_ids:List[int] = Query(None), top_k:int=2, num_question:int=10, 
                            branch_names:List[str]=Query(["SmartJen"]), question_levels:List[int]=Query([]), question_type_ids:List[int]=Query([]),
                            from_sys:int=0):
    tstart = time.time()
    try:
        #check whether subject_id is supported or not
        r = check_kt_ingredient(subject_id=subject_id, from_db=from_db)
        if r > 299 : return {"complete":False, "msg": ERROR[r]}

        #for each student
        output = {"complete":True, "msg":{}}
        parent_connections = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for student_id in student_ids:
                parent_conn, child_conn = mp.Pipe()
                parent_connections += [parent_conn]      
                executor.submit(gen_ws_4_substrand, student_id, student_level, substrand_id, subject_id, 
                        top_k, num_question, branch_names, question_levels, question_type_ids, from_db, from_sys, child_conn)
        
        for parent_connection in parent_connections:
            tmp_dict = parent_connection.recv()
            for name in tmp_dict:
                output["msg"][name] = tmp_dict[name]
            

        # print(output)
        output['execution time'] = time.time() - tstart
        return JSONResponse(output)

    except Exception as e:
        e = LOGGER.exception(str(e))
        return JSONResponse({"complete" : False, "msg": str(e),   "execution time" : time.time() - tstart})


@router.post("/gen-ws-by-ws", summary="Extended worksheet generation")
async def ts_gen_ws_by_ws(from_db:int, worksheet_id:int, num_question:int, student_level:int, student_ids:List[int] = Query(), keep_NquestDistr:bool=True,
                            branch_names:List[str]=Query(["SmartJen"]), question_levels:List[int]=Query([]), question_type_ids:List[int]=Query([]),
                            from_sys:int=0):
    tstart = time.time()
    try:
        #for each student
        output = {"complete":True, "msg":{}}
        parent_connections = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for student_id in student_ids:
                parent_conn, child_conn = mp.Pipe()
                parent_connections += [parent_conn]      
                executor.submit(gen_ws_4_substrand_based_on_givenWS, worksheet_id, student_id, student_level, 
                        num_question, keep_NquestDistr, branch_names, question_levels, question_type_ids, 
                        from_db, from_sys, child_conn)
        
        for parent_connection in parent_connections:
            tmp_dict = parent_connection.recv()
            for name in tmp_dict:
                output["msg"][name] = tmp_dict[name]
            

        # print(output)
        output['execution time'] = time.time() - tstart
        return JSONResponse(output)

    except Exception as e:
        e = LOGGER.exception(str(e))
        return JSONResponse({"complete" : False, "msg": str(e),   "execution time" : time.time() - tstart})

