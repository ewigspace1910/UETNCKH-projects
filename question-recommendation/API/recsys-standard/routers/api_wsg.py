from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from typing import List
import time
import random
from utils import LOGGER
from logics.logic_wsgen import get_custome_question_bank
from db_integration import call_db, QueryBank, QueryRnDBank
import pandas as pd
import numpy as np
router = APIRouter(
    prefix="/wsg",
    tags=['(5.x) Worksheet Generation']
)

def select_questions_based_on_Diff_n_Topic(df, num_questions, Hratio, Nratio, Eratio):
    """
    df is question_bank including  3 columns : question_id, difficulty_level and topic_id
    num_questions: the number of expected question list
    Hratio, Nratio, Eratio are ratios of Hard-norm-easy question
    """
    # assert df['topic_id'].nunique() <= N, "N must be greater than or equal to the number of unique topics"
    if df['topic_id'].nunique() > num_questions: num_questions = int(df['topic_id'].nunique() * 1.5) 
    
    # Select one question from each topic
    topics = set(df['topic_id'].unique()) | set(df['topic_id2'].unique()) 
    selected_questions = []
    for topic in topics:    
        if topic == 0: continue
        question = df[df['topic_id'] == topic]
        if len(question) == 0:
            question = df[df['topic_id2'] == topic]
        
        question = question.sample(1)
        selected_questions.append(question)
        df = df.drop(question.index)
    # Update the difficulty constraints
    # num_questions -= len(topics)
    tmp_diff_distribution = [ int(x['difficulty_level']) for x in selected_questions]
    Hratio, Nratio, Eratio = [max(0, int(num_questions * x )  - tmp_diff_distribution.count(diff) ) for x, diff in zip([Hratio, Nratio, Eratio], [3, 2, 1])] # number of hard question # number of norml question # number of easy question
    # Select questions based on the updated difficulty constraints
    for difficulty, count in zip([3, 2, 1], [Hratio, Nratio, Eratio]):
        questions = df[df['difficulty_level'] == difficulty][:count]
        selected_questions.append(questions)
        df = df.drop(questions.index)

    # If not enough questions, select from remaining pool
    cur_len = len(pd.concat(selected_questions))
    if cur_len < num_questions:
        remaining = num_questions - cur_len
        selected_questions.append(df[:remaining])

    return pd.concat(selected_questions)

@router.get("/dtest/bytopic", summary="generate diagnotic test (40 easy-40 normal - 20 hard) under a particular topic")
async def gen_dtest_topic(from_db:int, student_id:int, student_level:int, topic_id:int, subject_id:int,
                                question_levels:List[int]=Query([]), question_type_ids:List[int]=Query([]), branch_names:List[str]=Query(["SmartJen"]), 
                                ratios:List[int]=Query([40,40,20]), num_question:int=25):
    
    tstart=time.time()
    try:
        question_bank = get_custome_question_bank(topic_ids=[topic_id], student_id=student_id, student_level=student_level, 
                                        num_question=num_question, branch_names=branch_names, question_levels=question_levels, 
                                        question_type_ids=question_type_ids, from_db=from_db)
    
        if len(ratios) < 3: ratios = 0.4, 0.4, 0.2
        else:
            s = sum(ratios[:3]) 
            ratios = [round(r/s, 1) for r in ratios[:3] ]
        ratios = [int(x * num_question + 0.5) for x in ratios]
        question_choices={}
        question_choices["hard"]   = question_bank[question_bank['difficulty_level']==3]['question_id'].to_list()[:ratios[2]]
        question_choices["normal"] = question_bank[question_bank['difficulty_level']==2]['question_id'].to_list()[:ratios[1]]
        question_choices["easy"]   = question_bank[question_bank['difficulty_level']==1]['question_id'].to_list()[:ratios[0]]
        cur_len = sum([len(question_choices[x]) for x in question_choices])
        if  cur_len < num_question:
            selected_question = question_choices["hard"] + question_choices["normal"] + question_choices["easy"]
            the_left_over = list(set(question_bank['question_id'].to_list()) | set(selected_question))
            random.shuffle(the_left_over)
            question_choices['random'] = the_left_over[:num_question-cur_len+1]
        else: question_choices['random'] = []
        return JSONResponse({"complete" : True, "msg": question_choices,   "execution time" : time.time() - tstart})
    except Exception as e:
        LOGGER.exception(str(e), "error in api_wsg.dtest| inputs={}, {}, {}".format(student_id, topic_id, student_level))
        return JSONResponse({"complete" : False, "msg": str(e),   "execution time" : time.time() - tstart})
    

@router.get("/dtest/bysubstrand", summary="generate diagnotic test (40 easy-40 normal - 20 hard) under a particular topic")
async def gen_dtest_substrand(from_db:int, student_level:int, substrand_id:int, subject_id:int, student_id:int=-1,
                                question_levels:List[int]=Query([]), question_type_ids:List[int]=Query([]), branch_names:List[str]=Query(["SmartJen"]), 
                                ratios:List[int]=Query([40,40,20]), num_question:int=25):
    
    tstart=time.time()
    try:
        query = QueryBank.get_available_topic_ids(db_id=from_db, subject_id=subject_id, substrand_id=substrand_id, student_level=student_level)
        topic_bank = call_db("pals", query)['topic_id'].to_list()
        question_bank = get_custome_question_bank(topic_ids=topic_bank, student_id=student_id, student_level=student_level, 
                                        num_question=num_question, branch_names=branch_names, question_levels=question_levels, 
                                        question_type_ids=question_type_ids, from_db=from_db)
        if len(ratios) < 3: ratios = 0.4, 0.4, 0.2
        else:
            s = sum(ratios[:3]) 
            ratios = [round(r/s, 1) for r in ratios[:3] ]
        question_choices={}
        selected_questions = select_questions_based_on_Diff_n_Topic(question_bank, num_question, ratios[2], ratios[1], ratios[0])
        question_choices["hard"]   = selected_questions[selected_questions['difficulty_level']==3]['question_id'].to_list()
        question_choices["normal"] = selected_questions[selected_questions['difficulty_level']==2]['question_id'].to_list()
        question_choices["easy"]   = selected_questions[selected_questions['difficulty_level']==1]['question_id'].to_list()

        return JSONResponse({"complete" : True, "msg": question_choices,   "execution time" : time.time() - tstart})
    except Exception as e:
        LOGGER.exception(str(e), "error in api_wsg.dtest.substrand| inputs={}, {}, {}, {}".format(student_id, substrand_id, student_level, from_db))
        return JSONResponse({"complete" : False, "msg": str(e),   "execution time" : time.time() - tstart})