import json
import pandas as pd
import random
import requests
import math
from utils import LOGGER, check_kt_ingredient, check_model_config
from const import URL_KTmodel, URL_QUERY, MIN_QUESTS_IN_TOPIC, MASTERY_LEVEL
from routers.api_analysis import __explore_fail_question_by_strategy__
from db_integration import call_db, QueryBank, QueryRnDBank



############## SOME FILTERs based on logic ##########
def filter_topics_by_gradelevel(topic_list:list, student_level:int, filter_pendingmark:bool=False, from_db:int=-1):
    new_topic = []
    for topic in topic_list:
        query = QueryBank.get_num_quests_in_topics(db_id=from_db, topic_list=[topic], student_level=student_level, filter_pendingmark=filter_pendingmark)
        question_bank:pd.DataFrame = call_db("pals", query, to_json=True)
        num = list(question_bank['num'].values())[0]
        if num >= MIN_QUESTS_IN_TOPIC:
            new_topic += [topic]
    return new_topic




############## CREATE QUESTION BANK ############## 
def get_custome_question_bank(topic_ids:list, student_id:int, student_level:int, num_question:int, 
                            question_levels:list, question_type_ids:list, from_db:int=-1, branch_names:list=[],
                            filter_pendingmark:bool=True, prioritized_topics:list=[], excluded_quests:list=[]):
                    
    if len(topic_ids) == 0: topic_ids = [-99]
    prioritized_skills = __explore_fail_question_by_strategy__(student_id=student_id, topic_ids=topic_ids, from_db=from_db, student_level=student_level)        
    prioritized_skills = list(prioritized_skills.keys())[:5]       
    query = QueryBank.get_quetion_bank(db_id=from_db, topic_ids=topic_ids, student_id=student_id, student_level=student_level, question_levels=question_levels,
                            question_type_ids=question_type_ids, branch_names=branch_names, filter_pendingmark=filter_pendingmark, 
                            prioritized_skills=prioritized_skills, prioritized_topics=prioritized_topics, excluded_quests=excluded_quests)
    # print(query)
    question_bank:pd.DataFrame = call_db("pals", query)
    # print(query)
    # print(question_bank)
    ##########################################
    # get extra question
    if len(question_bank) < num_question:
        query = QueryBank.get_failed_question_bank(db_id=from_db, topic_ids=topic_ids, student_id=student_id, question_levels=question_levels,
                                        question_type_ids=question_type_ids, branch_names=branch_names, filter_pendingmark=filter_pendingmark)
        question_bank_extra = call_db("pals", query)
        question_bank = pd.concat([question_bank, question_bank_extra], axis=0)
    
    question_bank.drop_duplicates(subset=["question_id"], inplace=True)
    question_bank['question_id'] = question_bank['question_id'].astype(int)
    return question_bank


def get_custome_question_bank_based_content(topic_ids:list, student_id:int, student_level:int, num_question:int, 
                            prioritized_skills:list, rnd_group_id:list, prioritized_topics:list=[], excluded_quests:list=[],
                            question_levels:list=[], question_type_ids:list=[], from_db:int=-1, branch_names:list=[],
                            filter_pendingmark:bool=True, ):
                    
    if len(topic_ids) == 0: topic_ids = [-99]
       
    query = QueryBank.get_quetion_bank(db_id=from_db, topic_ids=topic_ids, student_id=student_id, student_level=student_level, question_levels=question_levels,
                            question_type_ids=question_type_ids, branch_names=branch_names, filter_pendingmark=filter_pendingmark, 
                            prioritized_skills=prioritized_skills, prioritized_topics=prioritized_topics, excluded_quests=excluded_quests, rnd_concepts=rnd_group_id)
    question_bank:pd.DataFrame = call_db("pals", query)
    ##########################################
    # get extra question
    if len(question_bank) < num_question:
        query = QueryBank.get_failed_question_bank(db_id=from_db, topic_ids=topic_ids, student_id=student_id, question_levels=question_levels,
                                        question_type_ids=question_type_ids, branch_names=branch_names, filter_pendingmark=filter_pendingmark)
        question_bank_extra = call_db("pals", query)
        question_bank = pd.concat([question_bank, question_bank_extra], axis=0)
    
    question_bank.drop_duplicates(subset=["question_id"], inplace=True)
    return question_bank

####################################
#           RULEBASE
#
def difficulity_distribution_rulebase(topic_success_rate):
    """topic_success_rate as {hard(3):0, norm(2):0.25, easy(1):0.4} with value -1, it means that DB has no record for this type of question
    @return: rate of easy, rate of normal, rate of hard question
    """
    rh = topic_success_rate[3] #hard rate
    rn = topic_success_rate[2]
    re = topic_success_rate[1]
    if rh==rn==re==-1: rh, rn, re = 0.2, 0.4, 0.4
    elif (re <= 0.5 and re!=-1) and (rn < 0.5 or rh < 0.5): rh, rn, re = 0, 0.2, 0.8
    elif (re < 0.9 and re!=-1) and (rn < 0.5 or rh < 0.5): rh, rn, re = 0, 0.6, 0.4
    elif (re >= 0.9 or re==-1) and rn <= 0.5: rh, rn, re = 0, 0.6, 0.4
    elif (re >= 0.9 or re==-1) and rn < 0.75: rh, rn, re = 0.1, 0.8, 0.1
    elif (re >= 0.9 or re==-1) and (0.9 > rn >= 0.75 or rn == -1) and rh <= 0.5: rh, rn, re = 0.4, 0.6, 0
    elif (rn >= 0.9 or rn==-1) and rh <= 0.5: rh, rn, re = 0.6, 0.4, 0
    elif (rn >= 0.9 or rn==-1) and rh > 0.5: rh, rn, re = 0.8, 0.2, 0
    #special cases
    elif ((re < 1 and re!=-1)) and rh > 0.5: rh, rn, re = 0.3, 0.4, 0.3
    elif ((rn < 1 and rn!=-1)) and rh > 0.5: rh, rn, re = 0.5, 0.5, 0.
    else:
        print("THERE IS A CASE CANNOT IS NOT COVERED --> (hard|norm|easy)= ",rh, rn, re )
        rh, rn, re = 0.2, 0.5, 0.3
    return re, rn, rh

def gen_ratio_based_mastery(mastery):
    cur_level = -1 if type(mastery) == str or mastery< 0 \
            else 6 if mastery >= MASTERY_LEVEL['c2'] \
            else 5 if mastery >= MASTERY_LEVEL['c1'] \
            else 4 if mastery >= MASTERY_LEVEL['b2'] \
            else 3 if mastery >= MASTERY_LEVEL['b1'] \
            else 2 if mastery >= MASTERY_LEVEL['a2'] \
            else 1 #advance - upper immediate - lower immediate - beginer

    if cur_level == 6:   ratios = 0, 0.2, 0.8  #easy-normal-hard
    elif cur_level == 5: ratios = 0, 0.4, 0.6 
    elif cur_level == 4: ratios = 0, 0.7, 0.3
    elif cur_level == 3: ratios = 0.1, 0.8, 0.1
    elif cur_level == 2: ratios = 0.4, 0.6, 0
    elif cur_level == 1: ratios = 0.8, 0.2, 0
    else: ratios = 0.4, 0.4, 0.2
    return ratios

def __question_selection__(ratios, qbank):
    """Ratios is distribution of difficulty level [easy, norm, hard]"""
    re, rn, rh = ratios
    _question_choices = {}
    _question_choices["hard"]   = qbank[qbank['difficulty_level']==3]['question_id'].to_list()[:rh] 
    
    if len(_question_choices["hard"]) < rh: rn += rh-len(_question_choices["hard"]) 
    _question_choices["norm"] = qbank[qbank['difficulty_level']==2]['question_id'].to_list()[:rn]
    
    if len(_question_choices["norm"]) < rh: re += rn-len(_question_choices["norm"])
    _question_choices["easy"]   = qbank[qbank['difficulty_level']==1]['question_id'].to_list()[:re]
    return _question_choices

#####Diagnostic test when mastery = -1
def gen_questions_base_mastery(student_id:int, student_level:int, topic:int, mastery:float, num_question:int, from_db:int=-1,
                               is_teacher_side:bool=False, branch_names:list=[], question_levels:list=[0], question_type_ids:list=[1], 
                               priority_ratio:float=0.5, quest_distribution:dict=None, prioritized_topics:list=[], excluded_quests:list=[]):
    # question bank
    question_bank = get_custome_question_bank(topic_ids=[topic], student_id=student_id, student_level=student_level, 
                                    num_question=num_question, branch_names=branch_names, question_levels=question_levels, 
                                    question_type_ids=question_type_ids,
                                    filter_pendingmark= not is_teacher_side, from_db=from_db,
                                    prioritized_topics=prioritized_topics, excluded_quests=excluded_quests)
    question_choices = {}
    # num_question =  8 if num_question < 8 else num_question 
    ratios = gen_ratio_based_mastery(mastery)
    #fintune ratios by check the recent resutls of students: If student didnt make mistake in dealing with easy questions ---> dont return easy questions 
    def __finetune__ratios__(student_id, topic_ids, default_ratios:list, from_db:str):
        try:
            success_rate = call_db("pals", QueryBank.get_student_success_answering_rate(db_id=from_db, student_id=student_id, topic_ids=topic_ids)).set_index("difficulty_level")['avg_sucess_rate'].to_dict()
            success_rate = [float(success_rate[1]) if 1 in success_rate else -1, float(success_rate[2]) if 2 in success_rate else -1, float(success_rate[3]) if 3 in success_rate else -1]
            for i in range(len(success_rate)-1, 0, -1): success_rate[i-1] =  success_rate[i] if i > 0 and success_rate[i] > success_rate[i-1] and success_rate[i-1] > -1 else success_rate[i-1] 
            
            erate, nrate, hrate = success_rate
            eratio, nratio, hratio = default_ratios
            if 1 > nrate >= 0.7 and hratio < 0.6: #should set erate =0 , reduce nrate 1/2 and increase hrate to satisfy erate+nrate+hrate = 1
                eratio, nratio = eratio / 2, max(nratio * 1.5, 0.4)
                hratio          = 1 - eratio - nratio 
            elif (erate == -1 or erate > 0.7)  and -1 < nrate < 0.7:  #should set hrate = 0, reduce erate to 2/3 initial value, increase nrate to satisfy erate+nrate+hrate = 1
                eratio, hratio  =  min(eratio, 0.2), min(max(hratio/2, 0.1), 0.4)
                nratio          =  1 - eratio - hratio
            elif -1 < erate < 0.7 : #should set hrate = 0, reduce nrate to 1/3 initial value, increase erate to satisfy erate+nrate+hrate = 1
                nratio, hratio = max(nratio/2, 0.2),  0
                eratio         = 1 - nratio - hratio
            return [eratio, nratio, hratio]
        except:
            LOGGER.error(f"[ERR] when finetune ratios of WS of student_id: {student_id}")
            return default_ratios

    #select prioritized questions
    if quest_distribution is None: 
        ratios = gen_ratio_based_mastery(mastery)
        ratios = __finetune__ratios__(student_id, topic_ids=[topic], default_ratios=ratios, from_db=from_db)
    else: ratios = [quest_distribution['easy'], quest_distribution['norm'], quest_distribution['hard']]

    expected_ratios = [int(x * num_question + 0.5) for x in ratios]
    ratios4prioritizedQuests  =  [int(x * 0.4) for x in expected_ratios]
    question_choices = __question_selection__(ratios= ratios4prioritizedQuests, qbank=question_bank[question_bank['priority'] > 0.5])

    ratios4otherQuests = [max(r - len(question_choices[k]), 0) for r, k in zip(expected_ratios, question_choices.keys())]
    question_choices_non_priority = __question_selection__(ratios=ratios4otherQuests, qbank=question_bank[question_bank['priority'] <= 0.5])
    for  k in question_choices: question_choices[k] += question_choices_non_priority[k]

    cur_len = sum([len(question_choices[x]) for x in question_choices])
    if  cur_len < num_question:
        selected_question = question_choices["hard"] + question_choices["norm"] + question_choices["easy"]
        the_left_over = list(set(question_bank['question_id'].to_list()) | set(selected_question))
        random.shuffle(the_left_over)
        question_choices['random'] = the_left_over[:num_question-cur_len+1]
    else: question_choices['random'] = []
    return question_choices


def gen_questions_base_masteryNworksheet(student_id:int, student_level:int, topic:int, mastery:float, worksheet_id:int, num_question:int, from_db:int=-1,
                               is_teacher_side:bool=False, branch_names:list=[], question_levels:list=[0], question_type_ids:list=[1], 
                               priority_ratio:float=0.8, quest_distribution:dict=None, prioritized_topics:list=[], excluded_quests:list=[]):
    failed_concepts = call_db("pals", QueryBank.get_concept_failed_worksheet(db_id=from_db, student_id=student_id, worksheet_id=worksheet_id, topic_id=topic))
    list_failed_strategy, list_faild_rnd_group = set(), set()
    for idx, row  in failed_concepts.iterrows():
        list_failed_strategy |= set([row.substrategy_id, row.substrategy_id2])
        list_faild_rnd_group |= set([0])#set([row.rnd_group_id])
    list_faild_rnd_group = list(list_faild_rnd_group-{0, -1}) 
    list_failed_strategy = list(list_failed_strategy-{0, -1})
    # question bank
    question_bank = get_custome_question_bank_based_content(topic_ids=[topic], student_id=student_id, student_level=student_level, 
                                    prioritized_skills=list_failed_strategy, rnd_group_id=list_faild_rnd_group, #content control
                                    num_question=num_question, branch_names=branch_names, question_levels=question_levels, 
                                    question_type_ids=question_type_ids,
                                    filter_pendingmark= not is_teacher_side, from_db=from_db,
                                    prioritized_topics=prioritized_topics, excluded_quests=excluded_quests)
    question_choices = {}
    if quest_distribution is None:
        ratios = gen_ratio_based_mastery(mastery)
        expected_ratios = [int(x * num_question + 0.5) for x in ratios]
    else: expected_ratios = [quest_distribution['easy'], quest_distribution['norm'], quest_distribution['hard']]

    
    #select prioritized questions
    question_choices = __question_selection__(ratios=expected_ratios, qbank=question_bank)

    #add more question if not enough question in each categories as expected 
    cur_len = sum([len(question_choices[x]) for x in question_choices])
    if  cur_len < num_question:
        selected_question = question_choices["hard"] + question_choices["norm"] + question_choices["easy"]
        the_left_over = list(set(question_bank['question_id'].to_list()) | set(selected_question))
        random.shuffle(the_left_over)
        question_choices['random'] = the_left_over[:num_question-cur_len+1]
    else: question_choices['random'] = []
    return question_choices

########################## WORKSHEET GENERATION ########################3
def gen_ws_4_topic(student_id:int, student_level:int, topic_id:int, subject_id:int, num_question:int, is_teacher_side:bool, 
                   from_db:int, from_sys:int,
                   branch_names:list=["SmartJen"], question_levels:list=[0], question_type_ids:list = [1], 
                   conn=None):
    LOGGER.log(1, msg="Gen WS 4 student_id = {}".format(student_id))
    output = {student_id:{topic_id:{"mastery-score":0, "ws":None}}}
    # check records
    r = check_kt_ingredient(subject_id=subject_id, student_id=student_id, topic_id=topic_id, from_db=from_db)
    if r > 0:
        #student have not enought valid data--> diagnotic test
        output[student_id][topic_id]['mastery-score'] = "unknown"
        output[student_id][topic_id]['ws'] = gen_questions_base_mastery(student_id=-1, student_level=student_level, topic=topic_id, 
                                                                        mastery=-1, num_question=num_question,is_teacher_side=is_teacher_side, 
                                                                        branch_names=branch_names, question_levels=question_levels,
                                                                        question_type_ids=question_type_ids, from_db=from_db)
    else:
        # else get estimate
        r = requests.get(URL_KTmodel['infer_user_topic'], params ={"student_id":student_id, "subject_id":subject_id, 'topic_id':topic_id, "from_db":from_db, 'from_sys':from_sys})
        r = json.loads(r.text)
        if not r['complete']:  raise Exception(r['body'])
        # Categorize the student into 3 group
        tmp_df = pd.DataFrame(r['body'])
        if "topic_id" in tmp_df.columns: mastery = tmp_df.set_index("topic_id")['mastery'][topic_id]
        else: mastery = tmp_df['mastery'][topic_id]
        output[student_id][topic_id]['mastery-score'] = round(mastery, 2)
        
        # Question choice
        question_choices=gen_questions_base_mastery(student_id, student_level, topic_id, mastery, num_question,from_db, is_teacher_side,
                                                branch_names=branch_names,
                                                question_levels=question_levels, question_type_ids=question_type_ids)
        
        output[student_id][topic_id]['ws'] = question_choices
    
    if conn is None:
        return output
    else:
        conn.send(output)
        conn.close()

#For only teacher side
def gen_ws_4_substrand(student_id:int, student_level:int, substrand_id:int, subject_id:int, top_k:int, num_question:int, 
                    branch_names:list=[], question_levels:list=[0], question_type_ids:list=[0], from_db:int=-1, from_sys:int=0,  conn=None):
    output = {student_id:{}}   
    
    try:
        #find top-k weakest topic 4 new student
        topics_in_substrand  = call_db("pals", 
                            query=QueryBank.get_available_topic_ids(db_id=from_db, subject_id=subject_id, substrand_id=substrand_id, student_level=student_level, extra_levels=[student_level-1])) 
        seen_topic = topics_in_substrand['topic_id'].to_list()        

        r = call_db("rnd", QueryRnDBank.get_mastery_records(student_id=student_id, substrand_id=substrand_id, from_db=from_db, from_sys=from_sys))
        
        r = r.dropna()
        excluded_quests = []
        if len(r) == 0:
            top_k_weakest = filter_topics_by_gradelevel(seen_topic, student_level=student_level, from_db=from_db)
            if len(top_k_weakest) == 0:    
                output[student_id] = "This subtrand have no topic/question at same student level"
            
            else:
                random.shuffle(top_k_weakest)
                for topic_id in top_k_weakest[:top_k]:
                    #estimate the number of question in each cateogry
                    topic_success_rate = call_db("pals", QueryBank.get_student_success_answering_rate(db_id=from_db, student_id=student_id, topic_ids=[topic_id], limit_record=20)).set_index("difficulty_level")['avg_sucess_rate'].to_dict()
                    topic_success_rate = {int(k):v for k, v in topic_success_rate.items()}
                    for i in range(1,4,1): topic_success_rate[i] = -1 if not i in topic_success_rate else topic_success_rate[i]

                    diff_ratio = difficulity_distribution_rulebase(topic_success_rate)
                    quest_distribution = {"easy":diff_ratio[0], 
                                            'norm':diff_ratio[1], 
                                            'hard':diff_ratio[2]}
           
                    output[student_id][topic_id] = {}
                    output[student_id][topic_id]['mastery-score'] = "unknown"
                    output[student_id][topic_id]['ws'] = qchoices =gen_questions_base_mastery(student_id=-1, student_level=student_level, topic=topic_id, 
                                                                                   mastery=-1, num_question=num_question,is_teacher_side=True,
                                                                                   branch_names=branch_names, question_levels=question_levels, 
                                                                                   question_type_ids=question_type_ids, from_db=from_db,
                                                                                   quest_distribution=quest_distribution,
                                                                                   prioritized_topics=top_k_weakest[:top_k],excluded_quests=excluded_quests) 
                    excluded_quests += qchoices["hard"] + qchoices["norm"] + qchoices["easy"] + qchoices["random"] 
        else:
            # top_k_weakest = r['topic_id'].to_list()
            top_k_weakest = {int(k):v for k, v in r.set_index('topic_id')['mastery'].to_dict().items()}
            # valid_topic = filter_topics_by_gradelevel(top_k_weakest, student_level=student_level, from_db=from_db)
            # if len(valid_topic) == 0:    
            #     output[student_id] = "This subtrand have no topic/question at same level of this student" 
            # else:
            valid_topic = seen_topic
            # mastery_topic = [x for i, x in enumerate(r['mastery'].to_list())]
            topic2mastery = {k:top_k_weakest[k] if k in top_k_weakest else 0.4 for k in valid_topic}
            valid_topic = sorted(topic2mastery.items(), key=lambda x: x[-1])
            #choose top-k + 1 extra --> categorize --> choose question
            for topic_id, mastery in enumerate(valid_topic[:top_k] + random.choice(valid_topic[top_k:]) ):
                output[student_id][topic_id] ={}
                output[student_id][topic_id]['mastery-score'] = mastery
                output[student_id][topic_id]['ws'] = qchoices = gen_questions_base_mastery(student_id, student_level, topic_id, mastery=mastery, 
                                                                                num_question=num_question, is_teacher_side=True, 
                                                                                branch_names=branch_names, question_levels=question_levels, 
                                                                                question_type_ids=question_type_ids, from_db=from_db,
                                                                                prioritized_topics=valid_topic,excluded_quests=excluded_quests) 
                excluded_quests += qchoices["hard"] + qchoices["norm"] + qchoices["easy"] + qchoices["random"] 
    except Exception as e:
        e = LOGGER.exception(str(e))
    
    if conn is None:
        return output
    else:
        conn.send(output)
        conn.close()


########################## For Extended worksheet generation ########################3
def adjust_quest_dist_based_failed_questions(new_num_question:int, org_question_dist:dict, failure_rate:dict=None):
    """both dicts should be in the format as {<topic_id>:<value>}"""
    #convert the org-->softmax distribution
    org_sum, org_exp_dist = 0, {} 
    for k, v in org_question_dist.items():
        org_exp_dist[k] = v #math.exp(v)
        org_sum += v #math.exp(v)
    org_exp_dist = {k:(v/org_sum) for k,v in org_exp_dist.items()}

    #convert failure rate --> softmax distribution
    failure_sum, failure_exp_dist = 0, {} 
    if not failure_rate is None:
        for k, v in failure_rate.items():
            failure_exp_dist[k] = v#math.exp(v)
            failure_sum += v#math.exp(v)
        failure_exp_dist = {k:(v/failure_sum) for k,v in failure_exp_dist.items()}
    else: failure_exp_dist = org_exp_dist

    #merge 2 distribution by EMA (exponential moving avarage)
    momentum = 0.75
    new_distr = {}
    for k in org_question_dist:
        ratio = (1-momentum) * org_exp_dist[k] + momentum * failure_exp_dist[k]
        new_distr[k] = int(round(ratio * new_num_question + 0.25, 0))
    return new_distr


def gen_ws_4_substrand_based_on_givenWS(worksheet_id:int, student_id:int, student_level:int, num_question:int, keep_NquestDistr: bool=True,
                    branch_names:list=["SmartJen"], question_levels:list=[0], question_type_ids:list=[0], from_db:int=-1, from_sys:int=0,  conn=None):
    output = {student_id:{}}   
    
    try:
        topicNquest_under_ws = call_db("pals", query=QueryBank.get_topics_under_ws(db_id=from_db,worksheet_id=worksheet_id))
        questInfo_under_ws  =  call_db("pals", query=QueryBank.get_questinfo_in_worksheet(db_id=from_db, worksheet_id=worksheet_id))
        org_question_dist  = topicNquest_under_ws.set_index("topic_id")['num_question'].to_dict()

        tmp_check_question = []
        #because a question could be tagged by mul-topics--> the distribtion should be modified
        org_question_dist   = dict(sorted(org_question_dist.items(), key=lambda x: x[-1], reverse=True))
        old_num_quest_of_ws       = len(set(questInfo_under_ws['question_id'].to_list()))
        # The above code is a Python script that is attempting to print the value of the variable
        # `old_num_quest_of_ws`. However, the variable `old_num_quest_of_ws` is not defined in the
        # code snippet provided, so it will result in a NameError.
        for k, nquest in org_question_dist.items():
            quest_of_k= questInfo_under_ws[questInfo_under_ws['topic_id']==int(k)]['question_id'].to_list()
            real_len_quest_of_k = len(set(quest_of_k) - set(tmp_check_question))
            if  real_len_quest_of_k > 0:
                org_question_dist[k] = max(1, min(real_len_quest_of_k, old_num_quest_of_ws))
            else:
                org_question_dist[k] = 1
            old_num_quest_of_ws -= real_len_quest_of_k
            tmp_check_question += quest_of_k
        
        try:
            failed_topic_df = call_db("pals", query=QueryBank.get_failed_topics_by_worksheet(db_id=from_db, student_id=student_id, worksheet_id=worksheet_id, topic_list=list(org_question_dist.keys())))
            failed_topic = []
            for t in  failed_topic_df['topic_id'].to_list(): 
                if not t in failed_topic: failed_topic += [t]  
        except Exception as e:
            print(f"[!!!ERROR] can find failed topic of the student {student_id} with worksheet {worksheet_id}")
        #redefine the number of question for each topic based on the orginal distribution and result of the student
        if keep_NquestDistr:
            #keep the distribution of the number questions between topics
            failure_rate_student_in_WS = None
        else:
            failure_rate_student_in_WS = {}
            for idx, row in failed_topic_df.iterrows():
                failure_rate_student_in_WS[row.topic_id]  = 1 if not row.topic_id in failure_rate_student_in_WS else 1 + failure_rate_student_in_WS[row.topic_id]
        num_quest_of_topics = adjust_quest_dist_based_failed_questions(num_question, org_question_dist, failure_rate_student_in_WS)
            
        #gen new questions
        if True: 
            #the students dont have record in RnD DB and subjuect without KT
            excluded_quests = []
            for topic_id in num_quest_of_topics:
                #estimate the number of question in each cateogry
                nqt = int(num_quest_of_topics[topic_id])
                topic_success_rate = call_db("pals", QueryBank.get_student_success_answering_rate(from_db, student_id, [topic_id], limit_record=10)).set_index("difficulty_level")['avg_sucess_rate'].to_dict()
                topic_success_rate = {int(k):v for k, v in topic_success_rate.items()}
                for i in range(1,4,1): topic_success_rate[i] = -1 if not i in topic_success_rate else topic_success_rate[i]

                diff_ratio = difficulity_distribution_rulebase(topic_success_rate)
                quest_distribution = {"easy":int(diff_ratio[0] * nqt + 0.5), 
                                      'norm':int(diff_ratio[1] * nqt + 0.5), 
                                      'hard':int(diff_ratio[2] * nqt + 0.5)}

                #gen question
                output[student_id][topic_id] = {}
                output[student_id][topic_id]['ws'] = qchoices =gen_questions_base_masteryNworksheet(student_id=student_id, student_level=student_level, topic=topic_id, 
                                                                                mastery=-1, worksheet_id=worksheet_id, 
                                                                                num_question=nqt, 
                                                                                is_teacher_side=True,
                                                                                branch_names=branch_names, question_levels=question_levels, 
                                                                                question_type_ids=question_type_ids, from_db=from_db,
                                                                                quest_distribution=quest_distribution,
                                                                                prioritized_topics=list(num_quest_of_topics.keys()), excluded_quests=excluded_quests
                                                                                )
                excluded_quests += qchoices["hard"] + qchoices["norm"] + qchoices["easy"] + qchoices["random"]

                
    except Exception as e:
        e = LOGGER.exception(str(e))
    
    if conn is None:
        return output
    else:
        conn.send(output)
        conn.close()