#############SIMPLE QUERY###############3
def get_topics_under_ws(db_id:int, worksheet_id:int, student_id:int=None):
    query = f""" 
        SELECT topic_id, question_id, created_at, 
            ROW_NUMBER() OVER (PARTITION BY topic_id, question_id order by created_at DESC) as num_row
        FROM pals_performance_score
        WHERE 1=1 and pals_dbid={db_id} and worksheet_id={worksheet_id} 
        """
    if not student_id is None:
        query += f"  and user_id={student_id} "

    wrap_query = f"""
    SELECT topic_id, count(*) as num_question from ({query})  as wrap_query
    WHERE num_row = 1 
    GROUP BY topic_id   
    HAVING num_question > 1
    """
    # print(wrap_query)
    return wrap_query

def get_topic_id_from_performance_records(db_id:int, student_id:int=None, subject_id:int=None):
    query = f"""SELECT distinct(p.topic_id) 
        FROM pals_performance_score as p 
        JOIN pals_topics as t ON p.topic_id = t.id
        WHERE p.pals_dbid={db_id} and p.topic_id >= 0  """ #and t.is_disable=0
    
    if not student_id is None: query += f"      and p.user_id={student_id} "
    if not subject_id is None: query += f"      and t.subject_type={subject_id} "
    return query

def get_num_record_by_student(db_id:int, student_id:int, topic_id:int=None, topic_id_list:list=[], subject_id=None):
    topic_id_list += [] if topic_id is None else [topic_id] 
    query = f"""SELECT count(*) as count 
            FROM pals_performance_score as p JOIN pals_topics as t ON p.topic_id = t.id
            WHERE p.pals_dbid={db_id} and p.user_id={student_id}  
            """
    if len(topic_id_list) > 0: 
        top_sent=",".join([str(t) for t in topic_id_list])
        query +=  f"    and p.topic_id in ({top_sent}) "
    if not subject_id is None: 
        query +=   f"   and t.subject_type={subject_id} "
    return query

def get_neibor_topic_under_substrand_of(db_id:int, topic_id):
    return f"SELECT id as topic_id  FROM pals_topics WHERE pals_dbid={db_id} and id > 0 and substrand_id = (SELECT substrand_id FROM pals_topics WHERE id={topic_id} LIMIT 1)"

def get_subject_id_having_strategy(db_id:int):
    return f"""
        SELECT distinct(t.subject_type) as subject_id  
        FROM pals_questions as q JOIN pals_topics as t ON q.topic_id = t.id 
        WHERE q.pals_dbid={db_id} and q.substrategy_id > 0
        """

def get_student_success_answering_rate(db_id:int, student_id:int, topic_ids:list, limit_record:int=25):
    topic_sent = ", ".join([ str(s) for s in topic_ids])
    query = f"""            
        SELECT s.topic_id, s.question_id, s.created_at,  s.sucess_rate,
                CASE WHEN  q.facility_index > 0.7 THEN 3
                    WHEN q.facility_index > 0.4 THEN 2
                    WHEN q.facility_index > 0.0 THEN 1
                    WHEN q.difficulty_level = 'hard' THEN 3
                    WHEN q.difficulty_level = 'easy' THEN 1
                    ELSE 2 END as difficulty_level
        FROM (
            SELECT question_id, topic_id, score / fullmark as sucess_rate, created_at,
                ROW_NUMBER() OVER (PARTITION BY topic_id, question_id order by created_at DESC) as num_row
            FROM pals_performance_score
            WHERE pals_dbid={db_id} and user_id={student_id} and topic_id in ({topic_sent})   
        )  as s
        JOIN pals_questions as q ON q.question_id = s.question_id   
        WHERE q.pals_dbid={db_id} and s.num_row = 1
        ORDER BY s.created_at desc
        LIMIT  {limit_record}    
        """
    wrap_query = f"""
                SELECT w1.difficulty_level, CASE WHEN count(*) > 2 THEN AVG(w1.sucess_rate) ELSE -1 END as avg_sucess_rate FROM ({query}) as w1 
                GROUP BY w1.difficulty_level
                """
    return wrap_query


################################
#       Query with multi-condition
#############################


#1. Question bank

def get_quetion_bank(db_id:int, topic_ids:list, student_id:int, student_level:int, question_levels:list=[], question_type_ids:list=[], 
                    prioritized_topics:list=[], prioritized_skills:list=[], branch_names:list=[],
                    filter_pendingmark:bool=True, excluded_quests:list=[], **kwargs):
    
    topic_sent = ",".join([str(t) for t in topic_ids])
    branch_names = ",".join([f"'{t}'" for t in branch_names if t.strip() != ""] )
    question_levels = ",".join([str(t) for t in question_levels])
    question_type_ids = ",".join([str(t) for t in question_type_ids])
    additional_selection = " ,".join(["topic_id", "topic_id2"])

    # if use_strategy:
    substrategy_sent = ",".join([str(t) for t in prioritized_skills+[-99]])
    prioritized_topic_sent   = ",".join([str(t) for t in prioritized_topics+[-99] + topic_ids])
    excluded_quest_sent   = ",".join([str(t) for t in excluded_quests+[-99]])

    query = f"""SELECT  q.reference_id as reference_id, q.question_id as question_id, ROUND(q.facility_index, 1) as rnd_facility_index,
                    CASE WHEN  q.facility_index > 0.7 THEN 3
                        WHEN q.facility_index > 0.4 THEN 2 
                        WHEN q.facility_index > 0.0 THEN 1
                        WHEN q.difficulty_level = 'hard' THEN 3 
                        WHEN q.difficulty_level = 'easy' THEN 1 
                        ELSE 2 END as difficulty_level, 
                    CASE
                        WHEN   (( q.topic_id in ({prioritized_topic_sent}) and q.topic_id2 in ({prioritized_topic_sent}) ) 
                            or  ( q.topic_id in ({prioritized_topic_sent}) and q.topic_id3 in ({prioritized_topic_sent}) )  
                            or  ( q.topic_id in ({prioritized_topic_sent}) and q.topic_id4 in ({prioritized_topic_sent}) ) 
                            or  ( q.topic_id2 in ({prioritized_topic_sent}) and q.topic_id3 in ({prioritized_topic_sent}) ) 
                            or  ( q.topic_id2 in ({prioritized_topic_sent}) and q.topic_id4 in ({prioritized_topic_sent}) ) 
                            or  ( q.topic_id3 in ({prioritized_topic_sent}) and q.topic_id4 in ({prioritized_topic_sent}) ) ) and  q.facility_index=0
                            THEN 2.5
                        WHEN    ( q.topic_id in ({prioritized_topic_sent}) and q.topic_id2 in ({prioritized_topic_sent}) ) 
                            or  ( q.topic_id in ({prioritized_topic_sent}) and q.topic_id3 in ({prioritized_topic_sent}) )  
                            or  ( q.topic_id in ({prioritized_topic_sent}) and q.topic_id4 in ({prioritized_topic_sent}) ) 
                            or  ( q.topic_id2 in ({prioritized_topic_sent}) and q.topic_id3 in ({prioritized_topic_sent}) ) 
                            or  ( q.topic_id2 in ({prioritized_topic_sent}) and q.topic_id4 in ({prioritized_topic_sent}) ) 
                            or  ( q.topic_id3 in ({prioritized_topic_sent}) and q.topic_id4 in ({prioritized_topic_sent}) ) 
                            THEN 2
                        WHEN    (q.substrategy_id in ({substrategy_sent})  or q.substrategy_id2 in ({substrategy_sent})) and  q.facility_index=0   THEN 1.5
                        WHEN    (q.level_id={student_level}  or q.level_id2={student_level} or q.level_id3={student_level}) THEN 1
                        ELSE 0 
                    END as priority, 
                    CASE    WHEN topic_id in ({topic_sent})  THEN topic_id
                            WHEN topic_id2 in ({topic_sent}) THEN topic_id2
                            ELSE topic_id3 
                            END as topic_id,
                    CASE    WHEN topic_id in ({topic_sent}) AND topic_id2 in ({topic_sent}) THEN topic_id2 
                            ELSE 0 
                            END as topic_id2

            FROM pals_questions as q 
            LEFT OUTER JOIN (SELECT question_id, worksheet_id FROM pals_performance_score 
                            WHERE pals_dbid={db_id} and user_id = {student_id}
                            )  as s
            ON q.question_id = s.question_id
            WHERE q.pals_dbid={db_id} and q.disabled = 0  and s.worksheet_id is NULL 
                and (q.topic_id in ({topic_sent}) 
                    or q.topic_id2 in ({topic_sent}) 
                    or q.topic_id3 in ({topic_sent}) 
                    or q.topic_id4 in ({topic_sent}) ) 
                and ( q.level_id={student_level}  or q.level_id2={student_level} or q.level_id3={student_level} 
                        or q.level_id={student_level-1}  or q.level_id2={student_level-1}) 
                and not q.reference_id in ({excluded_quest_sent}) 
            """

    if filter_pendingmark:
        query += "       and ( q.question_type_id NOT IN (10, 13, 12)  or q.answer_type_id <> 16 or NOT (q.question_type_id = 7 and q.answer_type_id in (2,6) ))  " 
    if question_levels != "":
        query += f"      and  q.question_level in ({question_levels})  "
    if question_type_ids != "":
        query += f"      and  q.question_type_id in ({question_type_ids})   "
    if branch_names != "":
        query += f"      and q.branch_name in ({branch_names})  "

    query += " ORDER BY rand() LIMIT 5000"
    #------WRAPPED QUERY TO ETL DATA FROM RAW DATA------------
    wrap_query = f"""
        SELECT reference_id as question_id, avg(rnd_facility_index) as rnd_facility_index,  avg(priority) as priority, topic_id ,topic_id2 ,
            CASE WHEN avg(difficulty_level) > 2.4 THEN 3
                WHEN avg(difficulty_level) > 1.4 THEN 2
                ELSE 1
            END as difficulty_level
        FROM (SELECT reference_id, question_id, rnd_facility_index, difficulty_level, priority ,
                    CASE WHEN topic_id > topic_id2 THEN topic_id ELSE topic_id2 END as topic_id,
                    CASE WHEN topic_id > topic_id2 THEN topic_id2 ELSE topic_id END as topic_id2,
                    ROW_NUMBER() OVER(PARTITION BY topic_id, topic_id2, difficulty_level ORDER BY rand()) as row_num
            FROM ({query}) as zz
            ) as tt
        WHERE row_num < 50
        GROUP BY reference_id
        ORDER BY difficulty_level desc, priority desc, rnd_facility_index 
        """
  
    return wrap_query

def get_failed_question_bank(db_id:int, topic_ids:list, student_id:int, question_levels:list, question_type_ids:list, branch_names:list=[], filter_pendingmark:bool=False):
    topic_sent = ",".join([str(t) for t in topic_ids])
    branch_names = ",".join([f"'{t}'" for t in branch_names]) if len(branch_names) > 0 else ""
    question_levels = ",".join([str(t) for t in question_levels]) if len(question_levels) > 0 else ""
    question_type_ids = ",".join([str(t) for t in question_type_ids]) if len(question_type_ids) > 0 else ""
    # system to recommend question under selected topic / substrand that student still answer it Wrongly
    query = f"""SELECT t.question_id, CASE WHEN q.difficulty_level = 'hard' THEN 3 
                                        WHEN q.difficulty_level = 'easy' THEN 1 
                                        ELSE 2 END as difficulty_level, 
                        ROUND(q.facility_index, 1) as rnd_facility_index, 
                        0 as priority   
                FROM (SELECT question_id, (score / fullmark) as percent, pals_dbid
                        FROM pals_performance_score 
                        WHERE pals_dbid={db_id} and user_id = {student_id}  and topic_id in ({topic_sent}) ) as t
                JOIN pals_questions as q ON q.question_id=t.question_id AND  q.pals_dbid = t.pals_dbid
                WHERE q.disabled=0 and q.question_id = q.reference_id  """       
    #filter
    if filter_pendingmark:
        query += "      and ( q.question_type_id NOT IN (10, 13, 12)  or q.answer_type_id <> 16 or NOT (q.question_type_id = 7 and q.answer_type_id in (2,6) ))  " 
    if question_levels != "":
        query += f"      and  q.question_level in ({question_levels})  "
    if question_type_ids != "":
        query += f"      and  q.question_type_id in ({question_type_ids})   "
    if branch_names != "":
        query += f"      and q.branch_name in ({branch_names})  "

    query += " GROUP BY t.question_id HAVING avg(t.percent) < 0.85 ORDER BY rand() LIMIT 3000 "
    return query





#2. Checking
def get_available_topic_ids(db_id:int, subject_id:int=None, substrand_id:int=None, student_level:int=None, extra_levels:list=[], get_name:bool=False, get_ssid:bool=False, substrand_id_list:list=[]):
    features = " topic_id "
    if get_name: features += ", name as topic_name "
    if get_ssid: features += ", substrand_id as substrand_id "

    query = f"""SELECT {features} 
                FROM (SELECT id as topic_id, 
                            primary_level_1 as grade_level_1, primary_level_2 as grade_level_2, primary_level_3 as grade_level_3, 
                            primary_level_4 as grade_level_4, primary_level_5 as grade_level_5, primary_level_6 as grade_level_6, 
                            level_7 as  grade_level_7, level_8 as grade_level_8,  level_9 as grade_level_9, level_10 as grade_level_10, 
                            level_11 as grade_level_11,level_12 as  grade_level_12, level_13 as grade_level_13, 
                            substrand_id, 
                            subject_type,
                            name 
                       FROM pals_topics 
                       WHERE  pals_dbid={db_id}) as t  """


    query += " WHERE 1=1 "
    if not subject_id is None:
        query += f"""  and subject_type={subject_id} """
    if not substrand_id is None: 
        query += f" and   substrand_id={substrand_id}"
    elif len(substrand_id_list) > 0:
        substrand_sent = ", ".join([str(x) for x in substrand_id_list])
        query += f" and   substrand_id in ({substrand_sent})"
    if not student_level is None:
        query += f" and  grade_level_{student_level} = 1   " + "   ".join([f" or grade_level_{l} = 1 " for l in extra_levels if 14 > l > 0])
    return query


def get_num_quests_in_topics(db_id:int, topic_list:list, student_level:int, filter_pendingmark:bool=False, branch_names:list=[]):
    topic_sent = ",".join([str(t) for t in topic_list])
    branch_sent = ",".join([f" '{t}'" for t in branch_names]) if len(branch_names) > 0 else ""
    query = f"""SELECT count(*) as num FROM pals_questions 
                WHERE pals_dbid={db_id} and disabled = 0 and question_id = reference_id
                    and ( topic_id in ({topic_sent}) or topic_id2 in ({topic_sent}) 
                        or topic_id3 in ({topic_sent}) or topic_id4 in ({topic_sent}) ) 
                    and ( level_id={student_level}  or level_id2={student_level} or level_id3={student_level}) 
            """  
    if filter_pendingmark:
        query += "      and ( question_type_id NOT IN (10, 13, 12)  or answer_type_id <> 16 or NOT (question_type_id = 7 and answer_type_id in (2,6) ))  " 
    if branch_sent != "":
        query += f"      and branch_name in ({branch_sent})"
    return query

#################################
#        Analysis Query
################################
def analyze_top_weak_topic(db_id:int, student_id, subject_id):
    query = f"""SELECT p.topic_id as topic_id, Round(count(distinct(p.question_id)) / (count(*) + 1e-6), 1) as success_rate,
                    sum(CASE WHEN q.difficulty_level ='hard' THEN 3
                             WHEN q.difficulty_level ='easy' THEN 1
                        ELSE 2 END ) AS psuedo_point
            FROM pals_performance_score  as p
            JOIN pals_questions as q ON p.question_id = q.question_id AND  p.pals_dbid=q.pals_dbid
            JOIN pals_topics    as t ON p.topic_id    = t.id      AND  p.pals_dbid=t.pals_dbid
            WHERE t.pals_dbid={db_id} and p.user_id={student_id} and t.subject_type={subject_id} and p.score/p.fullmark < 0.8
                AND (p.created_at BETWEEN DATE_SUB(NOW(), INTERVAL 60 DAY) AND NOW())
            GROUP BY p.topic_id
    """
    return query

# def get_content_failed_question(db_id:int, student_id:int, topic_ids:list):
#     topic_sent = ",".join([str(t) for t in topic_ids])
#     query = f"""SELECT question_text, subject_name 
#                 FROM (SELECT q.question_text, t.subject_id, t.db_id 
#                     FROM pals_questions as q 
#                     JOIN (SELECT question_id, score, fullmark, db_id, topic_id 
#                             FROM pals_performance_score 
#                             WHERE pals_dbid={db_id} and user_id={student_id} and  topic_id in ({topic_sent})  
#                             ORDER BY created_at desc LIMIT 35) as p 
#                 ON q.id = p.question_id AND p.pals_dbid=q.db_id
#                 JOIN pals_topics    as t ON p.topic_id= t.id AND  p.pals_dbid=t.db_id
#                 WHERE p.score < p.fullmark
#             ) as df 
#             JOIN pals_subjects as s ON s.id = df.subject_id AND s.pals_dbid=df.db_id                
#             """
#     return query

def get_strategy_failed_question(db_id:int, student_id:int, topic_ids:list):
    topic_sent = ",".join([str(t) for t in topic_ids])
    query = f""" SELECT substrategy_id, substrategy_id2 FROM pals_questions as q 
                JOIN (SELECT question_id, score, fullmark, pals_dbid FROM pals_performance_score 
                        WHERE pals_dbid = {db_id} and user_id={student_id} and  topic_id in ({topic_sent})  
                        ORDER BY created_at desc LIMIT 25) as p 
                ON q.question_id = p.question_id AND p.pals_dbid=q.pals_dbid
                WHERE score/fullmark < 0.8 """
    return query

def get_concept_failed_worksheet(db_id:int, student_id:int, worksheet_id:int, topic_id:int=None):
    query = f""" SELECT distinct(p.question_id), q.substrategy_id, q.substrategy_id2, 0 as rnd_content_group, q.pals_dbid
                FROM pals_performance_score as p 
                JOIN pals_questions as q ON p.question_id = q.question_id AND p.pals_dbid=q.pals_dbid
                WHERE p.pals_dbid={db_id} and p.worksheet_id={worksheet_id} and p.user_id={student_id} and p.score / p.fullmark < 0.8
                """
    if not topic_id is None:
        query += f"""     and (q.topic_id={topic_id} or q.topic_id2={topic_id} or q.topic_id3={topic_id} or q.topic_id4={topic_id} )"""
    return query 

def get_failed_topics_by_worksheet(db_id:int, student_id:int, worksheet_id:int, topic_list:list=None):
    
    # query = f""" SELECT p.question_id, p.topic_id FROM pals_performance_score as p 
    #             WHERE p.worksheet_id={worksheet_id} and p.user_id={student_id} and p.score / p.fullmark < 1 and p.topic_id <> 0 
    #             ORDER BY RAND()
    #             """
    query = f"""
    SELECT topic_id, question_id from ( 
        SELECT p.topic_id, p.question_id, created_at, score, fullmark,
            ROW_NUMBER() OVER (PARTITION BY topic_id, question_id order by created_at DESC) as num_row
        FROM pals_performance_score as p
        WHERE 1=1 and pals_dbid={db_id}
            and p.worksheet_id={worksheet_id}
        and p.user_id={student_id} )  as wrap_query
    WHERE num_row = 1 and score / fullmark < 0.8  
    """
    if not topic_list is None:
        topic_sent = ",".join([str(t) for t in topic_list])
        query += f"    and  topic_id in ({topic_sent})   "
    return query + "  ORDER BY RAND()"

def get_questinfo_in_worksheet(db_id, worksheet_id:int):
    query = f"""
    SELECT question_id, topic_id 
    FROM pals_performance_score
    WHERE 1=1 and pals_dbid={db_id}
        and worksheet_id={worksheet_id}
    GROUP BY question_id, topic_id
    """
    return query

def get_topics_with_filter_numquestion(db_id, subject_id, student_level, min_num_quest=50, branch_names:list=[], 
                                    filter_pendingmark:bool=True, question_levels:list=[], question_type_ids:list=[]):
    branch_names = ",".join([f"'{t}'" for t in branch_names if t.strip() != ""] ) if len(branch_names) > 0 else ""
    question_levels = ",".join([str(t) for t in question_levels])
    question_type_ids = ",".join([str(t) for t in question_type_ids])
    sub_query =  {
        1: f"""
        SELECT pals_dbid, topic_id as topic_id , count(*) as num FROM pals_questions 
        WHERE topic_id > 0 AND pals_dbid={db_id} AND subject_type={subject_id}  AND (level_id={student_level}  or level_id2={student_level} or level_id3={student_level})  and question_id = reference_id 
        
        """,
        2: f"""
        SELECT pals_dbid,  topic_id2 as topic_id , count(*) as num FROM pals_questions 
        WHERE topic_id2 > 0 AND pals_dbid={db_id} AND subject_type={subject_id}  AND (level_id={student_level}  or level_id2={student_level} or level_id3={student_level})  and question_id = reference_id
        
        """,
        3: f"""
        SELECT pals_dbid, topic_id3 as topic_id , count(*) as num FROM pals_questions 
        WHERE topic_id3 > 0 AND pals_dbid={db_id} AND subject_type={subject_id}  AND (level_id={student_level}  or level_id2={student_level} or level_id3={student_level})  and question_id = reference_id
        
        """,
        4: f"""
        SELECT pals_dbid, topic_id4 as topic_id , count(*) as num FROM pals_questions 
        WHERE topic_id4 > 0 AND pals_dbid={db_id} AND subject_type={subject_id}  AND (level_id={student_level}  or level_id2={student_level} or level_id3={student_level})  and question_id = reference_id
        
        """
    }
    for k in sub_query:
        if branch_names != "":
            sub_query[k] += f"    AND branch_name in ({branch_names})   "
        if filter_pendingmark:
            sub_query[k] += "     AND ( question_type_id NOT IN (10, 13, 12)  or answer_type_id <> 16 or NOT (question_type_id = 7 and answer_type_id in (2,6) ))  " 
        if question_levels != "":
            sub_query[k] += f"    AND  question_level in ({question_levels})  "
        if question_type_ids != "":
            sub_query[k] += f"    AND question_type_id in ({question_type_ids})   "

        sub_query[k] += "         AND disabled = 0 "

    query = f"""
    
        SELECT t.topic_id, sum(t.num) as num, c.substrand_id
                FROM (
                        {sub_query[1]}
                        GROUP BY topic_id

                        UNION ALL
                        {sub_query[2]}
                        GROUP BY topic_id2
                        
                        UNION ALL
                        {sub_query[3]}
                        GROUP BY topic_id3
                        
                        UNION ALL
                        {sub_query[4]}
                        GROUP BY topic_id4
                    ) as t
        JOIN pals_topics as c ON c.id = t.topic_id AND t.pals_dbid = c.pals_dbid
        WHERE num >= {min_num_quest}
        GROUP BY t.topic_id 
    """
    return query