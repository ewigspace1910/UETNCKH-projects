################SIMPLE GET QUERY #########
def get_model_config_link(category, model_name, use_like=False):
    model_name = f" like '{model_name}' " if use_like else f" ='{model_name}' "
    query = f"""SELECT link, model_name, extention  FROM sj_model_config_link 
                WHERE category='{category}' and model_name {model_name}
            """
    
    return query

############## READ QUERY############33
def get_last_update_date_in_mastery(student_id, subject_id, from_db="original", from_sys:int=0):
    return f"SELECT max(date) as max_date  FROM sj_student_mastery WHERE user_id={student_id} and subject_id={subject_id} and from_db='{from_db}' and from_sys={from_sys} "

def get_tmp_mastery_records(student_id:int, topic_id:int, time_interval=7, from_db='original', from_sys:int=0):
    query = f"""SELECT mscore * 100 as mastery from sj_student_mastery_tmp  
                WHERE updated BETWEEN DATE_SUB(NOW(), INTERVAL {time_interval} DAY) AND NOW()  
                        AND user_id={student_id} AND topic_id={topic_id} and from_db='{from_db}' """
    return query

def get_mastery_records(student_id:int, topic_list:list=None, substrand_id:int=None, subject_id:int=None, from_db='original', from_sys:int=0):
    if not topic_list is None and len(topic_list) > 0:
        topic_sent = ",".join([str(t) for t in topic_list])
        query = f"""SELECT topic_id, mastery FROM sj_student_mastery 
                    WHERE user_id={student_id} 
                        and topic_id in ({topic_sent}) and difficulty_level=-1 and from_db='{from_db}' and from_sys={from_sys} """
    elif not substrand_id  is None:
        query = f"""SELECT p.topic_id,  round(avg(p.mastery),2) as mastery  
                    FROM (SELECT topic_id,  mastery as mastery FROM sj_student_mastery  
                        WHERE user_id={student_id} and substrand_id = {substrand_id} 
                            and difficulty_level=-1 and from_db='{from_db}' and from_sys={from_sys})  as p 
                    GROUP BY p.topic_id ORDER BY mastery"""
    elif not subject_id is None:
        query = f"""
                SELECT topic_id, round(avg(mastery),2) as mastery FROM sj_student_mastery 
                WHERE user_id={student_id} and subject_id = {subject_id}  and from_db='{from_db}' and from_sys={from_sys}
                GROUP BY topic_id """
    else:
        raise "Either topic_list or subsstrand must be not None"
    return query

def get_standard_ws(student_level, subject_id, destination_id, type_id, from_db, update_at:str=None, is_available:int=1):
    query = f"""SELECT ws_id FROM sj_standard_wsb 
                WHERE   is_available={is_available} and from_db='{from_db}' and type_id={type_id} 
                    and destination_id={destination_id} and subject_id={subject_id} and student_level={student_level}
            """
    if not update_at is None : query += f"""   and updated_at='{update_at}'  """
    
    query += """
                ORDER BY RAND()
                LIMIT 1                 
    """
    return query

def get_question_of_fws(ws_id):
    return f"SELECT question_id FROM sj_standard_wsb_question WHERE ws_id={ws_id}"



############## WRITE QUERY############33
def insert_record2table_mastery_tmp(student_id:int, topic_id:int, mastery:float, from_db:str='original', from_sys:int=0):
    return f"""INSERT INTO sj_student_mastery_tmp (user_id, topic_id, mscore, from_db) 
                    VALUES  ({student_id}, {topic_id}, {round(mastery,2)}, from_db='{from_db}')"""

def insert_newWS2table_wsb(type_id:int, destination_id:int, subject_id:int, student_level:int,  updated_at:str, from_db:str='original'):
    query =  f"""INSERT INTO sj_standard_wsb (type_id, destination_id, subject_id, student_level, updated_at, from_db) 
                Values  ({type_id}, {destination_id}, {subject_id}, {student_level}, '{updated_at}', '{from_db}')
            """
    return query
def insert_newWS2table_wsb_question(ws_id:int, question_list:list):
    query =  f"""INSERT INTO sj_standard_wsb_question (ws_id, question_id) Values    """
    for qid in question_list:
        query += f" ({ws_id}, {qid}),"

    query = query[:-1]+";"
    return query
############# DELETE QUERY #############333
def del_record_in_table_mastery_by_date(student_id, subject_id, max_date, from_db:str='original', from_sys:int=0):
    return f"""DELETE FROM sj_student_mastery 
                    WHERE user_id={student_id} and subject_id={subject_id} 
                        and (date is NULL or date <= '{max_date}')
                        and from_db='{from_db} and from_sys={from_sys}' 
            """