################SIMPLE GET QUERY #########
def get_model_config_link(db_id, category, model_name, use_like=False):
    model_name = f" like '{model_name}' " if use_like else f" ='{model_name}' "
    query = f"""SELECT link, model_name,
                    CASE
                        WHEN LOCATE('.', REVERSE(link)) > 0 THEN REVERSE(LEFT(REVERSE(link), LOCATE('.', REVERSE(link) )-1))   
                        ELSE NULL
                    END AS extention
                FROM pals_model_config_link 
                WHERE db_id={db_id} and category='{category}' and model_name {model_name}
            """
    
    return query

############## READ QUERY############33
def get_mastery_records(db_id:int, student_id:int, topic_list:list=None, substrand_id:int=None, subject_id:int=None, from_sys:int=-1):
    if not topic_list is None and len(topic_list) > 0:
        topic_sent = ",".join([str(t) for t in topic_list])
        query = f"""SELECT topic_id, mastery FROM pals_student_mastery 
                    WHERE pals_dbid={db_id} and user_id={student_id} 
                        and topic_id in ({topic_sent})  and sys_id={from_sys} """
    elif not substrand_id  is None:
        query = f"""SELECT p.topic_id,  round(avg(p.mastery),2) as mastery  
                    FROM (SELECT topic_id,  mastery as mastery FROM pals_student_mastery  
                        WHERE pals_dbid={db_id} and user_id={student_id} 
                            and substrand_id = {substrand_id}  
                            and sys_id={from_sys})  as p 
                    GROUP BY p.topic_id ORDER BY mastery"""
    elif not subject_id is None:
        query = f"""
                SELECT topic_id, round(avg(mastery),2) as mastery FROM pals_student_mastery 
                WHERE pals_dbid={db_id} and user_id={student_id} and subject_id = {subject_id}  and sys_id={from_sys}
                GROUP BY topic_id """
    else:
        raise "Either topic_list or subtrand must be not None"
    # print(query)
    return query
