#############SIMPLE QUERY###############3
def get_content_question(question_ids:list):
    question_sent = ", ".join([ str(s) for s in question_ids + ["-99"]])
    query = f"""SELECT distinct(q.question_id), q.reference_id, 
                    CASE WHEN q.question_id = q.reference_id THEN q.question_text
                        ELSE concat(rq.question_text,"\n",q.question_text)
                    END as question_text,
                    a.answer_text
            FROM sj_questions as q
            JOIN sj_questions as rq ON q.reference_id = rq.question_id
            JOIN sj_answers as a On a.question_id = q.question_id
            WHERE q.question_id in ({question_sent})     
            """
    wrap_query = f"""SELECT s.question_id, s.reference_id, CONCAT('QUESTION: \n======\n\n', s.question_text, '\n======\n\nANSWER: ', s.answer_text) as text 
                    FROM ({query}) as s
                    GROUP BY s.question_id"""
    return wrap_query