import openai
import re
import json
from const import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

class QATemplate:
    categorize_concept_question = """
You are a well educated Virtual Tutor in {}. You will help student to analyse questions and specify the concept, skill student need to answer them.
Let analyze the text of following questions. Some questions contain html tag or latext tag due to error of converting, let ignore them. Try to understand the demand of questions. 
Then, Use your knowledge about {} cirriculumn and types of {} question, classify them into the categories I provide below.  
The category will be formed as [Main Skill].[Subskill] : [Definition]. For example, each line of categories will be look like:
        "Reading.Word Recognition : Recognize and read simple, common words."
Note that, as an expert, If you don't know the answer, just say "idk". Do NOT try to make up an answer. 
Besides that, the questions will organize as: 
    question 1,
    question 2,
    ...

Therefore, you need always return the answer in json format as:
{{
    1 : [Category for question 1],  
    2 : [Category for question 2],
    ...  
}}.
Here [Category for question] does not need the Definition, just in form [Main Skill].[Subskill] and Category has to belong the set of categories I provide.
"""

    arrange_topic_sequence="""
You are a good expert in Education. You have a list of topic in format as folowwing:
    topic_id1: name of topic_id1,
    topic_id2: name of topic_id2,
    ...

Based on you knowledge in education and your language understanding capability, you have to arrange the order of the set and build a learning path (which topic should learn first, which topic comes next, which topic does not need to be studied). So that students can MASTER ALL TARGET TOPICs sequencely in time-saving way. 
Note that, topics could not be related together, hence with a set of topics, you can build more than one learning path.

Therefore, you must always return ONLY the leanring path (without explaination) in json format as:

{{
    1 : [Topic_id_1]->[Topic_id_2]->...->[Topic_id_N]  
    2 : [Topic_id_A]->...->[Topic_id_ Z]
    ... 
}}.

Here,Topic 1 is the topic the student should learn first, Topic 2 is the next topic after student mastered Topic 1 and so on.
The output has to be only json file without explaination, redundant text and just include ID of topic. 
Remeber that, each topic is only allowed to exsist in one path.
"""

class PROMPT:
    


    def prompt_extract_concept_from_questions(self, question_texts:list, subject_name, taxonomy_skills=None):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Hi, How can I help you"},
                {"role": "user", "content": f"{QATemplate.categorize_concept_question.format(subject_name, subject_name, subject_name)}"},
                {"role": "system", "content": "What are skill categories"},
                {"role": "user", "content": f"Based on your knowlegde about subjects" if taxonomy_skills is None else f"Here \n {taxonomy_skills}"},
                {"role": "user", "content": f"now, let analyze some question bellow and tag them to precise categories :"},
                {"role": "user", "content": f",\n".join(question_texts)},
            ]
        )
        
        advice = completion.choices[0].message['content']
        # print(advice)
        pattern1 = r'[1-9]+:\s*"(\w+\.[\w\s\-]+):([\w\s\.\,\'\\]+)"'
        pattern2 = r'[1-9]+:\s*"(\w+\.[\w\s\-]+)"'
        mistakes = {}
        for line in advice.split("\n"):
            match = re.search(pattern1, line)

            if match:
                groups = match.groups()
                if groups[0].strip() in mistakes.keys(): mistakes[groups[0].strip()] += f" -{groups[1]}\n"
                else: mistakes[groups[0].strip()] = f" -{groups[1]}\n"
            else:
                match = re.search(pattern2, line)
                if match:
                    groups = match.groups()
                    if groups[0].strip() in mistakes.keys(): mistakes[groups[0].strip()] += f""
                    else: mistakes[groups[0].strip()] = f""                
        return mistakes

    def prompt_order_topics2sequence(self, topics:dict):
        set_of_topics="\t - \n".join([f"ID-{k} : {v}" for k, v in topics.items()])
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "Hi, How can I help you"},
                {"role": "user", "content": f"{QATemplate.arrange_topic_sequence}"},
                {"role": "system", "content": "What are topics you want to  arrange?"},
                {"role": "user", "content":  f"Here: \n {set_of_topics}"},
            ]
        )
        advice = completion.choices[0].message['content']
        
        #post-progress  
        finaladvice = None
        try:
            # Find the starting and ending index of the JSON-like section
            start_index = advice.find('{')
            end_index = advice.rfind('}') + 1

            # Extract the JSON-like section
            extracted_json = advice[start_index:end_index]
            advice = json.loads(extracted_json)
            
            seqpath = []
            for k in advice:
                path = [int(topic_id[3:]) for topic_id in  advice[k]]
                advice[k] = path
                seqpath += path
            finaladvice = {"seq_path":seqpath, 
                    "pal_path": {i+1: [v[i] for v in advice.values() if len(v) > i] for i in range(max(map(len, advice.values())))},
                    "reqs"    : {}}
        except Exception as e:
            print("[ERR]CANNOT parse text to json : \n", advice, "\n\n", e, "\n\n")
        return finaladvice

PROMPT_DICT = PROMPT()


