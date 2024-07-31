import openai
import os
import json
from botocore.config import Config as AWSConfig
from botocore import UNSIGNED
import boto3
import config
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from bs4 import BeautifulSoup
import time


def get_completion_image(prompt, image_url, model="gpt-4o"):
    print(image_url)
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {
            "url": image_url,
            "detail": 'high'
        }}
    ]}]
    OPENAIKEY= config.OPENAI_API_KEY_DICT['VISION'] if model=='gpt-4-vision-preview' else config.OPENAI_API_KEY_DICT['RUBRIC_MARKING']
    client = openai.OpenAI(api_key=OPENAIKEY)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4000,
        temperature=0.1,
    )
    return response.choices[0].message.content

def get_image_description(image_url, model_solution):
    image_description_structure = r"""
        {
            "objects": [
                {
                    "name": ,
                    "description":
                },
                ...
            ],
            "description":

        }
    """

    prompt = """
        You are given an image. Provide a concist list of objects you found in the image. \
        Make sure that the list of objects is comprehensive and exhaustive, before you continue. \
        Next, for each object in the list that you have provided, describe the object in detail. \
        To be more exact, you can refer to the following description passage of the image: \
        ```
        {model_solution}
        ```
        Next, based on the list of objects and the description passage, write a comprehensive description of the image.
        Your response should be in the following JSON format:
        {image_description_structure}
    """.format(model_solution=model_solution, image_description_structure=image_description_structure)

    prompt = """
        Given image that attached as a content or a solution for question in a quiz. 
        Your purpose is to give a clear description of the image given.
        If the image is containing a bar chart of Average Monthly Rent Apartments in 2020 (USD), please use this information: New Yorks = $3,500, London = $2,000, Tokyo = $2,500, Sydney = $3,000, Berlin = $1,500
        If the image is containing a text, ensure to convert all the text to the description.
        Your response should be in the following JSON format:
         {image_description_structure}
    """.format(image_description_structure=image_description_structure)

    image_description = get_completion_image(prompt, image_url)

    def parse_image_description(image_description):
        import re
        pattern = r"\{(.|\n)*\}"
        match = re.search(pattern, image_description)
        if match:
            return json.loads(match.group(0))
        return None
    
    return "Start image description. " + parse_image_description(image_description)['description'] + " End image description."

async def get_completion_turbo(prompt, model="gpt-4o", question=""):
    messages = [{"role": "system", "content": prompt},{"role":"user", "content":question}]
    OPENAIKEY= config.OPENAI_API_KEY_DICT['VISION'] if model=='gpt-4-vision-preview' else config.OPENAI_API_KEY_DICT['RUBRIC_MARKING']
    client = openai.AsyncOpenAI(api_key=OPENAIKEY)
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=4000,
        temperature=0.1,
        response_format= {"type": "json_object"}
    )
    return response.choices[0].message.content

def criteria_breakdown(criteria):
    if criteria['score_type'] == "range":
        breakdown = ""
        for i in range(len(criteria['breakdown'])):
            breakdown += "{from_score}-{to_score}: {description}\n".format(
                from_score=criteria['breakdown'][i]['from_score'],
                to_score=criteria['breakdown'][i]['to_score'],
                description=criteria['breakdown'][i]['description']
            )
    elif criteria['score_type'] == "fixed":
        breakdown = ""
        for i in range(len(criteria['breakdown'])):
            breakdown += "Score {score}: {description}\n".format(
                score=criteria['breakdown'][i]['score'],
                description=criteria['breakdown'][i]['description']
            )
    return breakdown

async def criteria_marking(question_statement, criteria, student_composition, student_class):
    output_structure = r"""
        {
            "mark": <your mark>,
            "feedback": <your feedback>,
        }
    """

    additional_string = ""

    if criteria['score_type'] == "fixed":
        additional_string = "Following the breakdown above, make sure to provide a mark that is in the breakdown options. That means the mark you give should only be one of the scores in the breakdown."

    elif criteria['score_type'] == "range":
        additional_string = "Following the breakdown above, make sure to provide a mark that is within the range of the breakdown options. That means the mark you give should not be above the highest score or below the lowest score in the breakdown."

    prompt = """
    You are a {student_class} instructor. You have just received a composition from a {student_class} student. \
    You can find the description of the composition below. \
    ```
    {question_statement}
    ```
    Your task is to mark the student's composition and provide feedback to the student. \
    You need to assess the student's performance based on a content criteria. \
    The criteria {criteria_name} has a maximum score of {criteria_max_score}, with the following breakdown: \
    ```
    {criteria_breakdown}
    ```

    {additional_string}

    Low score means the criteria is not met from the student's composition, while higher score means the criteria is passed. \

    You must explain what is missing in the student's composition and how the student can improve, if there is any. \
    You must cite exactly the part of the student's composition that you are referring to. \
    You must not include the student's score in the feedback. \
    You must answer in the same language as what the student has written. \
    You must give feedback in a short paragraph not more than 250 words. \
    Your response should be in the following JSON format:
    {output_structure}
    """.format(
        student_class = student_class,
        question_statement = question_statement,
        student_composition = student_composition,
        criteria_name = criteria['name'],
        criteria_breakdown = criteria_breakdown(criteria),
        output_structure = output_structure,
        criteria_max_score = criteria['max_score'],
        additional_string = additional_string
    )
    question = f"Here is my answer: {student_composition}"
    response = await get_completion_turbo(prompt,"gpt-4o",question)
    return json.loads(response)

def refine_rubric_table(rubric_table, model_solution):
    for criteria in rubric_table:
        highest_breakdown_id = -1
        highest_breakdown_score = 0
        # Find the highest breakdown
        for i in range(len(criteria['breakdown'])):
            if criteria['breakdown'][i]['to_score'] > highest_breakdown_score:
                highest_breakdown_score = criteria['breakdown'][i]['to_score']
                highest_breakdown_id = i
        # Refine the description of the highest breakdown
        prompt = """
        You are given a rubric table for marking a composition. The rubric table contains the following criteria: {criteria_name}. \
        The highest score for {criteria_name} is {highest_breakdown_score}. \
        The description for the highest score is as follows: {description}. \
        Based on the model solution, which is as follows: \
        ```
        {model_solution}
        ```
        Refine the description for the highest score for {criteria_name} to better match the model solution. \
        Your response should use the model solution as a reference to explain in more detail what is expected for the highest score. \
        Your response should be in the following JSON format:
        {output_structure}
        """.format(criteria_name=criteria['name'], highest_breakdown_score=highest_breakdown_score,
                     description=criteria['breakdown'][highest_breakdown_id]['description'], model_solution=model_solution,
                     output_structure=r"""
                      {
                            "description": <your description>
                      }
                     """)
        response = get_completion_turbo(prompt,'gpt-3.5-turbo')
        criteria['breakdown'][highest_breakdown_id]['description'] = json.loads(response)['description']
    return rubric_table

def refine_rubric_table_prompt(criteria, highest_breakdown_score, highest_breakdown_id, model_solution):
    prompt = """
    You are given a rubric table for marking a composition. The rubric table contains the following criteria: {criteria_name}. \
    The highest score for {criteria_name} is {highest_breakdown_score}. \
    The description for the highest score is as follows: {description}. \
    Based on the model solution, which is as follows: \
    ```
    {model_solution}
    ```
    Refine the description for the highest score for {criteria_name} to better match the model solution. \
    Your response should use the model solution as a reference to explain in more detail what is expected for the highest score. \
    Your response should be in the following JSON format:
    {output_structure}
    """.format(criteria_name=criteria['name'], highest_breakdown_score=highest_breakdown_score,
                description=criteria['breakdown'][highest_breakdown_id]['description'], model_solution=model_solution,
                output_structure=r"""
                {
                        "description": <your description>
                }
                """)
    return prompt

async def refine_rubric_table_v2(rubric_table, model_solution):
    for criteria in rubric_table:
        highest_breakdown_id = -1
        highest_breakdown_score = 0

        if criteria['score_type'] == "range":
            # Find the highest breakdown
            for i in range(len(criteria['breakdown'])):
                if criteria['breakdown'][i]['to_score'] > highest_breakdown_score:
                    highest_breakdown_score = criteria['breakdown'][i]['to_score']
                    highest_breakdown_id = i

        elif criteria['score_type'] == "fixed":
            for i in range(len(criteria['breakdown'])):
                if criteria['breakdown'][i]['score'] > highest_breakdown_score:
                    highest_breakdown_score = criteria['breakdown'][i]['score']
                    highest_breakdown_id = i

        # Refine the description of the highest breakdown
        prompt = refine_rubric_table_prompt(
            criteria, highest_breakdown_score, highest_breakdown_id, model_solution
        )
        response = await get_completion_turbo(prompt,'gpt-3.5-turbo')
        criteria['breakdown'][highest_breakdown_id]['description'] = json.loads(response)['description']

    return rubric_table

def compo_marking(question_statement, rubric_table, student_composition, model_composition, student_class):
    """"
    question_statement: An HTML string that contains the question statement. All the <img> tags will \
    be replaced with <p> tags that contain the description of the image.
    rubric_table: A list of criteria that the student's composition will be assessed against. The \
    structure of the table is as follows
    {
        'rubric': [
            {
                "name": "Content",
                "max_score": 20,
                "breakdown": [
                    {
                        "from_score": 0,
                        "to_score": 5,
                        "description": "The student's composition does not contain any relevant information."
                    },
                    ...
                ]
            },
            ...
        ]
    }
    student_composition: A string that contains the student's composition.
    model_composition: A string that contains the model composition. The model solution will have \
    the highest score for all the criteria.
    student_class: A string that contains the student's class. For example, "Primary 1".
    """
    
    # Replace all the <img> tags with <p> tags that contain the description of the image.
    soup = BeautifulSoup(question_statement, 'html.parser')
    for img in soup.find_all('img'):
        # Create a new empty p tag
        new_tag = soup.new_tag("p")
        # Insert the new p tag before the img tag
        img.insert_before(new_tag)
        # Insert the description of the image into the new p tag
        new_tag.string = get_image_description(img['src'], model_composition)
        # Remove the img tag
        img.decompose()
    question_statement = str(soup.text)
    # print("Question statement:", question_statement)

    # Refine the rubric table
    if (model_composition is not None):
        rubric_table = refine_rubric_table(rubric_table, model_composition)
    # print(rubric_table)
    final_score = 0
    feedback_list = []
    for criteria in rubric_table:
        result_student = criteria_marking(question_statement, criteria, student_composition, student_class)
        feedback_list.append({
            "criteria_name": criteria['name'],
            "feedback": result_student['feedback']
        })
        print("Student's mark for {criteria_name}: {mark}".format(criteria_name=criteria['name'], mark=result_student['mark']))
        # print("Feedback for {criteria_name}: {feedback}".format(criteria_name=criteria['name'], feedback=result_student['feedback']))
        if (model_composition is None):
            result_model = {
                "mark": criteria['max_score'],
            }
        else:
            result_model = criteria_marking(question_statement, criteria, model_composition, student_class)
        
        print("Model's mark for {criteria_name}: {mark}".format(criteria_name=criteria['name'], mark=result_model['mark']))
        # print("Model's feedback for {criteria_name}: {feedback}".format(criteria_name=criteria['name'], feedback=result_model['feedback']))
        # final_score += min(criteria['max_score'], result_student['mark'] * (criteria['max_score'] / result_model['mark']))

        ## updating final_score to only add the student mark without manipulating it regardless of the model mark
        final_score += min(criteria['max_score'], result_student['mark'])

    feedback_string = ""
    for feedback in feedback_list:
        feedback_string += "{criteria_name}: {feedback}\n".format(criteria_name=feedback['criteria_name'], feedback=feedback['feedback'])
    
    return final_score, feedback_string

async def compo_marking_v2(question_statement, rubric_table, student_composition, model_composition, student_class):
    """"
    question_statement: An HTML string that contains the question statement. All the <img> tags will \
    be replaced with <p> tags that contain the description of the image.
    rubric_table: A list of criteria that the student's composition will be assessed against. The \
    structure of the table is as follows
    {
        'rubric': [
            {
                "name": "Content",
                "max_score": 20,
                "breakdown": [
                    {
                        "from_score": 0,
                        "to_score": 5,
                        "score_type": "range" or "fixed"
                        "description": "The student's composition does not contain any relevant information."
                    },
                    ...
                ]
            },
            ...
        ]
    }
    student_composition: A string that contains the student's composition.
    model_composition: A string that contains the model composition. The model solution will have \
    the highest score for all the criteria.
    student_class: A string that contains the student's class. For example, "Primary 1".
    """
    
    # Replace all the <img> tags with <p> tags that contain the description of the image.

    soup = BeautifulSoup(question_statement, 'html.parser')
    for img in soup.find_all('img'):
        # Create a new empty p tag
        new_tag = soup.new_tag("p")
        # Insert the new p tag before the img tag
        img.insert_before(new_tag)
        # Insert the description of the image into the new p tag
        new_tag.string = get_image_description(img['src'], question_statement)
        # Remove the img tag
        img.decompose()

    question_statement = str(soup.text)

    print("Question statement: \n")
    print(question_statement)

    new_rubric_table = rubric_table

    # Refine the rubric table
    if (model_composition is not None and model_composition != ""):
        new_rubric_table = await refine_rubric_table_v2(rubric_table, model_composition)
        
    final_score = 0
    feedback_list = []
    score_detail = []

    for new_criteria, criteria in zip(new_rubric_table, rubric_table):
        result_student = await criteria_marking(question_statement, new_criteria, student_composition, student_class)
        feedback_list.append({
            "criteria_name": new_criteria['name'],
            "feedback": result_student['feedback']
        })

        print("Student's mark for {criteria_name}: {mark}".format(criteria_name=new_criteria['name'], mark=result_student['mark']))
        if (model_composition is None or model_composition == ""):
            result_model = {
                "mark": new_criteria['max_score'],
            }
        else:
            result_model = await criteria_marking(question_statement, new_criteria, model_composition, student_class)
        
        print("Model's mark for {criteria_name}: {mark}".format(criteria_name=new_criteria['name'], mark=result_model['mark']))

        ## updating final_score to only add the student mark without manipulating it regardless of the model mark
        final_score += min(new_criteria['max_score'], result_student['mark'])

        ## Getting the description of each category score
        if criteria['score_type'] == "range":
            for breakdown in criteria['breakdown']:
                if breakdown["from_score"] <= result_student['mark'] <= breakdown["to_score"]:
                    score_detail.append({
                        "component": criteria['name'],
                        "score": result_student['mark'],
                        "feedback": result_student['feedback'],
                        "description": breakdown["description"],
                        "id": breakdown["id"]
                    })
                    break

        elif criteria['score_type'] == "fixed":
            for breakdown in criteria['breakdown']:
                if breakdown["score"] == result_student['mark']:
                    score_detail.append({
                        "component": criteria['name'],
                        "score": result_student['mark'],
                        "feedback": result_student['feedback'],
                        "description": breakdown["description"],
                        "id": breakdown["id"]
                    })
                    break

    feedback_string = ""
    for feedback in feedback_list:
        feedback_string += "{criteria_name}: {feedback}\n".format(criteria_name=feedback['criteria_name'], feedback=feedback['feedback'])

    feedback_output = {
        "feedback": {
            "mark": final_score,
            "feedback": feedback_string,
            "score_detail": score_detail
        }
    }
    
    return feedback_output

def automatic_rubric_table_converter(rubric_table_string):
    example_rubric_table = r"""
        [
            {
                "name": "Content",
                "max_score": 20,
                "breakdown": [
                    {
                        "from_score": 1,
                        "to_score": 4,
                        "description": "Ideas largely irrelevant to the topic. Composition is vague and confusing."
                    },
                    {
                        "from_score": 5,
                        "to_score": 8,
                        "description": "A few relevant ideas. Mundane composition that is mostly undeveloped."
                    },
                    {
                        "from_score": 9,
                        "to_score": 11,
                        "description": "Some relevant ideas. Composition is of some interest and is minimally developed."
                    },
                    {
                        "from_score": 12,
                        "to_score": 14,
                        "description": "Generally relevant ideas. Fairly interesting and sufficiently developed composition."
                    },
                    {
                        "from_score": 15,
                        "to_score": 17,
                        "description": "Relevant ideas. Interesting and well-developed composition."
                    },
                    {
                        "from_score": 18,
                        "to_score": 20,
                        "description": "Fully relevant ideas. Highly interesting and thoroughly developed composition."
                    }
                ]
            },
            ...
        ]
    """
    prompt = """
        You are given a string that contains a rubric table. Please convert the rubric table into the following JSON format. \
        This is an example correct format for rubric table:
        ```
        {example_rubric_table}
        ```
        Please convert the rubric table into the correct JSON format. Here is the rubric table string:
        {rubric_table_string}
    """.format(example_rubric_table=example_rubric_table, rubric_table_string=rubric_table_string)
    response = get_completion_turbo(prompt)
    return json.loads(response)
        

def test():
    """
        Test calling with this data:
    
    """
    question_statement = """
        <p>Write a composition (around 8 sentences) about the image below.</p>
        <img src="https://pdf-digitalization.s3.amazonaws.com/compo_marking_sample.png">        
    """
    rubric_table = [
        {
            "name": "Content",
            "max_score": 20,
            "breakdown": [
                {
                    "from_score": 1,
                    "to_score": 4,
                    "description": "Ideas largely irrelevant to the topic. Composition is vague and confusing."
                },
                {
                    "from_score": 5,
                    "to_score": 8,
                    "description": "A few relevant ideas. Mundane composition that is mostly undeveloped."
                },
                {
                    "from_score": 9,
                    "to_score": 11,
                    "description": "Some relevant ideas. Composition is of some interest and is minimally developed."
                },
                {
                    "from_score": 12,
                    "to_score": 14,
                    "description": "Generally relevant ideas. Fairly interesting and sufficiently developed composition."
                },
                {
                    "from_score": 15,
                    "to_score": 17,
                    "description": "Relevant ideas. Interesting and well-developed composition."
                },
                {
                    "from_score": 18,
                    "to_score": 20,
                    "description": "Fully relevant ideas. Highly interesting and thoroughly developed composition."
                }
            ]
        }
    ]
    student_composition = """
    Emily and Sharon were in the classroom.  Emily took out her water bottle to have a sip of water.
    Emily accidentally dropped her water bottle.  Water spilt out and formed a puddle on Sharon’s desk.  Sharon was furious with Emily for making her desk so wet.
    Both Emily and Sharon began arguing with each other.  Their friend Alice tried to stop them from quarrelling.
    In the end, Emily apologised to Sharon and wiped Sharon’s desk dry with a rag.  Sharon forgave her.  Alice was pleased that both her friends had made up and were no longer angry with each other.
    """
    model_composition = """
    It was almost time for the first period and Emily and Sharon were in the classroom.  Feeling thirsty, Emily took out her water bottle to have a sip of water.
    Emily’s fingers slipped and she accidentally dropped her water bottle.  Water spilt out and formed a huge puddle on Sharon’s desk.  Sharon was furious with Emily for causing her desk to be so wet.
    Both Emily and Sharon began arguing fiercely with each other.  Their friend Alice tried her best to stop them from quarrelling.
    In the end, Emily apologised to Sharon and wiped Sharon’s desk dry with a rag.  Sharon forgave her in return.  Alice was over the moon that both her friends had made up and were no longer furious with each other.
    """
    student_class = "Primary 1"
    print(compo_marking(question_statement, rubric_table, student_composition, model_composition, student_class))

if __name__ == "__main__":
    test()