import os
import re
from langchain_core.output_parsers import JsonOutputParser
from openai import AsyncOpenAI
from config import OPENAI_API_KEY_DICT, TMPPATH
import autogen
from autogen.oai.client import OpenAIWrapper
from datetime import datetime
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from autogen.agentchat.contrib.society_of_mind_agent import SocietyOfMindAgent  # noqa: E402
from Modules.question_generator.chain import QuestionGeneratorChain
from autogen import gather_usage_summary
from langchain.callbacks import get_openai_callback

PRICE = {
'gpt-3.5-turbo-0125' : {'i': 0.5 / 1e6, 'o': 1.5 / 1e6}, 
'gpt-4-vision-preview': {'i': 10 / 1e6, 'o': 30 / 1e6},
'gpt-4-0125-preview': {'i': 10 / 1e6, 'o': 30 / 1e6}
}

# OpenAIWrapper.cache_path_root="/tmp"

async def summarize_math_solution_from_converation(math_problem, conversation, language='en',return_json=False):
    # SUMMARIZE SOLUTION
    summary_prompt = f"""
    You are a teacher. Given a question and a discussion aming to find solution for that question. You need to understand the content of the discussion. \
    Then, Extract all important steps, MUST INCLUDE equations in steps and the final answer for the question.
  
    HERE IS THE QUESTION :
    -----------------
    {math_problem}


    HERE IS THE DISCUSSION:
    --------------------
    {conversation}


    """
    if return_json:
        summary_prompt += """

    Finally ALWAYS reinterpret THOROUGHLY the solution as following guidlines
       - present in form of a list of steps (line break after each step) in order to help primary students understand.
       - You may ignore steps using 'Python'. 
    The output should be formatted as a JSON instance that conforms to the JSON schema below.

    {{
        'solution': 'The steps to find the final answer and the answer variables of all steps including the final answer .',
        'correct_answer':'Only the value of correct answer of the question without redundant texts.',
        'answer_options':'Based on the 'correct_answer' term, create 3 other fake answers,  example: ['fake_answer','fake_answer','correct_answer','fake_answer']'. Ensure choices MUST HAVE NO order symbols like 'A)', 'B)', 'C)', 'D)' 
    }}


    NOTE THAT, YOU MUST ALWAY TRANSLATE the content of json to '{language}' )
    """.format(language=language)
    else: 
        summary_prompt += f"""
    Finally answer the question by your way to help primary students understand and ALWAYS FOLLOW THESE RULES:
        - If the question is a simple equation or ASK ONLY THE FINAL ANSWER, Please return ONLY the final result of the question WITHOUT calculation steps. 
        - If the question inquire steps of solution by say something like 'how to solve ...', please return all your reinterpreted steps and the values of variables of all steps including the final result.
        - You may ignore steps using 'Python', your student certainly not understand that. 
    LAST BUT NOT LEAST, your response MUST BE TRANSLATED TO '{language}'
    """

    client = AsyncOpenAI(api_key=OPENAI_API_KEY_DICT['AUTOGEN'])
    cash_cost = tokens_cost=0
    with get_openai_callback() as cb:
        completion = await client.chat.completions.create(
        model="gpt-3.5-turbo-0125", #gpt-3.5-turbo-16k, #"gpt-4-0125-preview #gpt-3.5-turbo-0125
        messages=[{"role": "user", "content": summary_prompt}], response_format={ "type": "json_object" })
        response = completion.choices[0].message.content
        tokens_cost += cb.total_tokens 
        cash_cost  += cb.total_cost  

    if return_json:
        parser = QuestionGeneratorChain.question_parser(question_type_id=-1)
        response = response.replace("```json",'').replace("```",'').strip()
        try:
            solution = parser[1].parse(response) 
        except:
            try:
                solution = JsonOutputParser().parse(response)
            except: solution = {"solution": None, "correct_answer"  : None, "answer_options":None}
        
        solution['cost'] = {'tokens':tokens_cost, 'cash':cash_cost}
        def _standalize_option(text:str): 
            for i in ['A)', 'B)', 'C)', 'D)']: text =  text.replace(i, "")
            return text.strip()
        
        if not solution['correct_answer'] is None: solution['correct_answer'] = _standalize_option(solution['correct_answer'])
        if solution['answer_options'] is None: solution['answer_options'] = [_standalize_option(op) for op in solution['answer_options']]
        return solution
    else: 
        return response

async def summarize_math_solution_mathchat(math_problem, conversation, language = 'English', return_json = True):
    summary_prompt = f"""You are a teacher.

Given a question, your task is to find the solution for that question. You need to understand the content of the discussion.
In order to achieve this, you will also be given a conversation going between two different agents: USER and ASSISTANT.
Based on the question combined with the discussion, you will have to extract all the important steps in order to solve the question based on the conversation that has been going on. This means that you can take the summary of the conversation.
You must also include the equations in your steps and the final answer for the question.

HERE IS THE QUESTION:
-----------------
{math_problem}

HERE IS THE DISCUSSION:
--------------------
{conversation}

Finally, you need to always reinterpret thoroughly the solution by following these guidlines:
1. Present your solution interpretation in a form of a list of steps (line break after each step) in order to help the students understand.
2. Some agents might create a python code to help them get the answer. You can ignore this step (the steps that are using 'Python'), or you can also try to understand the logic behind the code and translate it into numerical calculations if needed.
3. The agents are designed to return the correct answer (the answer that they agreed upon) inside a '\boxed' tag. This means you can return the correct answer using this answer.
4. If there are more than 1 '\boxed' tag, use the one that is put on the last of the conversation.

Your output should be formatted as a JSON instance that conforms to the JSON schema below.

{{
    'solution': 'The steps to find the final answer and the answer variables of all steps including the final answer.',
    'correct_answer':'Only the value of correct answer of the question without redundant texts. Remember that the correct answer should be the one in the \boxed tag.',
}}

NOTE THAT, YOU MUST ALWAYS TRANSLATE the content of your JSON structure to '{language}'
"""

    client = AsyncOpenAI(api_key = OPENAI_API_KEY_DICT['AUTOGEN'])
    cash_cost = tokens_cost=0

    with get_openai_callback() as cb:
        completion = await client.chat.completions.create(
            model = "gpt-3.5-turbo-0125",
            messages = [{"role": "user", "content": summary_prompt}], response_format = { "type": "json_object" }
        )
        response = completion.choices[0].message.content
        tokens_cost += cb.total_tokens 
        cash_cost  += cb.total_cost

    if return_json:
        parser = QuestionGeneratorChain.question_parser(question_type_id = -1)
        response = response.replace("```json",'').replace("```",'').strip()
        try:
            solution = parser[1].parse(response)
        except:
            try:
                solution = JsonOutputParser().parse(response)
            except: solution = {"solution": None, "correct_answer": None}
        solution['cost'] = {'tokens':tokens_cost, 'cash':cash_cost}

        return solution
    else:
        return response

# given a math problem, we use the mathproxyagent to generate a prompt to be sent to the assistant as the initial message.
# the assistant receives the message and generates a response. The response will be sent back to the mathproxyagent for processing.
# The conversation continues until the termination condition is met, in MathChat, the termination condition is the detect of "\boxed{}" in the response.
async def find_solution_4mathprob(math_problem,  reference = "", language="en", use_image=False, return_json=True):
    config_list = [
        {
            'model': "gpt-4-vision-preview" if use_image else "gpt-3.5-turbo-0125",#'gpt-4-0125-preview',
            'api_key': OPENAI_API_KEY_DICT['VISION'] if use_image else OPENAI_API_KEY_DICT['AUTOGEN'],
            "max_tokens": 4024
        }
    ]
    # 1. create an AssistantAgent instance named "assistant" and the MathUserProxyAgent instance
    assistant = autogen.AssistantAgent(name="Tutor",llm_config={ "timeout": 180,"seed": 42, "config_list": config_list, }, 
                                       system_message="You are a well educated Learning Buddy called Jen.",)
    mathproxyagent = MathUserProxyAgent(name="mathproxyagent", human_input_mode="NEVER", code_execution_config={"use_docker": False},)

    prompt_temp = """
    Let find the solution for a math problem. Only Use Python to solve equations.
    If the question not is in English, please use langague the question present to discuss.

    Query requirements:
    You should always use the 'print' function for the output and use fractions/radical forms instead of decimals.
    You can use packages like sympy, numpy, math to help you.
    You must follow the formats below to write your code:
    ```python
    # your code
    ```
    Then, state the key idea to solve the problem STEP-BY-STEP. You may choose from two ways to solve the problem:
        Case 1: If the problem is mostly reasoning, you can solve it by yourself directly.
        Case 2: If the problem cannot be handled in the above way, please follow this process:
            1. Solve the problem STEP-BY-STEP (do not over-divide the steps).
            2. Take out any queries that can be asked through Python (for example, any calculations or equations that can be calculated). If using `round` funtion please add 5e-5 (for example round(x) --> round(x + 5e-5))
            3. Wait for me to give the results.
    Finally, return the  Show results of variables and the FINAL ANSWER for the question after all the queries are run.
    
    YOU ALSO NEED TO FOLLOW SOME BASIC MATH RULES as:
        - the first digit of a number had to be not 0.
        - Have to use only english in Python code script.

        
    """
    if reference != "":
        prompt_temp += f"""


    NOTE THAT, HERE IS A REFERENCE QUESTION AND SOLUTION Similar to the problem, which you can refer to this to build solution for the above problem, as :
    --------------------
    {reference}
    """
    prompt_temp += """
    
    HERE IS THE MATH QUESTION YOU NEED TO SOLVE:
    -----------------

    """
    
    #SOLVING PROBLEM
    chat_res = mathproxyagent.initiate_chat(assistant, problem=(math_problem),  clear_history=True, prompt_type="default", silent=True, customized_prompt= prompt_temp)
    
    conversation = ""
    trash_pattern = r"```python.*?```"
    for item in list(assistant._oai_messages.values())[0][1:]:
        if item['role'] == 'assistant':
            conversation += "\n----------\n"
            chat          = re.sub(trash_pattern, "", item['content'],  flags=re.DOTALL) 
            conversation += f"+ {chat}"
    response = await summarize_math_solution_from_converation(math_problem, conversation, language=language, return_json=return_json)
    return response

async def find_solution_4mathprobv2(math_problem,  reference = "", language="en", use_image=False, return_json=True):
    PROMPTS = """Given a math problem  try to solve it by logic inference and code.  If the question not is in English, please use langague the question present to discuss.
Query requirements:
You should always use the 'print' function for the outputs.
You can use packages like sympy to help you.
You must follow the formats below to write your code:
```python
# your code
```
TO SOLVE MATH QUESTION, YOU MUST ALWAYS DO this process STEP-BY-STEP:
    1. state the key idea to solve the problem. ALWAYS FOLLOW THIS :
        Simulate two brilliant, logical experts collaboratively answering a question. Each one verbosely explains their thought process in real-time, considering the prior explanations of others and openly acknowledging mistakes. At each step, whenever possible, each expert refines and builds upon the thoughts of others, acknowledging their contributions. They continue until there is a definitive answer to the question. For clarity, your entire response should be in a markdown table
        YOU ALSO NEED TO FOLLOW SOME BASIC MATH RULES as:
            - the first digit of a number had to be not 0.
            - 1 day = 24 hours, 13:00 is equivelant to 1:00 pm

    2. After all experts have provided their analysis, ALWAYS Simulate a Engineer to analyze all analyses and provide the consensus solution. 
    - Take out any queries that can be asked through Python code (for example, any calculations or equations that can be calculated) and functions you know in the context of this conversation.
    - Have to use only ENGLISH in Python code script.    
    - If using `round(X)` function YOU MUST ALWAYS add 5e-5 to X before call 'round' functions. For example, instead of round(x, 1), round(x + 5e-5, 1)
    - PAY ATTENTION on the requirement of the demand of the question (decimal form or fraction or radical forms) 

    3. Wait for me to give the results or wait for the executed results of the function call.
    4. Continue if you think the result is correct. If the result is invalid or unexpected, please correct your query or reasoning.

After all the queries are run and you get the answer, put the answer in \\boxed{}.
NOTE THAT, YOU HAVE TO SEPERATE STEPS of the process INTO AT LEAST 2 INDEPENDENT RESPONSES
    

"""
    if reference != "":
        prompt_temp += f"""


    NOTE THAT, HERE IS A REFERENCE QUESTION AND SOLUTION Similar to the problem, which you can refer to this to build solution for the above problem, as :
    --------------------
    {reference}
    """
    prompt_temp += """
    
    HERE IS THE MATH QUESTION YOU NEED TO SOLVE:
    -----------------

    """

    gpt_config = {
    "temperature": 0.5,
    "config_list": [
        {
            'model': "gpt-4-vision-preview" if use_image else "gpt-3.5-turbo-0125" , #'ft:gpt-3.5-turbo-0125:heyhi-pte-ltd::939cMaZ0',#'gpt-4-0125-preview'
            'api_key': OPENAI_API_KEY_DICT['VISION'] if use_image else OPENAI_API_KEY_DICT['AUTOGEN'],
            "max_tokens": 1024
        }
    ],
    "timeout": 120,
    "n": 5  # n should be 1 for now
    }
    assistant = autogen.AssistantAgent(name="Tutor",llm_config=gpt_config, #gpt_config,
                                    system_message="You are a well educated Learning Buddy called Jen.",
                                    human_input_mode="NEVER")
    mathproxyagent = MathUserProxyAgent(name="mathproxyagent", human_input_mode="NEVER", code_execution_config={"use_docker": False},)
    mathproxyagent.initiate_chat(assistant, problem=math_problem,  clear_history=True, prompt_type="default", silent=True, customized_prompt= PROMPTS)

    conversation = ""
    trash_pattern = r"```python.*?```"
    for item in list(assistant._oai_messages.values())[0][1:]:
        if item['role'].lower() in  ['assistant', 'teacher']:
            conversation += "\n----------\n"
            chat          = re.sub(trash_pattern, "", item['content'],  flags=re.DOTALL) 
            conversation += f"+ {chat}"

    response = await  summarize_math_solution_from_converation(math_problem, conversation, language=language, return_json=return_json)
    return response



async def find_solution_4mathprobv3(math_problem,  reference = "", language="en", use_image=False, return_json=True):
    gpt_config = {
    "temperature": 0.5,
    "config_list": [
        {
            'model': "gpt-4-vision-preview" if use_image else "gpt-3.5-turbo-0125" , #'ft:gpt-3.5-turbo-0125:heyhi-pte-ltd::939cMaZ0',#'gpt-4-0125-preview'
            'api_key': OPENAI_API_KEY_DICT['VISION'] if use_image else OPENAI_API_KEY_DICT['AUTOGEN'],
            "max_tokens": 1024
        }
    ],
    "timeout": 120,
    "n": 3  # n should be 1 for now
    }
    default_config = {
    "temperature": 0.5,
    "config_list": [
        {
            'model':  "gpt-3.5-turbo-0125" , #'ft:gpt-3.5-turbo-0125:heyhi-pte-ltd::939cMaZ0',#'gpt-4-0125-preview'
            'api_key': OPENAI_API_KEY_DICT['VISION'] if use_image else OPENAI_API_KEY_DICT['AUTOGEN'],
            "max_tokens": 1024
        }
    ],
    "n": 1,
    "timeout": 120
    }
    def termination_msg(x):
        return isinstance(x, dict) and  str(x.get("content", ""))[-9:].upper().find("TERMINATE") > -1
    expert1 = autogen.AssistantAgent(
        name="Teacher",
        llm_config=gpt_config,
        is_termination_msg= termination_msg,
        # max_consecutive_auto_reply=3,
        system_message="""
Given a math problem  try to solve it by logic inference \
Please Solve the problem step by step (do not over-divide the steps).  MUST ALWAYS FOLLOW THIS :
    - Simulate two brilliant, logical experts collaboratively answering a question. Each one verbosely explains their thought process in real-time, considering the prior explanations of others and openly acknowledging mistakes. At each step, whenever possible, each expert refines and builds upon the thoughts of others, acknowledging their contributions. \
    - They continue until there is a definitive answer to the question. For clarity, your entire response should be in a markdown table.
YOU ALSO NEED TO FOLLOW SOME BASIC MATH RULES as:
    - the first digit of a number had to be not 0.
    - 1 day = 24 hours, 13:00 is equivelant to 1:00 pm 
Finally, after Engineer and Executor recalculate results of your equations, update and review the final solution. NOTE THAT YOU MUST ALWAYS ONLY BELIEVE the result from engineer's code and SAY 'Reply `TERMINATE` to end the discussion. DON'T NEED SAY 'thank'
    """
    )
    engineer = autogen.AssistantAgent(
        name="Engineer",
        # max_consecutive_auto_reply=3,
        is_termination_msg=termination_msg,
        llm_config=default_config,
        system_message="""Engineer. You MUST ALWAYS write python code to implement Teacher's solution, in order to help him verify the results of all equations in the solution.
You should always use the 'print' function for the outputs MUST INCLUDE (name of variables, value of variable).
You can use packages like sympy, numpy to help you.
You must follow the formats below to write your code:
```python
# your code
```
If the code is not executed successfully, you would ask the Expert revise steps, then you can rewrite the code.
HERE ARE SOME RULES YOU HAVE TO FOLLOW AS:
    - Have to use only ENGLISH in Python code script.
    - If using `round(X)` function YOU MUST ALWAYS add 5e-5 to X before call 'round' functions. For example, instead of round(x, 1), round(x + 5e-5, 1)
    - PAY ATTENTION on the requirement of the demand of the question (decimal form or fraction or radical forms)

    """,
    )

    executor = autogen.UserProxyAgent(
        name="Executor",
        system_message="Executor. Execute the code written by the engineer and report the result.",
        human_input_mode="NEVER",
        code_execution_config={
            "last_n_messages": 3,
            # "work_dir": "paper",
            "use_docker": False,
        },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
    )
    # members = [ expert1, engineer, executor, critic]
    members = [ expert1, engineer, executor]
    groupchat = autogen.GroupChat(
        # agents=[ admin,expert1, expert2, engineer, executor], messages=[], max_round=15
        # agents=[ admin,expert1, expert2, executor], messages=[], max_round=20
        agents=members, messages=[], max_round=20,
        speaker_selection_method='round_robin'
    )

    prompt_temp = ''
    if reference != "":
        prompt_temp += f"""


    NOTE THAT, HERE IS A REFERENCE QUESTION AND SOLUTION Similar to the problem, which you can refer to this to build solution for the above problem, as :
    --------------------
    {reference}
    """
    prompt_temp += f"""
    
    HERE IS THE MATH QUESTION YOU NEED TO SOLVE:
    -----------------
    {math_problem}
    """

    #SOLVING PROBLEM
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=default_config)
    chathis = manager.initiate_chat(
        manager,
        clear_history=True,
        message= prompt_temp,
        silent=True,
    )

    conversation = ""
    trash_pattern = r"```python.*?```"
    for item in list(members[0]._oai_messages.values())[0][1:]:
        if item['role'].lower() in ['assistant', 'user', 'teacher', 'expert', 'critic'] or ('name' in item and item['name'].lower() in ['assistant', 'user', 'teacher', 'expert', 'critic']):
            if item['content'].lower().find("please keep solving") > -1: continue
            conversation += f"\n----------\n"
            chat          = re.sub(trash_pattern, "", item['content'],  flags=re.DOTALL) 
            conversation += f"+ {chat}"
            # print(f"=={item['role']}==")
            # print(item)
    # print(conversation)
    # return conversation

    response = await  summarize_math_solution_from_converation(math_problem, conversation, language=language, return_json=return_json)
    try:
        #ESTIMATE COST
        _, actual_usage_summary = gather_usage_summary(members)
        # _, actual_usage_summary = chathis.cost[0]
        tokens_cost = cash_cost = 0
        for mname in PRICE.keys():
            if mname in actual_usage_summary:
                tokens_cost += actual_usage_summary[mname]['total_tokens']
                cash_cost += PRICE[mname]['i'] * actual_usage_summary[mname]['prompt_tokens'] \
                            + PRICE[mname]['o'] * actual_usage_summary[mname]['completion_tokens']
        if isinstance(response, dict):
            response['cost']['cash'] += cash_cost
            response['cost']['tokens'] += tokens_cost
    except : pass
    return response

async def find_solution_mathchat(math_problem, language = "en", return_json = True):
    config_list = {
        "config_list": [
            {
                "model": "gpt-3.5-turbo-0125", 
                "api_key": OPENAI_API_KEY_DICT["AF"]
            }
        ]
    }

    assistant = autogen.AssistantAgent(
        name = "assistant",
        system_message = "You are a helpful assistant.",
        llm_config = {
            "timeout": 600,
            "config_list": config_list['config_list'],
            "cache_seed": None
        },
    )

    mathproxyagent = MathUserProxyAgent(
        name = "mathproxyagent",
        human_input_mode = "NEVER",
        code_execution_config = {"use_docker": False},
        llm_config = {
            "timeout": 600,
            "config_list": config_list['config_list'],
            "cache_seed": None
        },
    )
    print("start initiate chat")
    output = mathproxyagent.initiate_chat(assistant, message = mathproxyagent.message_generator, problem = math_problem)
    print("finish initiate chat")
    print(output)

    conversation = ""
    trash_pattern = r"```python.*?```"
    for item in list(mathproxyagent._oai_messages.values())[0][-3:]:
        if item['role'].lower() in ['assistant', 'user', 'teacher', 'expert', 'critic'] or ('name' in item and item['name'].lower() in ['assistant', 'user', 'teacher', 'expert', 'critic']):
            if item['content'].lower().find("please keep solving") > -1: continue
            conversation += f"{item['role'].upper()}: "
            conversation += f"{item['content']}"
            conversation += f"\n"

    conversation = conversation.rstrip('\n')

    response = await summarize_math_solution_mathchat(math_problem, conversation, language = language, return_json = return_json)

    return response


#####################Autogen for KB############3
async def find_answer_qna(question, reference = "", language="en"):
    default_config = {
    "temperature": 0.5,
    "config_list": [
        {
            'model':  "gpt-3.5-turbo-0125" , #'ft:gpt-3.5-turbo-0125:heyhi-pte-ltd::939cMaZ0',#'gpt-4-0125-preview'
            'api_key': OPENAI_API_KEY_DICT['AUTOGEN'],
            "max_tokens": 2024
        }
    ],
    "n": 3,
    "timeout": 120
    }

    assistant = autogen.AssistantAgent(
        "question-paraphaser",
        llm_config=default_config,
        max_consecutive_auto_reply=2,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        system_message=""" You MUST ALWAYS split the main question to 2-5 sub-questions to facilitate answering for question_interpreter . For example, who 40 years old can infer to people was born in 1984 if (today is in 2024)      """
    )
    question_interpreter = autogen.AssistantAgent(
        "question_interpreter",
        llm_config=default_config,
        max_consecutive_auto_reply=2,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        system_message=""" MUST ALWAYS give a detail by extracting infomation, figure, number, url from source document more specific as possible. 
        If The source document NOT include information for user's question, Say "not enough info". 
        """
    )

    # code_interpreter = autogen.UserProxyAgent(
    #     "question-solver",
    #     human_input_mode="NEVER",
    #     code_execution_config={
    #         "work_dir": "coding",
    #         "use_docker": False,
    #     },
    #     default_auto_reply="",
    #     is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
    # )

    groupchat = autogen.GroupChat(
        agents=[assistant,  question_interpreter],
        # agents=[assistant, code_interpreter, question_interpreter],
        messages=[],
        speaker_selection_method="round_robin",  # With two agents, this is equivalent to a 1:1 conversation.
        allow_repeat_speaker=False,
        max_round=4,
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
        llm_config=default_config
    )

    now = datetime.now()

    # Format the date as a string. For example, "April 2, 2024"
    date_str = now.strftime("%d %B %Y")

    society_of_mind_agent = SocietyOfMindAgent(
        "society_of_mind",
        chat_manager=manager,
        llm_config=default_config,
        response_preparer="""
        Output a response BASED ON the result of intermediate discussion but dont mentioning it.
        ALSWAY GIVE answer more detail (should include figures and url if they are available) as possible. 
        Break down your response to main points, subpoints and details but using friendly human tone instead of chatbot tone.
        - Dont say something like `from the source document provided`  and `If you have more questions..., feel free to ask!` in the end.
        If The source document NOT include information for user's question, DONT TRY ANSWER user's question by your personal knowledge. Even if I ask a simple questions like 1+1. Then, politely respond that you are tuned to only answer questions that are related to the context.
        """
    )

    user_proxy = autogen.UserProxyAgent(
        "user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        default_auto_reply="",
        is_termination_msg=lambda x: True,
    )

    problem=f"""
        Answer user's question BY the source document. \
        MUST ALWAYS FOLLOW STEP-BY-STEP the guidelines below to make the response:
            1. CHECK: If the question is a greeting (for example: "Hello", "Hi", "Good Morning"), please give a SHORT greeting as a daily conversational response.
            2. CHECK: If the user's question is related about date/age, please note that today is {date_str}. for example: a person 40 years old also means he was born in (year of today - 40)

        Your response should always be translated into '{language}' even if the source docs and previous discussions are in difference language.
        """ + f"""
        Here are the source documents:
        =========
        {reference}
        =========

        HERE IS THE QUESTION:
        {question}
        """

    chat_his=user_proxy.initiate_chat(society_of_mind_agent, message=problem, slient=True)
    chat_his = chat_his.chat_history
    # print(chat_his)

    response = {'cost':{'cash':0, 'tokens':0}, 'answer':chat_his[-1]['content']}
    try:
        #ESTIMATE COST
        _, actual_usage_summary = gather_usage_summary([assistant,  question_interpreter, society_of_mind_agent, manager])
        # _, actual_usage_summary = chathis.cost[0]
        tokens_cost = cash_cost = 0
        for mname in PRICE.keys():
            if mname in actual_usage_summary:
                tokens_cost += actual_usage_summary[mname]['total_tokens']
                cash_cost += PRICE[mname]['i'] * actual_usage_summary[mname]['prompt_tokens'] \
                            + PRICE[mname]['o'] * actual_usage_summary[mname]['completion_tokens']
        if isinstance(response, dict):
            response['cost']['cash'] += cash_cost
            response['cost']['tokens'] += tokens_cost
    except : pass
    return response
