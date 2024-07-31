from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain,LLMMathChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os 
import json
from langchain.schema import ( HumanMessage, SystemMessage )
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from datetime import datetime
from config import OPENAI_API_KEY_DICT
# from langchain.embeddings import GPT4AllEmbeddings

class QuestionBlank(BaseModel):
    question: str = Field(description="content of question")
    correct_answer: str = Field(description="['correct answer for blank 1','correct answer for blank 2',etc.] ensure the length of correct answer aligned with total of blank that provided in the question")
    difficulty_level: str = Field(description="easy, normal, hard")
    mark: str = Field(description="mark range from 1-5 mark")
    answer_options: List = Field(description="list of answer option for each blank if there's any. example: ['answeroption 1 blank 1','answeroption 2 blank 1'],['answeroption 1 blank 2','answeroption 2 blank 2']")
    solution: str = Field(description="short explanation of the correct answer")
class QuestionListBlank(BaseModel):
    questions: List[QuestionBlank] = Field(description="list of question")
class Question(BaseModel):
    question_header: str = Field(description="header content of question")
    question: str = Field(description="content of question")
    correct_answer: str = Field(description="correct answer of the question, always provide the correct answer. if there's answer options, ensure the correct answer is from the answer options for example if the answer option ['answer 1','answer 2'] so the correct answer should be 'answer 1' or 'answer 2'")
    difficulty_level: str = Field(description="easy, normal, hard")
    mark: str = Field(description="mark range from 1-5 mark")
    answer_options: List = Field(description="list of answer option if the reference question is multiple choice question. example: ['answer 1','answer 2','answer 3','answer 4']")
    solution: str = Field(description="short explanation of the correct answer")
class QuestionList(BaseModel):
    questions: List[Question] = Field(description="list of question")

class QuestionNoAns(BaseModel):
    question_header: str = Field(description="header content of question")
    question: str = Field(description="content of question")
    difficulty_level: str = Field(description="easy, normal, hard")
    mark: str = Field(description="mark range from 1-5 mark")
class QuestionListNoAns(BaseModel):
    questions: List[QuestionNoAns] = Field(description="list of question")
class QuestionAns(BaseModel):
    correct_answer: str = Field(description="correct answer of the question, always provide the correct answer. if there's answer options, ensure the correct answer is from the answer options for example if the answer option ['answer 1','answer 2'] so the correct answer should be 'answer 1' or 'answer 2'")
    answer_options: List = Field(description="list of answer option if the reference question is multiple choice question. example: ['answer 1','answer 2','answer 3','answer 4']")
    solution: str = Field(description="short explanation of the correct answer")
class QuestionAnsList(BaseModel):
    questions: List[QuestionAns] = Field(description="list of question")

class KBQnAChain:
    @staticmethod
    def question_parser(question_type_id):
        parser = JsonOutputParser(pydantic_object=QuestionList)
        if question_type_id in [5, 6, 7]:
            parser = JsonOutputParser(pydantic_object=QuestionListBlank)
        if question_type_id is [-1]:
            parserQ = JsonOutputParser(pydantic_object=QuestionListNoAns)
            parserA = JsonOutputParser(pydantic_object=QuestionAnsList)
            parser = (parserQ, parserA)

        return parser
    
    def convert_to_json_chain(question_type_id):
        qa_template = """
            Convert the provided question into strucutred JSON.

            The output should be formatted as a JSON instance that conforms to the JSON schema below.
             {{questions: [{{
                question_header:'header content of question',
                question:'content of question (the language must be same as the reference docs)',
                difficulty_level:'easy, normal, hard',
                mark:'mark range from 1-5 mark',
                solution:'The steps to find the final answer (the language must be same as the reference docs). If the question is a math question or a question that involves calculation, please make sure that the calculation is done correctly.',
                answer_options:'list of answer option if the reference question is multiple choice question. example: ['answer 1','answer 2','answer 3','answer 4']',
                correct_answer:'correct answer of the question, always provide the correct answer. if there's answer options, ensure the correct answer is from the answer options for example if the answer option ['answer 1','answer 2'] so the correct answer should be 'answer 1' or 'answer 2''
                }}]
            }}

            MAKE SURE THE OUTPUT IS A JSON INSTANCE
            """
        if question_type_id in [5, 6, 7]:
            qa_template = """
                Convert the provided question into strucutred JSON.

                The output should be formatted as a JSON instance that conforms to the JSON schema below.
                {{questions: [{{
                    question_header:'header content of question',
                    question:'content of question (the language must be same as the reference docs)',
                    difficulty_level:'easy, normal, hard',
                    mark:'mark range from 1-5 mark',
                    solution:'The steps to find the final answer (the language must be same as the reference docs). If the question is a math question or a question that involves calculation, please make sure that the calculation is done correctly.',
                    answer_options:'list of answer option for each blank if there's any. example: ['answeroption 1 blank 1','answeroption 2 blank 1'],['answeroption 1 blank 2','answeroption 2 blank 2']',
                    correct_answer: '['correct answer for blank 1','correct answer for blank 2',etc.] ensure the length of correct answer aligned with total of blank that provided in the question'
                    }}]
                }}

                MAKE SURE THE OUTPUT IS A JSON INSTANCE
                """
        systemPrompt = []
        userPrompt =[]
        systemPrompt.append({
            "type":"text",
            "text": qa_template
        })
        llm = ChatOpenAI(model_name="gpt-4-0125-preview", openai_api_key=OPENAI_API_KEY_DICT['QG'], temperature=0, 
                         request_timeout=120,max_tokens=4050, response_format={ "type": "json_object" })


        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
            partial_variables={},
        )
        prompts = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt = prompt),
            # HumanMessage(
            #     content=userPrompt
            # ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ])
        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
        )

        return chain
    

    def categorize_question(refprompt):
        qa_template = """
            You are an AI question classifier. 
            Given a question, You need to understand the question and specify which subject the question belongs to:

            The output should be formatted as a JSON instance that conforms to the JSON schema below.
                {{
                    subject: "name of subject such as math/physic/english/chinese/..."]
                }}

            MAKE SURE THE OUTPUT IS A JSON INSTANCE and the subject is one of highschool subjects. 
            """

        userPrompt =[]
        prompt = PromptTemplate(
                template=qa_template,
                input_variables=[],
                partial_variables={},
            )
        userPrompt.append({
            "type":"text",
            "text": "Here is my question's content:"
        })
        refpromptjson = json.loads(refprompt)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", openai_api_key=OPENAI_API_KEY_DICT['QG'], temperature=1.5, request_timeout=120,max_tokens=4020) #gpt-4-0125-preview gpt-3.5-turbo-16k
        for item in refpromptjson:
            if item["type"] == "image_url":
                llm = ChatOpenAI(model_name="gpt-4-vision-preview", openai_api_key=OPENAI_API_KEY_DICT['VISION'], temperature=0.5, request_timeout=120,max_tokens=4050)
            userPrompt.append(item)

        prompts = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt = prompt),
            HumanMessage(content=userPrompt),
        ])

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=False,
        )

        return chain

    def check_model_destination4quizreview(chat_history, refprompt):
        """
        This function is design, first for aming to use in chain_quiz_review
        Given context of chat (history and reference) and  user's question.
        Specify if the user's demand needs to use vision-gpt or not
        """
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY_DICT['AF'],
                        model_name="gpt-3.5-turbo-0125", temperature=0.5, request_timeout=120, max_tokens=500)
        # Additionally, analyze the provided answer for good points within the specified range. Do the checking specifically, don't give points that in general.

        qa_template = """
Given a list of question and user's question, which there are some questions probably have images (represented by image_url).  
You MUST ALWAYS PREDICT the PURPOSE of user's question. Then you return a label tag for that question as "NEED INFO FROM IMAGE" ONLY IF the user's question is one of these cases:
    1. he want to review his answer or the solution for the QUESTION CONTAINING IMAGE URL.
    2. he want to inquire explation or infomation related any image in Question list.

IF the purpose is not in above case, ALWAYS return a label tag "NONE".
NOTE THAT:
- You also are given chat history, so if the image is asking by student's question already described by text in chat history, dont need to tag "NEED INFO FROM IMAGE" but tag "NONE".
- You DO NOT NEED TO ANSWER the student's question, THIS's NOT YOUR TASK. 
        """

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
        )
        userPrompt =[]
        userPrompt.append({
            "type":"text",
            "text": "HERE IS THE QUIZ: "
        })
        refpromptjson = json.loads(refprompt)
        for item in refpromptjson:
            if item["type"] == "image_url": 
                _url = item['image_url']
                item = {'type':'text', 'text': f'[attached image]<url>({_url})'}
                
            userPrompt.append(item)

        userPrompt.append({
            "type":"text",
            "text": "\n\n =======================\nHERE IS THE CHAT HISTORY: "
        })
        for item in chat_history:
            # memory.chat_memory.add_user_message(item.question)
            # memory.chat_memory.add_ai_message(item.answer)
            userPrompt.append({
                "type":"text",
                "text": f"==>USER : {item.question} "
            })
            userPrompt.append({
                "type":"text",
                "text": f"==>TEACHER: {item.answer} "
            })

        prompts = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate(prompt = prompt),
                HumanMessage(
                    content=userPrompt
                ),
                # The `variable_name` here is what must align with memory
                # MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("HERE IS THE QUESTION YOU NEED TO PREDICT IT'S PURPOSE :\n\n {question}")
            ]
        )
        # memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)



        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            # memory=memory
        )

        return chain

    def predict_language():
        """
        This function is a design, aimed to be used in chain_quiz_review
        Given context of chat (history and reference) and user's question,
        Specify if the user's demand needs to use vision-gpt or not.
        """
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY_DICT['AF'],
                        model_name="gpt-3.5-turbo-0125", temperature=0, request_timeout=120, max_tokens=50)
        # Additionally, analyze the provided answer for good points within the specified range. Do the checking specifically, don't give points that in general.

        qa_template = """
        Given a text, predict which language is used in the given text. \
        The text could contain some words/symbols in different languages. Therefore, we MUST consider entire text to predict.
        The output MUST ALWAYS FORMATTED AS A STRING: 'full_name_of_language(english/vietnamese/...)' without extra explaination.
        """

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
        )

        prompts = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate(prompt = prompt),
                # HumanMessage(
                #     content=userPrompt
                # ),
                # The `variable_name` here is what must align with memory
                # MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("THE TEXT YOU NEED TO PREDICT THE LANGUAGE USED :\n\n  ```{question}``` ")
            ]
        )
        # memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)
        
        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=False,
            # memory=memory
        )

        return chain


    def analyze_question(history):
        now = datetime.now()

        # Format the date as a string. For example, "April 2, 2024"
        date_str = now.strftime("%d %B %Y")
        qa_template = """
            You are an AI assistant. Given a new question `A` from an user, a previous discusstion between user and a chatbot including some question-answer pairs (could not relevant to the question `A`).
            You must always make response by infering STEP-BY-STEP as:
            1. Based on discussion and the user's current question, SPECIFY what objects, person and demands user want to inquire infomation in the question `A`
            2. If the question is a greeting (for example: "Hello", "Hi", "Good Morning", "how are you"), please give a SHORT greeting as a daily conversational response. Please preserve the question.
               But if question `A` is asked to inquire information, you MUST paraphase the question `A` to eplore more information .
            The output should be formatted as a JSON instance that conforms to the JSON schema below.
                {{
                    'refined_question': "a new question form of `A`, which shows EXACTLY name of subject, object mentioned in question `A`. DO NOT USE pronouns like he, she, it" ,
                    'paraphased_question': "a List. you MUST create new 2 questions to make question `A` more clear. If question is a greating, return empty list.",
                }}
        
            """ + f""" MAKE SURE THE OUTPUT IS A JSON INSTANCE and  'refine_question' and 'paraphased_question' are in English. And Today is {date_str}
            
            """

        # Here are some examples:
             
        #     Previous Discusion: 
        #         user: - What is the topic of this Documents?
        #         chatbot: - It talk about ABC
        #     Current Question: can you summarize it?
        #     Output:
        #         "refined_question": "Can you summarize main points about ABC in the documents",
        #         "paraphased_question": ["What is ABC", "How ABC work", "Why ABC is created?"]
        userPrompt =[]
        prompt = PromptTemplate(
                template=qa_template,
                input_variables=[],
                partial_variables={},
            )
        # userPrompt.append({
        #     "type":"text",
        #     "text": f"GIVEN DOCUMENT: {context_doc}"
        # })
        userPrompt.append({
            "type":"text",
            "text": "HERE IS THE DISCUSSION:"
        })

        for item in history:
            user_q = chatbot_a = ""
            if isinstance(item, dict):
                if str(item['answer']).find("!!!!") > -1: continue
                user_q = item['question']
                chatbot_a = item['answer']
            else:
                if str(item.answer).find("!!!!") > -1: continue
                user_q = item.question
                chatbot_a = item.answer
            userPrompt.append({"type":"text", "text": f"USER: {user_q}"})   
            userPrompt.append({"type":"text", "text": f"CHATBOT: {chatbot_a}"})   


        prompts = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt = prompt),
            HumanMessage(content=userPrompt),
            HumanMessagePromptTemplate.from_template("THE USER'S CURRENT QUESTION :\n\n  ```{question}``` ")
        ])
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY_DICT['AF-KB'], model_name="gpt-3.5-turbo-0125", temperature=0.5, request_timeout=120, max_tokens=300, response_format={ "type": "json_object" })
        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=False,
        )

        return chain