from typing import List
from langchain.chat_models import ChatOpenAI
from Modules.prompt import prompt
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
from Modules.extentions.automarker_processor import clean_question_text
from typing import List
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from Modules import reasonningKB
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
class QuestionKBSingle(BaseModel):
    question_header: str = Field(description="header content of question")
    question: str = Field(description="content of question")
    correct_answer: str = Field(description="correct answer of the question, always provide the correct answer. if there's answer options, ensure the correct answer is from the answer options for example if the answer option ['answer 1','answer 2'] so the correct answer should be 'answer 1' or 'answer 2'")
    difficulty_level: str = Field(description="easy, normal, hard")
    mark: str = Field(description="mark range from 1-5 mark")
    answer_options: List = Field(description="list of answer option if the reference question is multiple choice question. example: ['answer 1','answer 2','answer 3','answer 4']")
class QuestionKB(BaseModel):
    question_header: str = Field(description="header content of question")
    question: str = Field(description="content of question")
    correct_answer: str = Field(description="correct answer of the question, always provide the correct answer. if there's answer options, ensure the correct answer is from the answer options for example if the answer option ['answer 1','answer 2'] so the correct answer should be 'answer 1' or 'answer 2'")
    difficulty_level: str = Field(description="easy, normal, hard")
    mark: str = Field(description="mark range from 1-5 mark")
    answer_options: List = Field(description="list of answer option if the reference question is multiple choice question. example: ['answer 1','answer 2','answer 3','answer 4']")
    sub_questions: List[QuestionKBSingle] = Field(description="list of question")
class QuestionListKB(BaseModel):
    questions: List[QuestionKB] = Field(description="list of question")

class QuestionGeneratorChain:
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
    
    def question_parser_kb():
        parser = JsonOutputParser(pydantic_object=QuestionListKB)

        return parser
    
    def question_chain(chat_history,refprompt,question_type_id,version,type):
        chain_of_thought = """
            - Stage 1: Change parts of the reference questions including objects, numbers, question demands and the order of sentences. 
            - Stage 2: Simulate a brilliant, logical expert who is trying to answer the new question STEP-BY-STEP.  \
                    At each step, whenever possible, give a short explanation. For solving equations, please show clearly all the transformation steps taken to find all the variables needed. \
                    Finally get the final answer.  If the answer is a decimal number, you ARE NOT ALLOWED to round it. Just keep it as it is.
            - Stage 3: Translate the new question and its solution to the langugue used in the reference doc if it is not in English. The new question MUST be same language with the reference docs.
        """
        if version == 2:
            if question_type_id not in [5, 6, 7]:
                if type == "regen":
                    if question_type_id == 4:
                        chain_of_thought ="""
                            Step 1: Understand the context and details presented in the reference question. Determine what language is being used on the main question content and put it as the Original Language.
                            Step 2: Create a new, similar content for main question and question header section. Ensure this question is not an exact copy of the original and varies to maintain uniqueness.
                            Step 3: Next, Solve your new question to get the correct answer. For question pertaining to mathematics, do a proper calculation to ensure the accuracy.
                            Step 4: Next, put your result from solving the question as the correct answer for the new question. 
                            Step 5: Provide an answer options, and make sure your correct answer is provde as one of the answer option.
                            Step 6: Next, put your solving step into solution section.
                            Step 7: Translate each section to the Original Language
                        """
                    else:
                        chain_of_thought ="""
                            Step 1: Understand the context and details presented in the reference question. Determine what language is being used on the main question content and put it as the Original Language.
                            Step 2: Create a new, similar content for main question and question header section. Ensure this question is not an exact copy of the original and varies to maintain uniqueness.
                            Step 3: Next, Solve your new question to get the correct answer. For question pertaining to mathematics, do a proper calculation to ensure the accuracy.
                            Step 4: Next, put your result from solving the question as the correct answer for the new question. No need to provide answer options.
                            Step 5: Next, put your solving step into solution section.
                            Step 6: Translate each section to the Original Language
                        """
                else:
                    if question_type_id == 4:
                        chain_of_thought = """
                            Step 1: Rephrase the original question but remain the main subject of the question. 
                            Step 2: You don't need to provide question header, if the question reference doesn't have question header.
                            Step 3: You should provide the answer option. No need to put a,b,c,d i the answer option.
                            Step 4: The correct answer must be one of the answer in answer option.
                            Step 5: You should provide step by step solution on how to get the correct answer of question.
                        """
                    elif question_type_id == 10:
                        chain_of_thought = """
                            Step 1: Rephrase the original question but remain the main subject of the question. 
                            Step 2: You don't need to provide question header, if the question reference doesn't have question header.
                            Step 3: You don't need to provide answer options.
                            Step 4: No need to provide correct answer.
                            Step 5: You should provide step by step solution on how to get the correct answer of question.
                        """
                    else:
                        chain_of_thought = """
                            Step 1: Rephrase the original question but remain the main subject of the question. 
                            Step 2: You don't need to provide question header, if the question reference doesn't have question header.
                            Step 3: You don't need to provide answer options.
                            Step 4: The correct answer should be same with the reference question. Remember you only rephrase the question content.
                            Step 5: You should provide step by step solution on how to get the correct answer of question.
                        """
            else:
                if type == "regen":
                    chain_of_thought ="""
                        Step 1: Understand the context and details presented in the reference question. Determine what language is being used on the main question content and put it as the Original Language.
                        Step 2: Create a new, similar content for main question and question header section first. Ensure this question is not an exact copy of the original and varies to maintain uniqueness. Provide the answer blank by using HTML tag like what in the reference question provided to determine blank that need to  be answer in the question.
                        Step 3: Next, Solve your new question to get the correct answer for each blank in the new question. For question pertaining to mathematics, do a proper calculation to ensure the accuracy.
                        Step 4: Next, put your result from solving the question as the correct answer for each blank on new question. if the question is have options, provide other option on each blank and randomize the position of the correct answer.
                        Step 5: Next, put your solving step into solution section.
                        Step 6: And the final step, Translate each section to the Original Language
                    """
                else:
                    chain_of_thought = """
                        Step 1: Rephrase the original question but remain the main subject of the question.
                        Step 2: You don't need to provide question header, if the question reference doesn't have question header.
                        Step 3: You should provide <ans> tag to define the blank on question.
                        Step 4: If the question provide answer option, you should provide the answer option for each blank. No need to put a,b,c,d i the answer option.
                        Step 5: You should provide correct answer for each blank on the question.
                        Step 6: You should provide step by step solution on how to get the correct answer of question on each blank.
                    """
        qa_template = """
            You are an AI question generator. 
            You need to generate question that similar to the reference given and fulfill with user requirement.

            You MUST follow this step to generate new question:

            {chain_of_thought}
            
            Final Step: You should format the new question into: 
            question header:\n main question:\n answer options:\n correct answer:\n solution:\n difficulty level:\n mark:\n
            
            """
        
        systemPrompt = []
        userPrompt =[]
        systemPrompt.append({
            "type":"text",
            "text": qa_template
        })
        userPrompt.append({
            "type":"text",
            "text": "Here is my reference: \n"
        })
        refpromptjson = json.loads(refprompt)
        llm = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=OPENAI_API_KEY_DICT['QG'],temperature=0.2, request_timeout=120,max_tokens=4020)
        for item in refpromptjson:
            if item["type"] == "image_url":
                item["image_url"] = {"url":item["image_url"]}
                llm = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=OPENAI_API_KEY_DICT['VISION'], temperature=0, request_timeout=120,max_tokens=4050)
            else : item['text'] = clean_question_text(item['text']) + " \n "
            
            userPrompt.append(item)

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
            partial_variables={"chain_of_thought": chain_of_thought},
        )
        prompts = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt = prompt),
            HumanMessage(
                content=userPrompt
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ])

        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)
        
        for item in chat_history:
            memory.chat_memory.add_user_message(item.question)
            memory.chat_memory.add_ai_message(item.answer)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory,
        )

        return chain
    
    def convert_to_json_chain(question_type_id):
        qa_template = """
            Convert the provided question into a JSON instance that conforms to the JSON schema below.
             {{questions: [{{
                question_header:'content of question header. if not provided, fill with empty string.',
                question:'content of main question',
                difficulty_level:'easy, normal, hard',
                mark:'mark range from 1-5 mark',
                solution:'content of solution',
                answer_options:'list of answer option, example: ['answer 1','answer 2','answer 3','answer 4']. if not provided, fill with empty array',
                correct_answer:'correct answer of the question, if not provided, fill with empty string.'
                }}]
            }}

            RULES THAT YOU NEED TO OBEY:
            - IF there's image url provided in the content, do not remove it. only put the one that have <img> tag.
            - IF there's a latex expression in the question. change the format to: \(latex_expression\)
            - You should not generate any content that not exist from the provided question.

            MAKE SURE THE OUTPUT IS A JSON INSTANCE
            """
        if question_type_id in [5, 6, 7]:
            qa_template = """
                Convert the provided question into a JSON instance that conforms to the JSON schema below.
                {{questions: [{{
                    question_header:'content of question header section. if not provided, fill with empty string.',
                    question:'content of main question. make sure there's <ans> at least one on this section',
                    difficulty_level:'easy, normal, hard',
                    mark:'mark range from 1-5 mark',
                    solution:'content of solution',
                    answer_options:'list of answer option for each blank, example: ['answeroption 1 blank 1','answeroption 2 blank 1'],['answeroption 1 blank 2','answeroption 2 blank 2']. if not provided, fill with empty array',
                    correct_answer: '['correct answer for blank 1','correct answer for blank 2',etc.] ensure the length of correct answer aligned with total of blank that provided in the question'
                    }}]
                }}

               RULES THAT YOU NEED TO OBEY:
                - IF there's image url provided in the content, do not remove it. only put the one that have <img> tag.
                - IF there's a latex expression in the question. change the format to: \(latex_expression\)
                - You should not generate any content that not exist from the provided question.

                MAKE SURE THE OUTPUT IS A JSON INSTANCE
                """
        systemPrompt = []
        userPrompt =[]
        systemPrompt.append({
            "type":"text",
            "text": qa_template
        })
        llm = ChatOpenAI(model_name="gpt-4-turbo", openai_api_key=OPENAI_API_KEY_DICT['QG'], temperature=0, 
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
            HumanMessagePromptTemplate.from_template("Here is my question: \n {question}"),
        ])
        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
        )

        return chain
    
    def gen_only_question_chain(chat_history, refprompt, rephrase='full'):
        #Hyper-params
        temperature = 1.5 if rephrase in ['full']  else 0 if rephrase in ['noun'] else 0.5

        #PROMPT
        qa_template=""
        if rephrase.strip().lower() in ['full']:
            qa_template = """
                You are an AI math question generator. 
                You need to generate a new different question based on the given reference question  by following steps:
                    1. Determine the main language is used in the reference text, let call that language as T. (If you not sure, set T='english')
                    2. If the question contain images, please carefully describe the image, which allows student to redraw the image. PLEASE use language T
                    3. ALWAYS CHANGE  suit noun-words, figures, and question DEMAND in the reference questions. DO NOT change the information of the image description. 
                """
        elif rephrase.strip().lower() in ['object']:
            qa_template = """
                You are an AI math question rephraser. 
                You need to generate question that similar to the given reference question by following steps:
                    1. Determine the main language is used in the reference text, let call that language as T. (If you not sure, set T='english')
                    2. If the question contain images, please carefully describe the image, which allows student to redraw the image. PLEASE use language T
                    3. CHANGING ONLY objects(nouns) and figures mentioned in the reference questions. DO NOT change the information of the image description and structure of the question. 
                """
        elif rephrase.strip().lower() in ['noun']:
            qa_template = """
                You are an AI paraphraser. 
                You need to change nouns in the given reference question by following steps:
                    1. Determine the main language is used in the reference text, let call that language as T. (If you not sure, set T='english')
                    2. If the question contain images, please carefully describe the image, which allows student to redraw the image. PLEASE use language T
                    3. ALWAY PRESERVE figures/numbers in the reference question in new question.
                    4. CHANGING ONLY words IF IT IS NOUN or Human Name in the reference questions to new factor/objects. DO NOT change the information of the image description 
                For example, If it say: "There are 10 dogs", You only change nouns without change figures as "Ben has 10 cats"
                    
                """

        else: raise '[ERR] Rephrase parameter in Math Question Paraphaser is invalid'
        



        qa_template +="""
            The output should be formatted as a JSON instance that conforms to the JSON schema below.
                {{questions: [{{
                'lang':'language T' ,
                'question_header':'MUST assign a header content for the new question',
                'question': 'the new question and The description of the given image (if it's attached). All texts are in language T ',
                'difficulty_level':'easy, normal, hard',
                'mark':'mark range from 1-5 mark'
                }}]
            }}

            MAKE SURE THE OUTPUT IS A JSON INSTANCE AND the content NOT contain noise characters.
            """


        # REFERENCE
        userPrompt =[]
        prompt = PromptTemplate(
                template=qa_template,
                input_variables=[],
                partial_variables={},
            )
        userPrompt.append({
            "type":"text",
            "text": "Here is my reference:"
        })
        refpromptjson = json.loads(refprompt)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125",#"gpt-4-0125-preview", 
                        openai_api_key=OPENAI_API_KEY_DICT['QG'], temperature=temperature, request_timeout=120,max_tokens=1020,
                        response_format={ "type": "json_object" }) #gpt-4-0125-preview gpt-3.5-turbo-16k
        for item in refpromptjson:
            if item["type"] == "image_url":
                llm = ChatOpenAI(model_name="gpt-4-vision-preview", openai_api_key=OPENAI_API_KEY_DICT['VISION'], temperature=0.5, request_timeout=120,max_tokens=4050)
            elif "question header" in item['text'].lower() or "question type" in item['text']: continue 
            else: item['text'] = clean_question_text(item['text'])
            userPrompt.append(item)

        prompts = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt = prompt),
            HumanMessage(content=userPrompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ])
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)
        for item in chat_history:
            memory.chat_memory.add_user_message(item.question)
            memory.chat_memory.add_ai_message(item.answer)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory,
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
    
    def predict_question():
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY_DICT['AF'],
                        model_name="gpt-3.5-turbo-0125", temperature=0, request_timeout=120, max_tokens=4096,response_format={"type":"json_object"})
        qa_template = """
        Given student's quiz result, student's discussion, and student's question.
        Your purpose is to predict what is related question from student's quiz result that being asked by student.
        Return the related question into a JSON format as follow:
        {{
            related_question: "content of related question, MUST include question number,question,answer option, correct answer, student answer, and solution",
        }}
        
        =====
        Student's quiz result:
        {refprompt}

        student's discussion:
        {chat_history}
        """

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
        )

        prompts = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate(prompt = prompt),
                # HumanMessage(
                #     content=f"HERE IS MY LIST OF QUESTION: {refprompt}"
                # ),
                # The `variable_name` here is what must align with memory
                # MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("student's question: {question}")
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

    def check_model_destination4quizreview(chat_history, refprompt, image_library):
        """
        This function is design, first for aming to use in chain_quiz_review
        Given context of chat (history and reference) and  user's question.
        Specify if the user's demand needs to use vision-gpt or not
        """
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY_DICT['AF'],
                        model_name="gpt-3.5-turbo-0125", temperature=0, request_timeout=120, max_tokens=4000,response_format={"type":"json_object"})
        # Additionally, analyze the provided answer for good points within the specified range. Do the checking specifically, don't give points that in general.

        qa_template = """
        Given student's quiz result, student's discussion, and student's question.
        Your purpose is to predict what is related question from student's quiz result that being asked by student.
        Return the related question into a JSON format as follow:
        {{
            image_url: ["list of related image url"],
        }}
        
        =====
        Student's quiz result:
        {refprompt}

        student's discussion:
        {chat_history}

        student's question:
        {question}
        """

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
        )
        # userPrompt =[]
        # userPrompt.append({
        #     "type":"text",
        #     "text": "HERE IS THE LIST OF QUESTION: "
        # })
        # userPrompt.append({"type":"text","text":refprompt})

        # userPrompt.append({
        #     "type":"text",
        #     "text": f"\n HERE IS THE IMAGE LIBRARY: {image_library}"
        # })

        prompts = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate(prompt = prompt),
                # HumanMessage(
                #     content=f"HERE IS MY LIST OF QUESTION: {refprompt}"
                # ),
                # The `variable_name` here is what must align with memory
                # MessagesPlaceholder(variable_name="chat_history"),
                # HumanMessagePromptTemplate.from_template("{question}")
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

    def predict_math_question():
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY_DICT['AF'],
            model_name = "gpt-3.5-turbo-0125",
            temperature = 0,
            request_timeout = 120,
            max_tokens = 50
        )

        qa_template = """
        Given a question, your task is to predict whether the given question is a math question or not. \
        A question can be considered a math question if it involves numbers, equations, or mathematical operations. The question is not strictly limited to question that has mathematical symbols, but also for questions that require mathematical reasoning. \
        The output MUST ALWAYS BE FORMATTED AS A STRING: 'type_of_question' without extra explaination. The value of your output will be either 'math' or 'general' without the '' mark as a string. So it will be either math or general.
        """

        prompt = PromptTemplate(
            template = qa_template,
            input_variables = [],
        )
        prompts = ChatPromptTemplate(
            messages = [
                SystemMessagePromptTemplate(prompt = prompt),
                HumanMessagePromptTemplate.from_template("Here is the question you need to analyze:\n\n  ```{question}``` ")
            ]
        )
        chain = LLMChain(
            llm = llm,
            prompt = prompts,
            verbose = False,
        )

        return chain
    
    def convert_to_json_chain_kb():
        qa_template = """
            Your purpose is to convert provided question into json, follow this rules to convert the provided question:
            - Article/passage and question instruction MUST be put under question header.
            - IF there's a question that have same article and instruction make it as one question and put the rest as the subquestion.
            - IF there's a latex expression in the question. change the format to: \(latex_expression\)
            - You should show return all the content that provided in the JSON. For example, if there's a pssage, you should put all the passage content in the question header.

            Convert the provided questions into a JSON instance that conforms to the JSON schema.
             {{questions: [{{
                question_header:'article and instruction of the question',
                question:'main content of the question',
                difficulty_level:'easy, normal, hard',
                mark:'mark range from 1-5 mark',
                answer_options:'list of answer option, example: ['answer 1','answer 2','answer 3','answer 4']. if not provided, fill with empty array',
                correct_answer:'correct answer of the question, if not provided, fill with empty string.',
                sub_question:[{{question:'',correct_answer:'',answer_options:'',difficulty_level:'',mark:''}}]
                }}]
            }}

            
            """
        
        systemPrompt = []
        userPrompt =[]
        systemPrompt.append({
            "type":"text",
            "text": qa_template
        })
        userPrompt.append({
            "type":"text",
            "text": """
                Follow this rules to convert my question:
                - Article, passage and instruction MUST be put under question header.
                - IF there's a question that have same article and instruction make it as one question and put the rest as the subquestion.
                - IF there's a latex expression in the question. change the format to: \(latex_expression\)
                - You should not generate any content that not exist from the provided question.
            """
        })
        llm = ChatOpenAI(model_name="gpt-4-0125-preview", openai_api_key=OPENAI_API_KEY_DICT['QG'], temperature=0, 
                         request_timeout=120, response_format={ "type": "json_object" })


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
            HumanMessagePromptTemplate.from_template("Here is my question: \n {question}"),
        ])
        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
        )

        return chain
    
    def convert_to_json_chain_kb_simple():
        qa_template = """
            Your purpose is to convert provided question into a JSON format. You must follow these step before convert it into a json:
            1. Put question number 1 as the main question
            2. Put another question as the sub_question of question number 1.
            3. Convert the questions into a JSON instance that conforms to the JSON schema.
             {{questions: [{{
                question_header:'instruction of the question',
                question:'main content of the question',
                difficulty_level:'easy, normal, hard',
                mark:'mark range from 1-5 mark',
                answer_options:'list of answer option, example: ['answer 1','answer 2','answer 3','answer 4']. if not provided, fill with empty array',
                correct_answer:'correct answer of the question, if not provided, fill with empty string.',
                sub_question:'put all the question except question number 1 into sub_questions with following structure:[{{question:'',difficulty_level:'',mark:'',answer_options:'',correct_answer:''}}]''
            }}

            
            """
        
        systemPrompt = []
        systemPrompt.append({
            "type":"text",
            "text": qa_template
        })
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", openai_api_key=OPENAI_API_KEY_DICT['QG'], temperature=0, 
                         request_timeout=120, response_format={ "type": "json_object" })


        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
            partial_variables={},
        )
        prompts = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt = prompt),
            HumanMessagePromptTemplate.from_template("Here is my question: \n {question}"),
        ])
        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
        )

        return chain
    
    def restructure_question():
        qa_template = """
            List out the question with this structure on each question as follow:
            - Question number: number of question
            - Question header: passage,article, and instruction of the question
            - Question:main content of the question
            - Answer: correct answer of the question
            - Difficulty level: difficulty level of question easy/normal/hard
            - Subquestion : question number, question, answer, difficulty level

            Follow this rules to convert my question:
                - Article, passage and instruction MUST be put under question header.
                - IF there's a question that have same article and instruction make it as one question and put the rest as the subquestion.
                - IF there's a latex expression in the question. change the format to: \(latex_expression\)
                - You should not generate any content that not exist from the provided question.
            """
        systemPrompt = []
        userPrompt =[]
        systemPrompt.append({
            "type":"text",
            "text": qa_template
        })
        llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY_DICT['QG'], temperature=0, 
                         request_timeout=120,max_tokens=4000)

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
            HumanMessagePromptTemplate.from_template("Here is my question: \n {question}"),
        ])
        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
        )

        return chain
