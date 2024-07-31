from typing import List
# from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec
from langchain.agents import create_json_agent
from Modules.prompt import prompt
from langchain.chains import LLMChain
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
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from typing import List
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from Modules import reasonningKB
from Modules.question_generator.chain import QuestionGeneratorChain
import config 
from langchain_core.output_parsers import JsonOutputParser
from langchain.callbacks import get_openai_callback
# from langchain.embeddings import GPT4AllEmbeddings

KB = reasonningKB.DocRetrievalKnowledgeBase(pdf_source_folder_path=config.TMPPATH)
class Question(BaseModel):
    question: str = Field(description="content of question")
    correct_answer: str = Field(description="correct answer of the question, always provide the correct answer. if there's answer options, ensure the correct answer is from the answer options for example if the answer option ['answer 1','answer 2'] so the correct answer should be 'answer 1' or 'answer 2'")
    difficulty_level: str = Field(description="easy, normal, hard")
    mark: str = Field(description="mark range from 1-5 mark")
    answer_options: List = Field(description="list of answer option if the reference question is multiple choice question. example: ['answer 1','answer 2','answer 3','answer 4']")
    solution: str = Field(description="short explanation of the correct answer")
class QuestionList(BaseModel):
    questions: List[Question] = Field(description="list of question")
class QuestionNew(BaseModel):
    question_header: str = Field(description="header content of question")
    question: str = Field(description="content of question")
    correct_answer: str = Field(description="correct answer of the question, always provide the correct answer. if there's answer options, ensure the correct answer is from the answer options for example if the answer option ['answer 1','answer 2'] so the correct answer should be 'answer 1' or 'answer 2'")
    difficulty_level: str = Field(description="easy, normal, hard")
    mark: str = Field(description="mark range from 1-5 mark")
    answer_options: List = Field(description="list of answer option if the reference question is multiple choice question. example: ['answer 1','answer 2','answer 3','answer 4']")
    solution: str = Field(description="short explanation of the correct answer")
class QuestionListNew(BaseModel):
    questions: List[QuestionNew] = Field(description="list of question")
class Finding(BaseModel):
    type: str = Field(description="invalid_content or incorrect_correct_answer or invalid_solution")
    suggestion: str = Field(description="suggestion for the invalid_content or incorrect_correct_answer or invalid_solution")
class QuestionCheckingList(BaseModel):
    finding: List[Finding] = Field(description="list of incorrect finding")
class Rubric(BaseModel):
    mark: str = Field(description="the total mark of student answer")
    fullmark: str = Field(description="full mark of the question")
    feedback_for_tutor: str = Field(description="A comprehensive feedback for student.")
    feedback_for_student: str = Field(description="A comprehensive feedback for student.")
    # translated_feedback_tutor: str = Field(description="Feedback that explain in english language.")
    # translated_feedback_student: str = Field(description="Feedback that explain in english language.")


class Checking(BaseModel):
    # type: str = Field(description="good_points or to_improve")
    target: str = Field(description="the finding that have suggestion")
    # target_start: str = Field(description="the finding start on which position from the answer")
    # target_end: str = Field(description="the finding end on which position from the answer")
    suggestion: str = Field(description="suggestions for improvement and explanation for good point")
class CheckingList(BaseModel):
    checking: List[Checking] = Field(description="list of checking")
class LearningContentQuiz(BaseModel):
    question: str 
    answer_option: List[str]
    correct_answer: str 
    solution: str
class LearningContent(BaseModel):
    lesson_title: str
    explanation: str 
    case_study: List[LearningContentQuiz]

class Chain:
    @staticmethod
    def chabot_chain(docs):
        embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY_DICT['default'])
        vectors = FAISS.from_documents(docs, embeddings)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
        # Create the conversational retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectors.as_retriever(search_type="similarity", search_kwargs={"k":4}),
            verbose=True,
            return_source_documents=True,
            max_tokens_limit=15000,
            combine_docs_chain_kwargs={'prompt': prompt.generate_qa_prompt()}
        )

        return chain
    
    def question_chain():
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['default'], model_name="gpt-3.5-turbo-0125", temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt.generate_question_prompt())
        return chain
    
    def summary_chain():
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'], temperature=0, model_name="gpt-3.5-turbo-0125")
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt.summary_prompt())
        return chain
    
    def grammar_chain():
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'], temperature=0, model_name="gpt-3.5-turbo-0125")
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt.generate_grammar_prompt())
        return chain
    
    def content_checking_chain():
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'], temperature=0, model_name="gpt-3.5-turbo-0125")
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt.content_checking_prompt())
        return chain
    
    def readability_chain():
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'], temperature=0, model_name="gpt-3.5-turbo-0125")
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt.readability_score_prompt())
        return chain
    
    def suggestion_comment_chain():
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'], temperature=0, model_name="gpt-3.5-turbo-0125")
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt.suggestion_prompt())
        return chain
    
    def question_v2_chain():
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['QG'],model_name="gpt-3.5-turbo-16k", temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt.generate_question_v2_prompt())
        return chain
    
    def content_chain():
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'], model_name="gpt-3.5-turbo-16k", temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt.generate_content_prompt())
        return chain
    
    def content_learning_chain():
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'], model_name="gpt-4-0125-preview", temperature=0,max_tokens=4050, response_format={ "type": "json_object" })
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt.generate_content_learning_prompt())
        return chain
    
    def question_json_agent(data):
        data = json.loads(data)
        # print(data)
        json_spec = JsonSpec(dict_=data, max_value_length=40000)
        json_toolkit = JsonToolkit(spec=json_spec)
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['default'],model_name="gpt-3.5-turbo-16k", temperature=0)
        json_agent_executor = create_json_agent(
            llm,
            toolkit=json_toolkit,
            verbose=True
        )
        return json_agent_executor
    
    def chabot_chain_als(chat_history):
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'],
                        model_name="gpt-3.5-turbo-0125", temperature=0)
        #   You are a well educated Virtual Tutor called HeyJen. 
        #     You will help student to understand the upcoming exam.
        #     Generate a series of comprehensive lessons to help students prepare for an upcoming exam 
        #     Each lesson should provide thorough and detailed explanations of the subtopic or concept related to the quiz material. 
        #     The explanations should be comprehensive and leave no important details uncovered. 
        #     Please include extensive explanations, and example in each lesson to ensure that students grasp the content effectively.
        #     Do not give example and quiz from the upcoming quiz. You can give similar question but do not duplicate the question.
        #     Avoid using any image-based question and example as my platform does not support images.

        #     Please follow this scenario:
        #     1. When user chat for the first time, please introduce yourself and tell them what will they learn on this lesson session.
        #     2. After user is ready, explain them the lesson. at the end of the explanation, ask them if they already understand about the lesson or not. 
        #     3. if they already understand ask them with the mini quiz with set of 3 questions. Do not start mini quiz before they understand.
        #     4. please tell to them if they are wrong or correct and give the explanation
        #     5. if the student finish all the mini quiz, ask them if they want to get more mini quiz or go to next lesson
        #     6. if they are in the beginning of session and they ask out of scenario, do not answer the question. after that ask them again if they are ready to start the lesson or not.
        #     7. if they are in the process of lesson session and they ask out of scenario, do not answer the question. after that ask them again if they are ready to go back to the lesson session or not.
        qa_template = """
            You are a well educated Learning Buddy called Jen. 
            You will help student to understand the upcoming exam.
            Student already listen to your explanation and  finish their case study. 

            Please follow this rules:
            1. If the user answer is wrong on their case study, please explain why their answer is wrong
            2. If the user ask anything outside the learning material, please ignore it
            
            ========
            Learning Material:
            {learning_reference}
            Case Study:
            {case_study}
            Upcoming exam:
            {reference}
            """
        
        prompts = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    qa_template,
                ),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)
        
        for item in chat_history:
            memory.chat_memory.add_user_message(item.question)
            memory.chat_memory.add_ai_message(item.answer)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory
        )

        return chain
    
    def chabot_chain_als_result(chat_history,refprompt):
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['VISION'],
            model_name="gpt-4-vision-preview", temperature=0,max_tokens=4010)
        qa_template = """
            You are a well educated Learning Buddy called Jen. 
            You will help student to review their quiz result.

            Please follow this rules:
            1. When user chat for the first time,
            tell them that in this step they will do the review phase.
            then tell them that they can go to their learning module whenever the learning modules are ready.
            then tell them that they can choose which question they want to review by click button "Review Question"
            2. If they asking out of learning scope, do not answer their question and ask them politely to ask only around the quiz result.
            3. if they ask about the wrong question, please explain to them why their answer is wrong and give them the solution.
            4. When user ask to review their answer on selected question, no need to repeat the question on your answer
            
            """
        # print(refprompt)
        systemPrompt = []
        # userPrompt =[]
        systemPrompt.append({
            "type":"text",
            "text": qa_template
        })
        # refpromptjson = json.loads(refprompt)
        # for item in refpromptjson:
        #     userPrompt.append(item)
        userPrompt = refprompt
        
        prompts = ChatPromptTemplate.from_messages([
            SystemMessage(
                content=systemPrompt
            ),
            HumanMessage(
                content=userPrompt
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ])
        
        # prompts = ChatPromptTemplate(
        #     messages=[
        #         SystemMessagePromptTemplate.from_template(
        #             qa_template,
        #         ),
        #         # The `variable_name` here is what must align with memory
        #         MessagesPlaceholder(variable_name="chat_history"),
        #         HumanMessagePromptTemplate.from_template("{question}")
        #     ]
        # )
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)
        
        for item in chat_history:
            memory.chat_memory.add_user_message(item.question)
            memory.chat_memory.add_ai_message(item.answer)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory
        )

        return chain
    
    def chabot_chain_als_practice(chat_history,refprompt):
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['VISION'],
            model_name="gpt-4-vision-preview", temperature=0,max_tokens=4010)

        qa_template = """
            You are a well educated Learning Buddy called Jen. 
            You will help student to review their quiz result.

            Please follow this rules:
            1. When user chat for the first time, 
            tell them that in this step they will do the review phase.
            if their score below 75%, engage them to review their incorrect question.
            else told them that they can take the review test whenever they are ready.
            then tell them that they can choose which question they want to review by click button "Review Question"
            2. If they asking out of learning scope, do not answer their question and ask them politely to ask only around the quiz result.
            3. if they ask about the wrong question, please explain to them why their answer is wrong and give them the solution.
            
            Result Score: {result_score}
            ========
            """
        systemPrompt = []
        userPrompt =[]
        systemPrompt.append({
            "type":"text",
            "text": qa_template
        })
        refpromptjson = json.loads(refprompt)
        for item in refpromptjson:
            userPrompt.append(item)
        
        prompts = ChatPromptTemplate.from_messages([
            SystemMessage(
                content=systemPrompt
            ),
            HumanMessage(
                content=userPrompt
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ])
        # prompts = ChatPromptTemplate(
        #     messages=[
        #         SystemMessagePromptTemplate.from_template(
        #             qa_template,
        #         ),
        #         # The `variable_name` here is what must align with memory
        #         MessagesPlaceholder(variable_name="chat_history"),
        #         HumanMessagePromptTemplate.from_template("{question}")
        #     ]
        # )
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)
        
        for item in chat_history:
            memory.chat_memory.add_user_message(item.question)
            memory.chat_memory.add_ai_message(item.answer)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory
        )

        return chain
    
    def question_v2_chain_chat(chat_history):
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'],
            model_name="gpt-3.5-turbo-16k", temperature=0, request_timeout=600)
        qa_template = """
            You are an AI question generator. 
            Please generate a set of questions that are similar to those in my question bank, but make sure they are not exact duplicates. 
            These questions should include a mix of difficulty levels: easy, medium, and hard. 
            Please ensure the questions are diverse and do not overlap with existing questions in my bank. 
            My question bank includes questions like:
            {reference}

            Please generate the questions and format it as follows:
            [{{
                question:'content of question',
                correct_answer:'correct answer of the question',
                difficulty_level:'easy, normal, hard',
                mark:'mark range from 1-5 mark',
                answer_options:['always provide answer options when user ask for multiple choice question'],
                solution:'short explanation of the correct answer',
                question_reference_id:'if theres question_id attach on the context'
            }}].
            Always return in json format so i can encode it.
            If there is no question in the combination, return an empty list like [].
            """
        
        prompts = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    qa_template,
                ),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)
        
        for item in chat_history:
            memory.chat_memory.add_user_message(item.question)
            memory.chat_memory.add_ai_message(item.answer)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory
        )

        return chain
    
    def question_parser():
        parser = JsonOutputParser(pydantic_object=QuestionList)
        return parser
    
    def question_parser_new():
        parser = JsonOutputParser(pydantic_object=QuestionListNew)
        return parser
    
    def question_checking_parser():
        parser = JsonOutputParser(pydantic_object=QuestionCheckingList)
        return parser
    
    def learning_content_parser():
        parser = JsonOutputParser(pydantic_object=LearningContent)
        return parser

    def question_v3_chain_chat(chat_history,refprompt):
        qa_template = """
            You are an AI question generator. 
            Please generate question that similar to the reference given and fullfil with user requirement.

            - Stage 1: Change parts of the reference questions including objects, numbers, question demands and the order of sentences. 
            - Stage 2: Simulate a brilliant, logical expert who is trying to answer the new question STEP-BY-STEP.  \
                    At each step, whenever possible, give a short explanation. For solving equations, please show clearly all the transformation steps taken to find all the variables needed. \
                    Finally get the final answer.  If the answer is a decimal number, you ARE NOT ALLOWED to round it. Just keep it as it is.
            
            As for the new question, please generate the question and solution using the same language used in the reference question.

            The output should be formatted as a JSON instance that conforms to the JSON schema below.
            {{questions: [{{
                question_header:'header content of question',
                question:'content of question',
                difficulty_level:'easy, normal, hard',
                mark:'mark range from 1-5 mark',
                solution:'The steps to find the final answer. If the question is a math question or a question that involves calculation, please make sure that the calculation is done correctly.',
                answer_options: 'list of answer option if the reference question is multiple choice question. example: ['answer 1','answer 2','answer 3','answer 4']',
                correct_answer: 'Always provide the correct answer of the new question found in the final steps of the solution. if there's answer options, ensure the correct answer is from the answer options for example if the answer option ['answer 1','answer 2'] so the correct answer should be 'answer 1' or 'answer 2''
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
        userPrompt.append({
            "type":"text",
            "text": "Here is my reference:"
        })
        refpromptjson = json.loads(refprompt)
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['QG'],
                        model_name="gpt-4-0125-preview", temperature=0.2, request_timeout=120,max_tokens=4050, response_format={ "type": "json_object" })
        for item in refpromptjson:
            if item["type"] == "image_url":
                llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['VISION'],
                    model_name="gpt-4-vision-preview", temperature=0, request_timeout=120,max_tokens=4050)
            userPrompt.append(item)


        parser = JsonOutputParser(pydantic_object=QuestionListNew)

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
            partial_variables={"format_instructions": parser.get_format_instructions()},
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
    
    def question_check_v3_chain_chat(refprompt):
        # 1. Invalid Content:
        #     - Detect if there's any typo on the content.
        #     - Identify and flag questions with inappropriate or irrelevant content. Examples include offensive language or content unrelated to the topic.
        #     2. Incorrect Correct Answer:
        #     - Detect and highlight questions where the correct_answer is incorrect or doesn't align with the provided options or no correct answer provided.
        #     - Suggest the right correct_answer for the question if it flagged as incorrect_correct_answer
        #     3. Invalid Solution:
        #     - Recognize and flag questions with invalid or incomplete or empty solutions.
        #     - Provide the right solution for the question if it flagged as invalid_solution.
        #     - Make sure your solution is understandable for a Primary or Secondary students.
        qa_template = """
            You are an AI content checker.
            You need to check a question for potential issues, including invalid content, incorrect correct answers, and invalid solution.
            To do the checking, you MUST always follow these step one by one:

            Step 1: Understand the context and details presented in the given question.
            Step 2: Check the content of the question. if there's any typo or irrelevant content, Mark the question has an invalid content and suggest the correct one.
            Step 3: Next, solve the existing question, and see if the correct answer match with the given correct answer. if not match, mark the question has an invalid correct answer and suggest your correct answer.
            Step 4: Next, put your solving step into a solution. match your solution and given solution, if the existing solution is wrong or empty, mark the question has an invalid solution and suggest your solution.
            Step 5: Final step, list out your finding into type of the error (invalid_content/incorrect_correct_answer/invalid_solution), following with the suggestion.

            return only list of your finding.
            """
        systemPrompt = []
        userPrompt =[]
        systemPrompt.append({
            "type":"text",
            "text": qa_template
        })
        userPrompt.append({
            "type":"text",
            "text": "Here is my question:"
        })
        refpromptjson = json.loads(refprompt)
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'],
            model_name="gpt-4-turbo", temperature=0, request_timeout=120,max_tokens=4050)
        for item in refpromptjson:
            if item["type"] == "image_url":
                llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['VISION'],
                    model_name="gpt-4-turbo", temperature=0, request_timeout=120,max_tokens=4050)
            userPrompt.append(item)

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
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

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory,
        )

        return chain

    def question_check_v3_json_parser():
        qa_template = """
            Convert the provided finding into a json with format below:
            {{finding: [{{
                type:'invalid_content or incorrect_correct_answer or invalid_solution',
                suggestion:'the correct content for the invalid_content or the correct correct_answer for incorrect_correct_answer or the correct solution for invalid_solution'
                }}]
            }}
            """
        systemPrompt = []
        systemPrompt.append({
            "type":"text",
            "text": qa_template
        })
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'],
            model_name="gpt-4-turbo", temperature=0, request_timeout=120,max_tokens=4050, response_format={ "type": "json_object" })

        parser = JsonOutputParser(pydantic_object=QuestionCheckingList)

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        prompts = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate(prompt = prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ])

        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory,
        )

        return chain
    
    def auto_solution_chain(chat_history):
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'],
            model_name="gpt-3.5-turbo-0125", temperature=0, request_timeout=120)
        qa_template = """
            Student is taking practice.
            Please explain the solution each answer of student on each question.
            If the student answer incorrectly, please explain why the student's answer is wrong.
            Please explain in short paragraph for student called {student_name}.
            Make sure your answer is cheerful and personalised.

            Practice:
            {reference}
            """
        
        prompts = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    qa_template,
                ),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)

        for item in chat_history:
            memory.chat_memory.add_user_message(item.question)
            memory.chat_memory.add_ai_message(item.answer)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory
        )

        return chain
    
    def first_level_checking_parser():
        parser = JsonOutputParser(pydantic_object=CheckingList)
        return parser
    
    def first_level_checking(chat_history):
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'],
            model_name="gpt-4-0125-preview",max_tokens=4050 , temperature=0, request_timeout=120, response_format={ "type": "json_object" })
        # Additionally, analyze the provided answer for good points within the specified range. Do the checking specifically, don't give points that in general.

        qa_template = """
            Analyze the provided answer for grammatical errors, clarity issues, and factual inaccuracies within the specified character range. Prioritize critical errors that could lead to misunderstandings. List them out with explanations and suggestions for improvement, including alternative phrasing, missing information, and corrected factual points. 
            You should explain the suggestion in language {language}
            The output should be formatted as a JSON instance that conforms to the JSON schema below.
            {{checking: [{{
                target:'word that need to be improve',
                suggestion:'suggestions for improvement and explanation for the target'
                }}]
            }}

            If there's no error found, please return with an empty array like this []
            """
        parser = JsonOutputParser(pydantic_object=CheckingList)

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        prompts = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate(prompt = prompt),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)

        for item in chat_history:
            memory.chat_memory.add_user_message(item.question)
            memory.chat_memory.add_ai_message(item.answer)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory
        )

        return chain
    
    def translate_chain(chat_history):
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['CONTENT_CHECKING'],
                        model_name="gpt-3.5-turbo-0125", temperature=0, request_timeout=120)
        # Additionally, analyze the provided answer for good points within the specified range. Do the checking specifically, don't give points that in general.

        qa_template = """
            Translate the text given to desired language by user
            """

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
        )

        prompts = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate(prompt = prompt),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)

        for item in chat_history:
            memory.chat_memory.add_user_message(item.question)
            memory.chat_memory.add_ai_message(item.answer)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory
        )

        return chain
    
    def image_description_chain(chat_history,refprompt):
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['VISION'],
                        model_name="gpt-4-vision-preview", temperature=0, request_timeout=120, max_tokens=4050)
        # Additionally, analyze the provided answer for good points within the specified range. Do the checking specifically, don't give points that in general.

        qa_template = """
            Given image that attached as a content or a solution for question in a quiz. 
            Your purpose is to give a clear description of the image given.
            If the image is containing a text, convert all the text to the description.
            """

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
        )
        userPrompt =[]
        userPrompt.append({
            "type":"text",
            "text": "Here is my image:"
        })
        refpromptjson = json.loads(refprompt)
        for item in refpromptjson:
            userPrompt.append(item)

        prompts = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate(prompt = prompt),
                HumanMessage(
                    content=userPrompt
                ),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)

        for item in chat_history:
            memory.chat_memory.add_user_message(item.question)
            memory.chat_memory.add_ai_message(item.answer)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory
        )

        return chain

    async def check_image_library(chat_history,refprompt,question, image_library):
        predict_question_chain = QuestionGeneratorChain.predict_question()
        predict_image_chain = QuestionGeneratorChain.check_model_destination4quizreview(chat_history=chat_history, refprompt=refprompt, image_library=image_library)
        related_question = ""
        with get_openai_callback() as cb:
            history = ""
            for chat in chat_history:
                history += "Student: "+chat.question+" \n"
                history += "AI: "+chat.answer+" \n"

            predict_question_response = await predict_question_chain.arun(question=question, chat_history=history, reference=refprompt, refprompt=refprompt, image_library=image_library)
            print("PREDICT QUESTION:",predict_question_response)
            predict_question_response = json.loads(predict_question_response)

            predict_image_response = await predict_image_chain.arun(question=question, chat_history=history, reference=refprompt, refprompt=refprompt, image_library=image_library)
            print("PREDICT IMAGE:",predict_image_response)
            predict_image_response = json.loads(predict_image_response)

            related_question = predict_question_response["related_question"]
            if predict_image_response["image_url"]:
                for url in predict_image_response["image_url"]:
                    if not any(item["image_url"] == url for item in image_library):
                        imagePrompt = []
                        imagePrompt.append({
                            "type":"image_url",
                            "image_url": url
                        })
                        imagePrompt = json.dumps(imagePrompt)
                        image_chain = Chain.image_description_chain(chat_history=[], refprompt=imagePrompt)
                        responseImage = await image_chain.arun({"question":"","chat_history":[],"reference":""})
                        image_library.append({
                            "image_url":url,
                            "description":responseImage
                        })
            print("new image library")
            print(image_library)

            related_image = []
            if predict_image_response["image_url"]:
                for url in predict_image_response["image_url"]:
                    for item in image_library:
                        if item["image_url"] == url:
                            related_image.append({
                                "image_url":url,
                                "description": item["description"]
                            })

        return {"image_library":image_library,"related_question":related_question,"related_image":related_image}

    async def chabot_chain_quiz_result(chat_history,refprompt,language, question=None, image_library=[]):
        # llm = ChatOpenAI(model_name="gpt-4-vision-preview", temperature=0,max_tokens=4010)
        baseLanguage = 'English'
        if language != '':
            baseLanguage = language 
      
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['AF'], model_name="gpt-3.5-turbo-0125", temperature=0, request_timeout=120,max_tokens=4096)
    #    "To provide you with a detailed explanation of question [Question Number], the correct answer is [Correct Answer], and here's why: [Explanation]. Considering the answer you selected, [Student's Answer], [Explanation for why it might be incorrect or the rationale if it was correct]."
        # qa_template = f"""
        #     You are well educated learning buddy called Jen. 
        #     Your purpose is to help student reviewing their student result based on the given student quiz result.
            
        #     To assisting the review question, YOU MUST FOLLOW the rules below:
        #     1. Introduction: 
        #     "Hello! I'm here to help you review your quiz results. Just ask me about any questions you're curious about, or if you want explanations for specific answers. Let's get started!"

        #     2. Initial Response for Direct Quiz Content Questions:
        #     For the first time a student asks about a specific question:
        #     "Looking at question [Question Number], the correct answer is [Correct Answer]. Here’s the reason: [Explanation]. Considering you chose [Student's Answer], let's talk about why it might not be the best choice: [Explanation for why it might be incorrect or rationale if it was correct]."

        #     3. Handling Repeat Questions:
        #     If the student asks about the same question again:
        #     - "Sure, let’s go over question [Question Number] again. Remember, the key point here is [Brief Recap of Explanation]. It seems like [Particular Aspect] might still be a bit unclear. Would you like to dive deeper into this part?"

        #     4. Handling Out-of-Topic Questions and General Topic Questions:
        #     If a student asks a question not related to the quiz content or even asking about something general like a famous person information:
        #     "I'm here to focus on helping you understand your quiz results better. Let's stick to topics directly related to your quiz. Feel free to ask anything about the quiz itself!"

        #     5. Handling Math Questions Not Covered in the Quiz Results:
        #     If a question is about a math problem or equation not provided in the quiz results:
        #     "EXECUTE AUTOGEN"

        #     6. Encouraging Further Questions:
        #     Always encourage the student to keep asking questions to clarify their understanding:
        #     "Do you have any more questions about your quiz, or is there another part you'd like to review?"

        #     Remember to always put the explanation for the correct answer and personalize the response based on the specific context of the student's quiz and questions. Be clear, concise, and ensure the explanations are tailored to the student's level of understanding. Adjust the level of detail based on whether the student's answer was correct or incorrect, providing constructive feedback to facilitate learning.
            
        #     ====
        #     Student quiz result:
        #     {refprompt}

        #     ====
        #     Image dictionary:
        #     {json.dumps(image_library)}
        #     """
        
        qa_template = f"""
            You are well educated learning buddy called Jen. 
            Your purpose is to help student reviewing their student result based on the given student quiz result.
            
            To assisting the review question, YOU MUST FOLLOW the rules below:
            1. When student ask to review a question, you should explain step by step why their answer is wrong or why their answer is right.
            2. If the student ask out of topic, not about the question inside quiz result, politely ask them that they are not allowed to.
            3. if the student ask about math equation, return only "EXECUTE AUTOGEN".
            4. You should explain in language {baseLanguage}, but remain the original language of the question that being review.
            5. When user ask to review their answer on selected question, no need to repeat the question on your answer

            Remember to always put the explanation for the correct answer and personalize the response based on the specific context of the student's quiz and questions. 
            Remember to explain as a friendly tutor, you should be clear, concise, and ensure the explanations are tailored to the student's level of understanding. Adjust the level of detail based on whether the student's answer was correct or incorrect, providing constructive feedback to facilitate learning.
            
            ====
            Student quiz result:
            {refprompt}

            ====
            Image dictionary:
            {json.dumps(image_library)}
            """

        systemPrompt = []
        systemPrompt.append({
            "type":"text",
            "text": qa_template
        })
        
        prompts = ChatPromptTemplate.from_messages([
            SystemMessage(
                content=systemPrompt
            ),
            # HumanMessage(
            #     content=userPrompt
            # ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}. Please explain in {baseLanguage}"),
        ])
        
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)
        
        for item in chat_history[-4:]:
            memory.chat_memory.add_user_message(item.question)
            memory.chat_memory.add_ai_message(item.answer)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory
        )

        return chain

    
    def question_kb_chain_chat(chat_history,kb_ids):
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'],
                        model_name="gpt-4o", temperature=0, request_timeout=120,max_tokens=4050)
        # qa_template = """
        #     You are an AI question generator. 
        #     Please generate question that related to the reference and fullfil with user requirement.
        #     If you can't find any related information on the reference with the user requirement, please return with empty array [].
            
        #     - Stage 1: Look into the reference given and create the reference as your knowledge base. \
        #     - Stage 2: You are a tutor that already have some existing question and you want to create new question for your student's exam. \
        #     - Stage 3: Based on your knowledge and your experience, you should create a question that able to make your student understand the topic. \
        #     - Stage 4: Finally you will be able to create a question based on given knowledge and have same standart with your existing question.

        #     existing question:
        #     {example}

        #     reference:
        #     {context}

        #     {format_instructions}

        #     {chat_history}

        #     {question}
        #     """
        qa_template = """
            You are an AI question generator. 
            Your purpose is to generate question that related to the reference and fullfil with user requirement.
            
            To generate a question you should do it step by step:
            Step 1:  Look into the reference given and make it as your knowledge base.
            Step 2: Next, Decide if the knowledge base have an example question inside or it just a paper.
            - If it has an example question: make the example question as your guideline to generate a question.
            - if it just a paper:  you should generate question based on knowledge provided.
            - if it contain both: you should pull information from the paper and make the example as your guideline to generate a question.
            - if it doesn't meet any criteria: return "NOT ELIGIBLE"
            Step 3: Next, define the user needed.
            - if user asking to generate question: make sure you generate based on the user requirement.
            - If user's requirement are about converting question: don't need to generate new question, you only need to convert the question that exist on the reference.
            - if user's requirement are not about generating question: return "NOT ELIGIBLE"
            Step 4: finally you will generate question based on user's requirement and following the certain rules that provided and list out the question into a structured format as follow:
            - Question article: passage/any instruction of the question
            - Question:main content of question
            - Answer: the correct answer of the question
            - Difficulty level: level of question easy/normal/hard

            reference:
            {context}

            {question}
        """

        parser = JsonOutputParser(pydantic_object=QuestionList)

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=["context","question"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",output_key="answer",return_messages=True)

        # embeddings = GPT4AllEmbeddings()
        embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'],)
        retriever = KB.return_aws_opensearch_retriever(embeddings=embeddings, indexs=kb_ids)

        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True, 
            combine_docs_chain_kwargs = {'prompt': prompt},
            output_key="answer",
            verbose=True,
            # memory=memory,
        )

        return chain
    
    def rubric_parser():
        parser = JsonOutputParser(pydantic_object=Rubric)
        return parser
    
    def rubric_marking():
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['RUBRIC_MARKING'],
            model_name="gpt-4-0125-preview", temperature=0, request_timeout=120, response_format={ "type": "json_object" })
        # Additionally, analyze the provided answer for good points within the specified range. Do the checking specifically, don't give points that in general.

        qa_template = """
            You are an educator. Based on the following question and student answer, determine \
            the score for the student's answer based on the model answer. \
            While grading, make sure that the student's score is based from expectation that defined in model's anser.
            Next, explain your grading in a detail explanation for each paragraph of the student's answer. Make your explanation in a point format, for example: 1. explanation 1,2. explanation 2.. and attach based on which paragraph you are refering to.\
            In your explanation, adopt the role of an educator, elucidating the strengths and weaknesses in your student's response. Offer specific examples from the student's answer to ensure a clear understanding of their performance. \
            Do not use word that redundant in your explanation. make your explanation clear as much as possible.
            Your explanation must be in the same language as the student's answer. \
            
            The question statement is put inside the triple backticks below:
            ```{question}```

            Student's answer is put inside the triple backticks. You need to decide the score of this answer:
            ```{student_answer}```

            The model's answer is put inside the triple backticks below:
            ```{model_answer}```

            If the student's answer is in Chinese, your explanation paragraph must be in Chinese. \
            DO NOT mention the score in the explanation.

            The output should be formatted as a JSON instance that conforms to the JSON schema below.
                {{
                    mark: "the total mark of student answer",
                    fullmark: "full mark of the question",
                    feedback_for_tutor: "An explanation about student's answer grading and toned for a tutor",
                    feedback_for_student: "An explanation about student's answer grading and toned for a student",
                }}
            """

                    # translated_feedback_tutor: "Feedback for Tutor that explain in english language"
                    # translated_feedback_student: "Feedback for Student that explain in english language"
        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
        )

        prompts = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate(prompt = prompt),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                # HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory
        )

        return chain
    
    def auto_answer():
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['AF'], model_name="gpt-3.5-turbo-0125", temperature=0, request_timeout=120)
        # Additionally, analyze the provided answer for good points within the specified range. Do the checking specifically, don't give points that in general.
# First, if the question is not in english, translate both the question and answer option first to help you in answering the question. \
        qa_template = """
            You are a Chinese Teacher and you need to answer for the question provided. \
            
            First, you need to understand the content and context of the question. \
            Next, take a look on the answer option, and choose which option is the correct answer for the question. \
            Make sure you choose the correct answer from provided answer options.

            Only answer with the number of the correct answer from the answer option. 1 or 2 or 3 or 4.
            """

        prompt = PromptTemplate(
            template=qa_template,
            input_variables=[],
        )

        prompts = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate(prompt = prompt),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        memory = ConversationBufferMemory(memory_key="chat_history",input_key="question",return_messages=True)

        chain = LLMChain(
            llm=llm,
            prompt=prompts,
            verbose=True,
            memory=memory
        )

        return chain
    

