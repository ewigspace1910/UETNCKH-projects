from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile, APIRouter, Request, Query, Depends
# from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI,ChatOllama
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory,VectorStoreRetrieverMemory,CombinedMemory
from langchain.chains import LLMChain, RetrievalQA, ConversationalRetrievalChain
import azure.cognitiveservices.speech as speechsdk
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import ( HumanMessage, SystemMessage )
# from langchain.vectorstores import Chroma
# from chromadb.config import Settings
# import chromadb
from typing import List
import time
import random
import json
import os
import re
from Modules import utils, reasonningKB, helper, prompt
from Modules.question_generator.chain import QuestionGeneratorChain
from Modules.autogen import autogen
from Modules.kb_chain.chain import KBQnAChain
import config
import requests
import fitz
import os
import base64
import multiprocessing as mp
import concurrent.futures
from db_integration import call_db, QueryBank
from langchain.callbacks import get_openai_callback
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio

KB = reasonningKB.DocRetrievalKnowledgeBase(pdf_source_folder_path=config.TMPPATH)
router = APIRouter(
    prefix="/lamini",
    tags=['Adaptive Feedback (using Llama)']
)

class ChatItem(BaseModel):
    question: str 
    answer: str

class QnAreasonning(BaseModel):
    question: str
    kb_ids: List[str]
    chat_history: List[ChatItem]
    image_link:str=None
    req_type: str="eval|explain"

class QnAreasonningMath(BaseModel):
    question: str
    chat_history: List[ChatItem]

#class for KB building request
class KBfromURL(BaseModel):
    url: str
    kb_id: str = None
    document_name: str = "HeyHiKB",
    is_test:bool = False

class KBfromText(BaseModel):
    plaintext: str
    kb_id: str = None
    document_name: str = None

class KBfromQB(BaseModel):
    qids : List[int]
    kb_id: str = None
    qb_name: str = None

class KBfromPDF(BaseModel):
    file:UploadFile=File(...)
    kb_id: str = None
    document_name: str = None

##################FUNCTION########################
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def rname(): return time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(0, 99)) 

def extract_links(text):
    # The above Python code is defining a regular expression pattern to match URLs in a given text. It
    # then uses this pattern to find and return all URLs present in the text.
    # Regular expression pattern to match URLs
    url_pattern = re.compile(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})")

    # Find all URLs in the text
    links = re.findall(url_pattern, text)
    return links

@router.post("/qna", summary="[product-version]read more in the docs: <repo_link>/docs/AdaptiveFeedback.md")
async def qna_student_answer_v2(request:Request, reqs:QnAreasonning): 
    question, kb_ids, image_link, req_type, chat_history = reqs.question, reqs.kb_ids, reqs.image_link, reqs.req_type, reqs.chat_history
    if not image_link is None and image_link.strip() != "":
        #extract infomation in image and add to question.
        content = KB.describe_img(image_link)
        question += f"\n{content}"
        print("Content from image = ", content)
    
    conv_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question")
    for item in chat_history:
        if isinstance(item, ChatItem):
            if str(item.answer).find("!!!!") > -1: continue
            conv_memory.chat_memory.add_user_message(item.question)
            conv_memory.chat_memory.add_ai_message(item.answer)            
        else:
            if str(item['answer']).find("!!!!") > -1: continue
            conv_memory.chat_memory.add_user_message(item['question'])
            conv_memory.chat_memory.add_ai_message(item['answer'])

    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'])
    # retriever = KB.return_aws_opensearch_retriever(embeddings=embeddings, indexs=kb_ids)
    # docs = retriever.get_relevant_documents(question)
    callback = AsyncIteratorCallbackHandler()
    llm=ChatOllama(
        stream=True,
        verbose=True,
        callbacks=[callback],
        model="llama3"
    )

    #predict lang
    langpredictor_chain = QuestionGeneratorChain.predict_language()
    lang = await langpredictor_chain.arun({"question": question})
    max_history_round = 5
    with get_openai_callback() as cb:
        qanalysis_chain = KBQnAChain.analyze_question(chat_history[-max_history_round:])
        paraphased_questions = await  qanalysis_chain.arun({"question": question})
    list_docs = []
    has_error = False
    #get extra documents
    try: 
        if isinstance(paraphased_questions, str):
            paraphased_questions = paraphased_questions.replace("```json",'').replace("```",'').strip()
            paraphased_questions = json.loads(paraphased_questions)
        retriever = KB.return_aws_opensearch_retriever(embeddings=embeddings, indexs=kb_ids, k=2)
        for query in paraphased_questions['paraphrased_question']:
            docs = retriever.get_relevant_documents(query)
            list_docs += [doc.page_content for doc in docs]
        new_question =  f"{paraphased_questions['refined_question']}.  {question}?"
    except Exception as e: 
        print("[ERR] in api_KB_pipeline.qna-v2===> ", str(e))
        has_error = True
        new_question = question
    retriever = KB.return_aws_opensearch_retriever(embeddings=embeddings, indexs=kb_ids, k=1 if has_error else None)
    docs = retriever.get_relevant_documents(new_question)
    list_docs += [doc.page_content for doc in docs]
    docs = "\n\n".join(list(set(list_docs)))
    # lamini code in here

    if question.lower().find("student response") > -1: 
        prompt_template = prompt.prompt.generate_student_answer_evaluation_prompt(True)
    else:
        prompt_template = prompt.prompt.generate_qna_prompt_v2(lang = lang)

    prompts = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            prompt_template,
        ),
        # MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])
    chain = LLMChain(
        llm=llm,
        prompt=prompts,
        verbose=True,
        # memory=conv_memory
    )

    task = asyncio.create_task(
        chain.arun({"question":question,"context":docs})
    )

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task
   
    # result = await chain.arun({"question":question,"context":docs})
    # answer = result
    # ###################################
    # #prosprocessing
    # for w in ["from the provided documents", "from the given documents", "from the provided document", "from the given document", "from the document", "from the documents",
    #           "in the provided documents", "in the given documents", "in the provided document", "in the given document", "in the document", "in the documents"]: answer = answer.replace(w, "from the source information")
    # for w in ["based on the provided documents", "based on the given documents", "based on the provided document", "based on the given document", "based on the document", "based on the documents"]: answer = answer.replace(w, "based on the source information")
    # all_links = extract_links(answer)
    # for link in all_links:
    #     if not (link[-1].isalpha() or  link[-1].isdigit()): link = link[:-1]
    #     answer = answer.replace(link, f""" <a href="{link}">{link}</a> """)
    
    # docsource = {}

    # return JSONResponse({
    #     "complete": True,
    #     "data"  : {"response": answer + f"\n\n===> Lang= {lang}" if question.find("@180866") > -1 else answer,  
    #                "refs"      : docsource,
    #             }
    # })   


async def send_message(content: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=[callback],
        openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB']
    )

    task = asyncio.create_task(
        model.agenerate(messages=[[HumanMessage(content=content)]])
    )

    try:
        async for token in callback.aiter():
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task

@router.post("/qna-stream", summary="[product-version]read more in the docs: <repo_link>/docs/AdaptiveFeedback.md")
async def stream_chat(request:Request, reqs:QnAreasonning):
    generator = qna_student_answer_v2(request,reqs)
    return StreamingResponse(generator, media_type="text/event-stream")

