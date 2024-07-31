from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile, APIRouter, Request, Query, Depends
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.docstore.document import Document
from langchain.memory import ConversationBufferMemory, VectorStoreRetrieverMemory, CombinedMemory
from langchain.chains import LLMChain, RetrievalQA, ConversationalRetrievalChain
import azure.cognitiveservices.speech as speechsdk
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import ( HumanMessage, SystemMessage )
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
import boto3
from Modules.question_generator.agents import create_gen,create_thread_and_run_kb_chat, assistant_stream
from Modules.openai_assistant.event_handler import EventHandler
from Modules.openai_assistant.assistant_service import AssistantService

KB = reasonningKB.DocRetrievalKnowledgeBase(pdf_source_folder_path=config.TMPPATH)
router = APIRouter(
    prefix="/reasoning",
    tags=['Adaptive Feedback (QnA reasoning)']
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

################## API Endpoint #################
@router.post("/convert2kb/pdf", summary="Extract knowledge from PDF file to KB")
async def convert_pdf2KB(docfile:UploadFile=File(...), textbook_name="HeyJenKB", kb_id:str=None):
# async def convert_pdf2KB(reqs:KBfromPDF):
#   docfile, textbook_name, kb_id = reqs.docfile, reqs.textbook_name, reqs.kb_id
    starttime = time.time()
    # try:
    contents = await docfile.read()
    root, randname = os.path.join(config.TMPPATH) , time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(0, 999))
    filename = randname #if filename is None else filename   
    path = os.path.join(root, filename)
    imgspath = os.path.join(root, "imgs", filename)
    if not os.path.exists(path): os.makedirs(path)
    if not os.path.exists(imgspath): os.makedirs(imgspath)
    pdf_path = os.path.join(root, f"{filename}.pdf")
    txt_path = os.path.join(path, f"{filename}.txt")
    with open(pdf_path, "wb") as f: f.write(contents)
    
    #convert pdf2image for chatgpt4
    zoom_x = 1.5  # horizontal zoom
    zoom_y = 1.5  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
    doc = fitz.open(pdf_path)  # open document
    contents = []
    def __ocr_img(page_queue, mat, imgspath, conn):
        KB = reasonningKB.DocRetrievalKnowledgeBase(pdf_source_folder_path=config.TMPPATH)
        base64img_queue = []
        for page in page_queue:
            pix = page.get_pixmap(matrix=mat)  # render page to an image
            impath = os.path.join(imgspath, f"{page.number}.png")
            pix.save(impath)
            base64_image = encode_image(impath)
            base64img_queue += [base64_image]
        try: 
            content =  KB.load_img(base64img_queue)
            # content = "hsshjhạkhsshdkshkjdshkjád"
        except: content = ""
        conn.send([page_queue[0].number, content])
        conn.close()

    parent_connections = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        img_queue = []
        page_len = len(doc)
        for idx, page in enumerate(doc):
            img_queue += [page]
            if len(img_queue) == config.BATCH_SIZE or idx >= page_len-1:
                parent_conn, child_conn = mp.Pipe()
                parent_connections += [parent_conn]
                executor.submit(__ocr_img, img_queue, mat, imgspath, child_conn)
                img_queue = []
            if (idx+1) % config.SLEEP_PER_PAGE ==0:  
                st = time.time()
                while True:
                    if config.SLEEP_INTERVAL < (time.time() - st): break
                # time.sleep(15)
    elements = []
    for parent_connection in parent_connections:
        elements += [parent_connection.recv()]
    contents = sorted(elements, key=lambda x: x[0])
    with open(txt_path, "w", encoding='utf-8') as f: 
        for c in contents : f.write(c[1] + "\n")
        f.close()

    # try:documents = KB.load_pdfs(txt_path) 
    # except:documents = KB.load_txt(txt_path) 
    try:documents = KB.load_pdfs(path) 
    except:documents = KB.load_pdfs(path) 
    for doc in documents:  doc.metadata['source'] = f"text-book: {textbook_name}"  
    
    chunked_documents = KB.split_documents(loaded_docs=documents)
    # embeddings = GPT4AllEmbeddings()
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'])
    filename = filename if kb_id is None else kb_id.lower()
    KB.load_KB_into_Opensearch(embeddings, docs=chunked_documents, index_name=filename)

    print(f"==>[INFO] Converted pdf file {filename} to KB in {time.time()-starttime}s")
    return JSONResponse({
        "complete": True,
        "data"  : {"kb-id": filename},
        "msg"   : f"File '{filename}' is already updated into KB "
    })   


@router.post("/convert2kb/pdf-url", summary="Extract knowledge from url of PDF file to KB")
async def convert_pdf_url2KB(reqs:KBfromURL):
    url, textbook_name, kb_id = reqs.url, reqs.document_name, reqs.kb_id
    starttime = time.time()
    # readfile from url
    contents = requests.get(url).content
    
    root, randname = os.path.join(config.TMPPATH) , time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(0, 999))
    filename = randname #if filename is None else filename   
    path = os.path.join(root, filename)
    imgspath = os.path.join(root, "imgs", filename)
    if not os.path.exists(path): os.makedirs(path)
    if not os.path.exists(imgspath): os.makedirs(imgspath)
    pdf_path = os.path.join(root, f"{filename}.pdf")
    txt_path = os.path.join(path, f"{filename}.txt")
    with open(pdf_path, "wb") as f: f.write(contents)
    if reqs.is_test: return "saved file into /tmp folder"
    zoom_x = 1.5  # horizontal zoom
    zoom_y = 1.5  # vertical zoom
    mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
    doc = fitz.open(pdf_path)  # open document
    contents = []
    def __ocr_img(page_queue, mat, imgspath, conn):
        KB = reasonningKB.DocRetrievalKnowledgeBase(pdf_source_folder_path=config.TMPPATH)
        base64img_queue = []
        for page in page_queue:
            pix = page.get_pixmap(matrix=mat)  # render page to an image
            impath = os.path.join(imgspath, f"{page.number}.png")
            pix.save(impath)
            base64_image = encode_image(impath)
            base64img_queue += [base64_image]
        try: 
            content =  KB.load_img(base64img_queue)
        except: content = ""
        conn.send([page_queue[0].number, content])
        conn.close()

    parent_connections = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        img_queue = []
        page_len = len(doc)
        for idx, page in enumerate(doc):
            img_queue += [page]
            if len(img_queue) == config.BATCH_SIZE or idx >= page_len-1:
                parent_conn, child_conn = mp.Pipe()
                parent_connections += [parent_conn]
                executor.submit(__ocr_img, img_queue, mat, imgspath, child_conn)
                img_queue = []
            if (idx+1) % config.SLEEP_PER_PAGE ==0:  
                st = time.time()
                while True:
                    if config.SLEEP_INTERVAL < (time.time() - st): break
                # time.sleep(15)
    elements = []
    for parent_connection in parent_connections:
        elements += [parent_connection.recv()]
    contents = sorted(elements, key=lambda x: x[0])
    with open(txt_path, "w", encoding='utf-8') as f: 
        for c in contents : f.write(c[1] + "\n")
        f.close()

    try:documents = KB.load_pdfs(path) 
    except:documents = KB.load_pdfs(path) 
    for doc in documents:  doc.metadata['source'] = f"text-book: {textbook_name}"  
    
    chunked_documents = KB.split_documents(loaded_docs=documents)
    print(chunked_documents)
    # embeddings = GPT4AllEmbeddings()
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'])
    filename = filename if kb_id is None else kb_id.lower()
    KB.load_KB_into_Opensearch(embeddings, docs=chunked_documents, index_name=filename)

    print(f"==>[INFO] Converted pdf file {filename} to KB in {time.time()-starttime}s")
    return JSONResponse({
        "complete": True,
        "data"  : {"kb-id": filename},
        "msg"   : f"File '{filename}' is already updated into KB "
    })   


@router.post("/convert2kb/plantext", summary="Extract knowledge from plan text")
async def convert_text2KB(reqs:KBfromText):
    docfile, textbook_name, kb_id = reqs.plaintext, reqs.document_name, reqs.kb_id
    starttime = time.time()
    # try:
    # contents = await docfile.read()
    root, randname = os.path.join(config.TMPPATH) , "t-" + time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(0, 999))
    filename = randname #if filename is None else filename   
    path = os.path.join(root, filename)
    if not os.path.exists(path): os.makedirs(path)
    txt_path = os.path.join(path, f"{filename}.txt")
    with open(txt_path, "w", encoding='utf-8') as f: 
        f.write(docfile)
        f.close()

    # try:documents = KB.load_pdfs(txt_path) 
    # except:documents = KB.load_txt(txt_path) 
    try:documents = KB.load_pdfs(path) 
    except:documents = KB.load_pdfs(path) 
    for doc in documents:  doc.metadata['source'] = f"text-book: {textbook_name}"  
    
    chunked_documents = KB.split_documents(loaded_docs=documents)
    # embeddings = GPT4AllEmbeddings()
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'])
    filename = filename if kb_id is None else kb_id
    KB.load_KB_into_Opensearch(embeddings, docs=chunked_documents, index_name=filename)

    print(f"==>[INFO] Converted pdf file {filename} to KB in {time.time()-starttime}s")
    return JSONResponse({
        "complete": True,
        "data"  : {"kb-id": filename},
        "msg"   : f"File '{filename}' is already updated into KB "
    })   

@router.post("/convert2kb/QB", summary="Extract knowledge from question bank")
async def convert_QB2KB(reqs:KBfromQB):
    question_ids, qb_name, kb_id = reqs.qids, reqs.qb_name, reqs.kb_id
    starttime = time.time()
    # try:
    # contents = await docfile.read()
    if kb_id is None:
        randname =  "qb-" + time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(0, 999))
        filename = randname #if filename is None else filename   
    else:
        filename = kb_id
        def __del_question_id_in_kb(kb_id, qid):
            url = f"{config.OPENSEARCH_URL}/{kb_id}/_delete_by_query"
            headers = {"Content-Type": "application/json"}
            data = {"query": {
                "match": {"metadata.question_id": qid}
                }
            }
            requests.post(url, auth=requests.auth.HTTPBasicAuth(config.OPENSEARCH_ACCOUNT.split(":")[0], config.OPENSEARCH_ACCOUNT.split(":")[-1]), headers=headers, json=data)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _, qid in enumerate(question_ids):
                executor.submit(__del_question_id_in_kb, kb_id, qid)

    query2db = QueryBank.get_content_question(question_ids=question_ids)
    question_contents = call_db("smartjen", query2db)
    list_of_docs = []
    for idx, row in question_contents.iterrows():
        doc =  Document(page_content=row.text, metadata={"source": qb_name, "_id":row.question_id, 'question_id':row.question_id, 'reference_id':row.reference_id})
        list_of_docs += [doc]
    # embeddings = GPT4AllEmbeddings()
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'])
    filename = filename if kb_id is None else kb_id
    KB.load_KB_into_Opensearch(embeddings, docs=list_of_docs, index_name=filename)

    print(f"==>[INFO] Converted question_list {question_ids[:10]} to KB in {time.time()-starttime}s")
    return JSONResponse({
        "complete": True,
        "data"  : {"kb-id": filename},
        "msg"   : f"QB-{filename} is already updated into KB "
    })   



###################################################
#                   REASONING
####################################################

@router.post("/qna", summary="[product-version]read more in the docs: <repo_link>/docs/AdaptiveFeedback.md")
async def qna_student_answer(request:Request, reqs:QnAreasonning): 
    question, kb_ids, image_link, req_type, chat_history = reqs.question, reqs.kb_ids, reqs.image_link, reqs.req_type, reqs.chat_history
    if not image_link is None and image_link.strip() != "":
        #extract infomation in image and add to question.
        content = KB.describe_img(image_link)
        question += f"\n{content}"
        print("Content from image = ", content)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer', input_key="question")
    for item in chat_history:
        if isinstance(item, ChatItem):
            if str(item.answer).find("!!!!") > -1: continue
            memory.chat_memory.add_user_message(item.question)
            memory.chat_memory.add_ai_message(item.answer)            
        else:
            if str(item['answer']).find("!!!!") > -1: continue
            memory.chat_memory.add_user_message(item['question'])
            memory.chat_memory.add_ai_message(item['answer'])

    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'])
    retriever = KB.return_aws_opensearch_retriever(embeddings=embeddings, indexs=kb_ids)
    # docs = retriever.get_relevant_documents(question)
    llm=ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'], temperature=config.temperature, model_name="gpt-3.5-turbo-16k")

    #predict lang
    langpredictor_chain = QuestionGeneratorChain.predict_language()
    lang = await langpredictor_chain.arun({"question": question})
    # print('kblang==>', lang)

    if question.lower().find("student response") > -1: 
        prompt_template = prompt.prompt.generate_student_answer_evaluation_prompt(True)
    else:
        prompt_template = prompt.prompt.generate_qna_prompt(lang = lang, chat_history = memory)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm = llm, 
        chain_type=config.chain_type,
        retriever=retriever,
        return_source_documents=True, 
        memory= memory,
        output_key='answer',
        combine_docs_chain_kwargs = {'prompt': prompt_template},
        verbose=False,
    )
    question = f"""
     {question}.

     NOTE THAT:  
     - Do not answer if my question is not related to the context information in the reference documents.
     - Do not use your personal knowledge to answer my question.

    """
    result = qa_chain({"question": question})
    answer = result["answer"]
    ###################################
    #prosprocessing
    for w in ["from the provided documents", "from the given documents", "from the provided document", "from the given document", "from the document", "from the documents",
              "in the provided documents", "in the given documents", "in the provided document", "in the given document", "in the document", "in the documents"]: answer = answer.replace(w, "from the source information")
    for w in ["based on the provided documents", "based on the given documents", "based on the provided document", "based on the given document", "based on the document", "based on the documents"]: answer = answer.replace(w, "based on the source information")
    all_links = extract_links(answer)
    for link in all_links:
        if not (link[-1].isalpha() or  link[-1].isdigit()): link = link[:-1]
        answer = answer.replace(link, f""" <a href="{link}">{link}</a> """)
    
    docsource = {}
    if True:
        used_doc = []
        for i, doc in enumerate(result['source_documents']):
            if not doc.metadata['source'] in used_doc:
                docsource[f"Doc-{i+1}"] = doc.metadata['source'] 
                used_doc += [doc.metadata['source']]
    else: docsource = {}
    return JSONResponse({
        "complete": True,
        "data"  : {"response": answer + f"\n\n===> Lang= {lang}" if question.find("@180866") > -1 else answer,  
                   "refs"      : docsource,
                }
    })   

@router.post("/qna-v2", summary="[product-version]read more in the docs: <repo_link>/docs/AdaptiveFeedback.md")
async def qna_student_answer_v2(request:Request, reqs:QnAreasonning): 
    question, kb_ids, image_link, req_type, chat_history = reqs.question, reqs.kb_ids, reqs.image_link, reqs.req_type, reqs.chat_history
    if not image_link is None and image_link.strip() != "":
        #extract infomation in image and add to question.
        content = KB.describe_img(image_link)
        question += f"\n{content}"
        print("Content from image = ", content)
    
    gatewayapi = boto3.client('apigatewaymanagementapi', endpoint_url="https://s1o2zdgy57.execute-api.ap-southeast-1.amazonaws.com/production")
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
    llm=ChatOpenAI(
        streaming=True,
        verbose=False,
        callbacks=[callback],
        openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'], 
        temperature=config.temperature, 
        model_name="gpt-3.5-turbo-16k"
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
        verbose=False,
        # memory=conv_memory
    )

    task = asyncio.create_task(
        chain.arun({"question":new_question,"context":docs})
    )
    
    request_body = await request.json()
    connection_id = request_body.get('connection_id', '')
    # Send connection ID back to the client
    counting = 0
    try:
        async for token in callback.aiter():
            gatewayapi.post_to_connection(
                ConnectionId=connection_id,
                Data=json.dumps({'message': token,'type': 'message',"counting":counting})
            )
            counting = counting + 1
            yield token
    except Exception as e:
        print(f"Caught exception: {e}")
    finally:
        callback.done.set()

    await task
   

    # # print(result)
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


@router.post("/qna-v3", summary="[product-version]read more in the docs: <repo_link>/docs/AdaptiveFeedback.md")
async def qna_student_answer_v3(request:Request, reqs:QnAreasonning): 
    """
    In ver3, we will call gpt 2 times:
    - the first one is to summarize chat history and predict the language used in the lasted questions
    - the second one is to answer the question

    """
    question, kb_ids, image_link, req_type, chat_history = reqs.question, reqs.kb_ids, reqs.image_link, reqs.req_type, reqs.chat_history
    max_history_round = 5
    total_token=total_cost = 0
    if not image_link is None and image_link.strip() != "":
        #extract infomation in image and add to question.
        content = KB.describe_img(image_link)
        question += f"\n{content}"
        print("Content from image = ", content)
    
    conv_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, input_key="question")
    for item in chat_history[-max_history_round:]:
        if isinstance(item, ChatItem):
            if str(item.answer).find("!!!!") > -1: continue
            conv_memory.chat_memory.add_user_message(item.question)
            conv_memory.chat_memory.add_ai_message(item.answer)            
        else:
            if str(item['answer']).find("!!!!") > -1: continue
            conv_memory.chat_memory.add_user_message(item['question'])
            conv_memory.chat_memory.add_ai_message(item['answer'])

    #predict lang
    langpredictor_chain = QuestionGeneratorChain.predict_language()
    lang = await langpredictor_chain.arun({"question": question})
    # print('kblang==>', lang)
    #summary topic of current question.
    with get_openai_callback() as cb:
        qanalysis_chain = KBQnAChain.analyze_question(chat_history[-max_history_round:])
        paraphased_questions = await  qanalysis_chain.arun({"question": question})
        total_token += cb.total_tokens 
        total_cost  += cb.total_cost
    embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'])
    
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
        print("[ERR] in api_KB_pipeline.qna-v3===> ", str(e))
        has_error = True
        new_question = question
    retriever = KB.return_aws_opensearch_retriever(embeddings=embeddings, indexs=kb_ids, k=1 if has_error else None)
    docs = retriever.get_relevant_documents(new_question)
    list_docs += [doc.page_content for doc in docs]
    list_docs = list(set(list_docs))

    response = await autogen.find_answer_qna(new_question, "\n\n".join(list_docs), language=lang)
    answer = response['answer']
    ###################################
    #prosprocessing
    all_links = extract_links(answer)
    for link in all_links:
        if not (link[-1].isalpha() or  link[-1].isdigit()): link = link[:-1]
        answer = answer.replace(link, f""" <a href="{link}">{link}</a> """)
    
    docsource = {}
    # if True:
    #     used_doc = []
    #     for i, doc in enumerate(result['source_documents']):
    #         if not doc.metadata['source'] in used_doc:
    #             docsource[f"Doc-{i+1}"] = doc.metadata['source'] 
    #             used_doc += [doc.metadata['source']]
    # else: docsource = {}
    extra_infor = response['cost']
    extra_infor['cash'] = extra_infor['cash']      + total_cost
    extra_infor['tokens'] = extra_infor['tokens']    + total_token
    return JSONResponse({
        "complete": True,
        "data"  : {"response": answer + f"\n\n===> Lang= {lang}, cost=${extra_infor['cash']}, {extra_infor['tokens']}t" if question.find("@180866") > -1 else answer,  
                   "refs"       : docsource,
                    'cost'      : extra_infor
                }
    })   

class Message(BaseModel):
    content: str


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

## MathJen
async def return_summary_math(chat_history, question, lang, conv_memory):
    for item in chat_history:
        if isinstance(item, ChatItem):
            if str(item.answer).find("!!!!") > -1: continue
            conv_memory.chat_memory.add_user_message(item.question)
            conv_memory.chat_memory.add_ai_message(item.answer)            
        else:
            if str(item['answer']).find("!!!!") > -1: continue
            conv_memory.chat_memory.add_user_message(item['question'])
            conv_memory.chat_memory.add_ai_message(item['answer'])

    llm = ChatOpenAI(
        openai_api_key = config.OPENAI_API_KEY_DICT['AF-KB'], 
        temperature = config.temperature, 
        model_name = "gpt-3.5-turbo-0125"
    )

    prompt_template = prompt.prompt.generate_question_history_prompt(lang = "English")

    prompts = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            prompt_template,
        ),
        MessagesPlaceholder(variable_name = "chat_history"),
        HumanMessagePromptTemplate.from_template(
            "Here is the chat history: {chat_history} and the current input is: {question}"
        )
    ])
    
    chain = LLMChain(
        llm = llm,
        prompt = prompts,
        verbose = False,
        memory = conv_memory
    )

    return chain

async def get_query(problem):
    # Query postfix to guide answer based on model tutorial
    query_postfix = '\nPlease reason step by step and concise, and put your final answer within \\boxed{}.'
    # Create query
    return { 'role': 'user', 'content': problem + query_postfix }


@router.post("/qna-math", summary = "[product-version]read more in the docs: <repo_link>/docs/AdaptiveFeedback.md")
async def qna_math_student_answer(request:Request, reqs:QnAreasonningMath):
    question, chat_history = reqs.question, reqs.chat_history

    # Predicting the Language
    typepredictor_chain = QuestionGeneratorChain.predict_math_question()
    question_type = await typepredictor_chain.arun({"question": question})

    print(question_type)

    if question_type == "math" or question_type == "'math'":
        if chat_history == []:
            pass
        else:
            conv_memory = ConversationBufferMemory(
                memory_key = "chat_history", 
                return_messages = True, 
                input_key = "question"
            )
            chain = await return_summary_math(chat_history, question, "English", conv_memory)
            result = await chain.arun(
                {
                    "chat_history": chat_history,
                    "question": question
                }
            )
            #print(f"Result is: {result}")
            question = result
        
        ## Code for MathChat Autogen
        #############################
        solution_autogen = await autogen.find_solution_mathchat(
            question,
            language = "English",
            return_json = True
        )

        correct_answer = solution_autogen['correct_answer']
        solution = solution_autogen['solution']

        ## Code for DeepSeekMath
        #############################
        # model_name = "./deepseek-math-7b-instruct"
        # tokenizer = AutoTokenizer.from_pretrained(model_name)

        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.bfloat16, # use original bfloat16 data type!
        #     device_map='auto',
        # )
        # # Load generation configuration
        # model.generation_config = GenerationConfig.from_pretrained(model_name)
        # # Set padding token
        # model.generation_config.pad_token_id = model.generation_config.eos_token_id

        # problem = "The average mass of 6 pupils was 40 kg. The average mass of the first 4 pupils was 12 kg more than the average mass of the remaining 2 pupils, Lisa and Bobby.Find the average mass of Lisa and Bobby."
        # math_problem = await get_query(problem)

        # input_tensor = tokenizer.apply_chat_template(
        #     [math_problem],
        #     add_generation_prompt=True,
        #     return_tensors='pt',
        # ).to(model.device)

        # generated_ids = model.generate(
        #     input_tensor,
        #     max_new_tokens=1000,
        #     do_sample=False,
        # )

        # solution = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        # correct_answer = re.findall(r'\\boxed\{(\d+)}', solution)

        ## Code for Rephrasing Solution
        #############################
        llm = ChatOpenAI(
            openai_api_key = "", 
            temperature = config.temperature, 
            model_name = ""
        )

        prompt_template = prompt.prompt.generate_qna_math_rephrase_prompt(lang = "English")

        prompts = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                prompt_template,
            ),
            #MessagesPlaceholder(variable_name = "chat_history"),
            HumanMessagePromptTemplate.from_template(
                "Here is the solution that you need to rephrase into the Singaporean format: {solution}"
            )
        ])
        
        chain = LLMChain(
            llm = llm,
            prompt = prompts,
            verbose = False,
        )
    
        result = await chain.arun(
            {
                "solution": solution,
            }
        )

        answer = result

    elif question_type == "general" or question_type == "'general'":
        conv_memory = ConversationBufferMemory(
            memory_key = "chat_history", 
            return_messages = True, 
            input_key = "question"
        )

        for item in chat_history:
            if isinstance(item, ChatItem):
                if str(item.answer).find("!!!!") > -1: continue
                conv_memory.chat_memory.add_user_message(item.question)
                conv_memory.chat_memory.add_ai_message(item.answer)            
            else:
                if str(item['answer']).find("!!!!") > -1: continue
                conv_memory.chat_memory.add_user_message(item['question'])
                conv_memory.chat_memory.add_ai_message(item['answer'])

        llm = ChatOpenAI(
            openai_api_key = config.OPENAI_API_KEY_DICT['AF-KB'], 
            temperature = config.temperature, 
            model_name = "gpt-3.5-turbo-0125"
        )

        prompt_template = prompt.prompt.generate_qna_general_prompt(lang = "English")

        prompts = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                prompt_template,
            ),
            MessagesPlaceholder(variable_name = "chat_history"),
            HumanMessagePromptTemplate.from_template(
                "Here is the question: {question}"
            )
        ])
        
        chain = LLMChain(
            llm = llm,
            prompt = prompts,
            verbose = False,
            memory = conv_memory
        )
    
        result = await chain.arun(
            {
                "question": question,
            }
        )

        answer = result

    return JSONResponse(
        {
            "complete": True,
            "data": {
                "solution": answer,
                "answer": correct_answer if question_type == "math" or question_type == "'math'" else " ",
            }
        }
    )

@router.post("/qna-assistant", summary="[product-version]read more in the docs: <repo_link>/docs/AdaptiveFeedback.md")
async def qna_student_answer_v3(request:Request): 
    request_body = await request.json()
    vector_id = request_body.get('vector_id')
    question = request_body.get('question')
    connection_id = request_body.get('connection_id')
    thread_id = request_body.get('thread_id')

    # thread, assistant_id = await create_thread_and_run_kb_chat(
    #     question, vector_id
    # )

    callback = EventHandler()
    assistant_service = AssistantService()
    # callback = AsyncIteratorCallbackHandler()
    gatewayapi = boto3.client('', endpoint_url="")
    await assistant_service.create_message(thread_id,question)
    gen = assistant_service.create_gen(thread_id,connection_id, callback, gatewayapi)

    return StreamingResponse(gen, media_type="text/event-stream")

