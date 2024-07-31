from yt_dlp import YoutubeDL
from langchain.embeddings.openai import OpenAIEmbeddings
from pathlib import Path
from typing import List
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile, APIRouter, Request, Query, Depends, HTTPException
from Modules import utils, reasonningKB, helper, prompt
from langchain.chains import LLMChain, RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from Modules.question_generator.chain import QuestionGeneratorChain
from D_ID import Clips

import config
router = APIRouter(
    prefix="/learning-module",
    tags=['Learning Module']
)
KB = reasonningKB.DocRetrievalKnowledgeBase(pdf_source_folder_path=config.TMPPATH)


@router.post("/genfromkb", summary="Get knowledge from KB then generate to video")
async def generate_from_kb(kb_id: str):
  embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'])
  retriever = KB.return_aws_opensearch_retriever(embeddings=embeddings, indexs=kb_id)
  llm=ChatOpenAI(openai_api_key=config.OPENAI_API_KEY_DICT['AF-KB'], temperature=config.temperature, model_name="gpt-3.5-turbo-16k")
  print(f"=> prepare qa_chain")
  qa_chain = ConversationalRetrievalChain.from_llm(
        llm = llm, 
        chain_type=config.chain_type,
        retriever=retriever,
        return_source_documents=True, 
        output_key='answer',
        verbose=False,
  )
  q = f"""
     - Please list the major topics of this document so it can be parsed as array on python; [1, 2, 3, ...]
    """
  result = qa_chain({"question": q, "chat_history": []})
  answer = result['answer']
  a_s = answer.splitlines()
  print(f"{a_s}")
  tr = []
  for a in a_s:
    q_2 = f"""
    - Please make a summary from this topic : {a}
    - start with : "In this lesson, you will learn about ..."
    """
    summary_res = qa_chain({"question": q_2, "chat_history": []})
    tr.append({'topic': a, 'summary': summary_res['answer'] })
  return tr