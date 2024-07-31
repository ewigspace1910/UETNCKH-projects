from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, confloat
from Modules.knowledgebase import knowledgebase
from Modules.chain import Chain
from Modules.helper import Helper
from Modules import utils
from Modules.rubric_marking import rubric_mark
from Modules.compo_marking import compo_marking, compo_marking_v2
from typing import List
import json
import os
import csv
from dotenv import load_dotenv
import uvicorn 
from fastapi import HTTPException
from langchain.callbacks import get_openai_callback
import azure.cognitiveservices.speech as speechsdk
import uuid, re
from fastapi.responses import FileResponse
from pylatexenc.latex2text import LatexNodes2Text
from mangum import Mangum
import time, random
from Routers import api_lamini, api_KB_pipeline, api_gen_ai, api_adaptive_feedback, yt_kb_pipeline, api_learning_module
import elevenlabs
from elevenlabs import Voice, VoiceSettings, generate
from pydub import AudioSegment
import inflect

import requests
import io
from io import BytesIO

import warnings, xlsxwriter
warnings.filterwarnings("ignore")
from Modules.autogen import autogen
import config
load_dotenv()


import warnings
warnings.filterwarnings("ignore")
def rname(): return time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(0, 99)) 

def read_csv_to_array(filename):
    data_array = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile,delimiter=";")
        for row in csvreader:
            data_array.append(row)
    return data_array

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class KBCreateRequest(BaseModel):
    pdf_url: str

class KBGenerateRequest(BaseModel):
    kb_id: str
    prompt: str
class KBSummaryRequest(BaseModel):
    kb_id: str

class KBCheckingRequest(BaseModel):
    kb_id: str
    
class ChatItem(BaseModel):
    question: str 
    answer: str

class KBChatRequest(BaseModel):
    kb_id: str
    question: str
    chat_history: List[ChatItem]
    requirement: str

class GenerateQuestionRequest(BaseModel):
    prompt: str
    reference: str

class GenerateContentRequest(BaseModel):
    prompt: str
    reference: str
    student_level: str
    student_name: str
    learning_step: str
    substrand_name: str
    topic_name: str


class QuestionCheckingRequest(BaseModel):
    question: str
    answer: str

class ContentCheckingRequest(BaseModel):
    content: str

class KBChatAlsRequest(BaseModel):
    reference: str
    learning_reference: str
    case_study: str
    question: str
    chat_history: List[ChatItem]

class KBChatAlsResultRequest(BaseModel):
    reference: str
    question: str
    chat_history: List[ChatItem]

class RubricRequest(BaseModel):
    question: str
    student_answer: str
    model_answer: str

class KBChatAlsPracticeRequest(BaseModel):
    reference: str
    question: str
    result_score: str
    chat_history: List[ChatItem]

class AutoSolutionRequest(BaseModel):
    question: str
    reference: str
    answer: str
    student_name: str
    chat_history: List[ChatItem]

class RubricMarkingRequest(BaseModel):
    high_level_criteria: List[dict]
    question: str
    correct: str
    student: str
    student_class: str

class CompoMarkingRequest(BaseModel):
    question_statement: str
    rubric_table: List[dict]
    student_composition: str
    model_composition: str
    student_class: str

class VoiceObject(BaseModel):
    code: str 
    name: str
class KBChatQuizResultRequest(BaseModel):
    reference: str
    question: str
    language: str
    voice: VoiceObject
    chat_history: List[ChatItem]
    autogenver: float = 4.0
    image_description: List[dict]

class VoiceObject(BaseModel):
    code: str 
    name: str

class TTSVoiceRequest(BaseModel):
    text: str
    voice_id: str
    stability: confloat(ge=0.0, le=1.0)
    similarity_boost: confloat(ge=0.0, le=1.0)
    style: confloat(ge=0.0, le=1.0)

class TTSVoiceMerge(BaseModel):
    audio_path: List[dict]

@app.get( path="/audio/{file_name}")
async def post_media_file(file_name):
    return FileResponse(os.path.join(config.TMPPATH, f"{file_name}"), media_type="audio/mpeg")


@app.get("/")
def read_root():
    return {"msg": "404 NOT FOUND!"}

@app.post("/knowledge-base/create")
async def create_kb(req:KBCreateRequest):
    # parse pdf url to text and chunks
    pdf_content = await knowledgebase.parse_pdf(req.pdf_url)
     # Check if the PDF is empty
    if not pdf_content.strip():
        raise HTTPException(status_code=400, detail="The PDF is empty.")
    # Check if the PDF contains only images (no text)
    contains_text = any(char.isalnum() for char in pdf_content)
    if not contains_text:
        raise HTTPException(status_code=400, detail="The PDF only contains images.")
    return {'knowledge_base_id':pdf_content}

@app.post("/knowledge-base/chatbot")
async def chatbot(req:KBChatRequest):
    # parse pdf url to text and chunks
    docs = await knowledgebase.get_pdf(req.kb_id)
    chain = Chain.chabot_chain(docs)
    chat_history = Helper.normalized_history(req.chat_history)
    response = chain({"question": req.question, "chat_history": chat_history, "requirement": req.requirement})
    return response

@app.post("/knowledge-base/generate-question")
async def generate_question(req:KBGenerateRequest):
    # parse pdf url to text and chunks
    docs = await knowledgebase.get_pdf(req.kb_id)
    chain = Chain.question_chain()
    response = chain.run(input_documents=docs, question=req.prompt)
    jsonResponse = json.loads(response)
    return jsonResponse

@app.post("/knowledge-base/summarize")
async def summarize(req:KBSummaryRequest):
    # parse pdf url to text and chunks
    docs = await knowledgebase.get_pdf(req.kb_id)
    chain = Chain.summary_chain()
    response = chain.run(docs)
    jsonResponse = json.loads(response)
    return jsonResponse

@app.post("/knowledge-base/check-grammar")
async def check_grammar(req:KBCheckingRequest):
    # parse pdf url to text and chunks
    docs = await knowledgebase.get_pdf(req.kb_id)
    chain = Chain.grammar_chain()
    reponse = chain.run(docs)
    jsonResponse = json.loads(reponse) 
    return jsonResponse

@app.post("/knowledge-base/check-readability")
async def check_readability(req:KBCheckingRequest):
    # parse pdf url to text and chunks
    docs = await knowledgebase.get_pdf(req.kb_id)
    chain = Chain.readability_chain()
    reponse = chain.run(docs)
    jsonResponse = json.loads(reponse) 
    return jsonResponse

@app.post("/knowledge-base/suggestion-comment")
async def suggestion_comment(req:KBCheckingRequest):
    docs = await knowledgebase.get_pdf(req.kb_id)
    chain = Chain.suggestion_comment_chain()
    reponse = chain.run(docs)
    jsonResponse = json.loads(reponse)
    return jsonResponse


@app.post("/global/generate-question")
async def global_generate_question(req:GenerateQuestionRequest):
    chain = Chain.question_v2_chain()
    with get_openai_callback() as cb:
        response = chain.run(input_documents='', reference=req.reference, question=req.prompt)
        #print(cb)

    jsonResponse = json.loads(response)
    return jsonResponse

@app.post("/global/generate-question-v2")
async def global_generate_question_v2(req:KBChatAlsResultRequest):
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = Chain.question_v2_chain_chat(req.chat_history)
    response = await chain.arun({"question": req.question,"chat_history":chat_history,"reference":req.reference})
    print(response)
    jsonResponse = json.loads(response) 
    return jsonResponse

@app.post("/global/generate-question-v3")
async def global_generate_question_v3(req:KBChatAlsResultRequest):
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = Chain.question_v3_chain_chat(req.chat_history,req.reference)
    parser = Chain.question_parser_new()
    total_token = 0
    total_cost = 0
    try:
        with get_openai_callback() as cb:
            response = await chain.arun(question=req.question,chat_history=chat_history,reference=req.reference)
            total_token = cb.total_tokens 
            total_cost = cb.total_cost
        response = response.replace("```json",'')
        response = response.replace("```",'')
        response = response.strip()
        parsed_output = parser.parse(response)
        parsed_output['total_token'] = total_token
        parsed_output['total_cost'] = total_cost
    except Exception as e:
        raise HTTPException(status_code=420, detail=str(e))

    return parsed_output

@app.post("/global/check-question-v3")
async def global_check_question_v3(req:KBChatAlsResultRequest):
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = Chain.question_check_v3_chain_chat(req.reference)
    chainJson = Chain.question_check_v3_json_parser()
    parser = Chain.question_checking_parser()
    total_token = 0
    total_cost = 0
    try:
        with get_openai_callback() as cb:
            response = await chain.arun(question=req.question,chat_history=chat_history,reference=req.reference)
            print(response)
            jsonparser = await chainJson.arun(question=response)
            total_token = cb.total_tokens 
            total_cost = cb.total_cost
        result = jsonparser.replace("```json",'')
        result = result.replace("```",'')
        result = result.strip()
        parsed_output = parser.parse(result)
        parsed_output['total_token'] = total_token
        parsed_output['total_cost'] = total_cost
    except Exception as e:
        raise HTTPException(status_code=420, detail=str(e))

    return parsed_output
                       
async def question_grammar_checking(req:QuestionCheckingRequest):
    chain = Chain.content_checking_chain()
    response = chain.run(question=req.question, answer=req.answer)
    jsonResponse = json.loads(response)
    return jsonResponse

@app.post("/global/generate-content")
async def global_generate_content(req:GenerateQuestionRequest):
    chain = Chain.content_chain()
    with get_openai_callback() as cb:
        response = await chain.arun(input_documents='', reference=req.reference, question=req.prompt)
        #print(cb)
    
    jsonResponse = json.loads(response)
    return jsonResponse

@app.post("/global/generate-content-learning")
async def global_generate_content_learning(req:GenerateContentRequest):
    chain = Chain.content_learning_chain()
    parser = Chain.learning_content_parser()

    with get_openai_callback() as cb:
        response = await chain.arun(input_documents='', reference=req.reference, question=req.prompt, student_level=req.student_level, student_name=req.student_name, learning_step=req.learning_step, substrand_name=req.substrand_name, topic_name=req.topic_name)
        #print(cb)
    response = response.replace("```json",'')
    response = response.replace("```",'')
    response = response.strip()
    parsed_output = parser.parse(response)
    return parsed_output

@app.post("/global/content-checking")
async def global_generate_content(req:ContentCheckingRequest):
    # in here will pass the content that from param to be check
    return True

@app.post("/knowledge-base/chatbot-als")
async def chatbotAls(request:Request, req:KBChatAlsRequest):
    print(str(request.url))
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = Chain.chabot_chain_als(req.chat_history)
    response = await chain.arun({"question": req.question,"chat_history":chat_history,"reference":req.reference,"learning_reference":req.learning_reference,"case_study":req.case_study})

    # Creates an instance of a speech config with specified subscription key and service region.
    speech_key = ""
    service_region = "southeastasia"

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # Note: the voice setting will not overwrite the voice element in input SSML.
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

    # use the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    text = response
    text = re.sub(r'_+', '_', text)
    result = speech_synthesizer.speak_text_async(text).get()
    # audio_id = str(uuid.uuid4())
    audio_id = rname()
    stream = speechsdk.AudioDataStream(result)
    # stream.save_to_wav_file(f"audio/{audio_id}.mp3")
    stream.save_to_wav_file(os.path.join(config.TMPPATH, f"{audio_id}.mp3"))
    audio_path = utils.return_mediaURL(filename=audio_id, extension='mp3', url=str(request.url))
    # audio = f"audio/{audio_id}.mp3"
    audio = audio_path.split("/")[-1]
    print('audio',audio, "\taudio_url:", audio_path)
    return {"text":response,"audio":audio, "audio-url": audio_path}

@app.post("/knowledge-base/chatbot-als-result")
async def chatbotAls(request:Request, req:KBChatAlsResultRequest):
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = Chain.chabot_chain_als_result(req.chat_history,req.reference)
    response = await chain.arun({"question": req.question,"chat_history":chat_history,"reference":req.reference})
    # Creates an instance of a speech config with specified subscription key and service region.
    speech_key = ""
    service_region = "southeastasia"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # Note: the voice setting will not overwrite the voice element in input SSML.
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

    # use the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    # print(response)
    # response = LatexNodes2Text().latex_to_text(response)
    # print(response)
    text = response
    text = re.sub(r'_+', '_', text)

    result = speech_synthesizer.speak_text_async(text).get()
    audio_id=rname()
    stream = speechsdk.AudioDataStream(result)
    stream.save_to_wav_file(os.path.join(config.TMPPATH, f"{audio_id}.wav"))
  
    audio_path = utils.return_mediaURL(filename=audio_id, extension='wav', url=str(request.url))
    audio = audio_path.split("/")[-1]

    return {"text":response,"audio":audio, "audio-url": audio_path}

@app.post("/knowledge-base/chatbot-als-practice")
async def chatbotAlsPractice(request:Request, req:KBChatAlsPracticeRequest):
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = Chain.chabot_chain_als_practice(req.chat_history,req.reference)
    response = await chain.arun({"question": req.question,"chat_history":chat_history,"reference":req.reference,"result_score":req.result_score})
    # Creates an instance of a speech config with specified subscription key and service region.
    speech_key = ""
    service_region = "southeastasia"

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    # Note: the voice setting will not overwrite the voice element in input SSML.
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"

    # use the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    text = response
    text = re.sub(r'_+', '_', text)
    result = speech_synthesizer.speak_text_async(text).get()
    audio_id=rname()
    stream = speechsdk.AudioDataStream(result)
    stream.save_to_wav_file(os.path.join(config.TMPPATH, f"{audio_id}.wav"))
    audio_path = utils.return_mediaURL(filename=audio_id, extension='wav', url=str(request.url))

    audio = audio_path.split("/")[-1]
    return {"text":response,"audio":audio, 'audio-url': audio_path}

@app.post("/global/auto-solution")
async def auto_solution(req:AutoSolutionRequest):
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = Chain.auto_solution_chain(req.chat_history)
    with get_openai_callback() as cb:
        response = await chain.arun({"question": req.question, "reference": req.reference,"chat_history":chat_history,"student_name":req.student_name,"student_answer":req.answer})
        # print(cb)
    return response

@app.post("/global/first-level-checking")
async def first_level(request:Request,req:KBChatAlsResultRequest):
    print('start flc')
    request_body = await request.json()
    language = request_body.get('language', '')
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = Chain.first_level_checking(req.chat_history)
    parser = Chain.first_level_checking_parser()
    total_token = 0
    total_cost = 0
    # cleaning question
    question = req.question 
    question = question.replace("&nbsp;"," ")
    try:
        with get_openai_callback() as cb:
            response = await chain.arun({"question": question, "language":language,"reference": req.reference,"chat_history":chat_history})
            total_token = cb.total_tokens 
            total_cost = cb.total_cost
        # if str(response) == "[]":
        #     parsed_output = 
        # else:
        parsed_output = parser.parse(response)
        # parsed_output['total_token'] = total_token
        # parsed_output['total_cost'] = total_cost
    except Exception as e:
        raise HTTPException(status_code=420, detail=str(e))
    
    return parsed_output

@app.post("/global/translation")
async def translation(req:KBChatAlsResultRequest):
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = Chain.translate_chain(req.chat_history)
    total_token = 0
    total_cost = 0
    parsed_output = []
    try:
        with get_openai_callback() as cb:
            response = await chain.arun({"question": req.question, "reference": req.reference,"chat_history":chat_history})
            total_token = cb.total_tokens 
            total_cost = cb.total_cost
    except Exception as e:
        raise HTTPException(status_code=420, detail=str(e))
    
    return {"result":response,"total_token":total_token,"total_cost":total_cost}

@app.post("/knowledge-base/chatbot-quiz-result")
async def chatbotAlsQr(request:Request, req:KBChatQuizResultRequest):
    request_body = await request.json()
    using_voice = request_body.get('is_using_voice', False)
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    total_token = 0
    total_cost = 0
    try:
        chain = await Chain.chabot_chain_quiz_result(req.chat_history,req.reference,req.language, question=req.question, image_library=req.image_description)
        with get_openai_callback() as cb:
            checkImageLibrary = await Chain.check_image_library(req.chat_history,req.reference,question=req.question, image_library=req.image_description)
            chain = await Chain.chabot_chain_quiz_result(req.chat_history,req.reference,req.language, question=req.question, image_library=checkImageLibrary)
            response = await chain.arun({"question": req.question,"chat_history":chat_history,"reference":req.reference})
            total_token = cb.total_tokens 
            total_cost = cb.total_cost
            # print("\n\n======RESP====\n", response)
            if response.upper().find("EXECUTE_AUTOGEN") > -1:
                #call autogen agents only the student ask a new Math problem related to knowledge in quiz.
                print("[CALL AUTOGEN] ==> solve problem in quiz review as  ==> ", req.question)
                if req.autogenver == 4.0: response = await autogen.find_solution_4mathprob(math_problem=req.question, language=req.language, return_json=False)
                elif req.autogenver == 4.1: response = await autogen.find_solution_4mathprobv2(math_problem=req.question, language=req.language, return_json=False)
                elif req.autogenver == 4.2: response = await autogen.find_solution_4mathprobv3(math_problem=req.question, language=req.language, return_json=False)
                else: response = "AUTO GENVERSION is invalid"
                try:
                    if 'cost' in response:
                        total_token += response['cost']['tokens']
                        total_cost  += response['cost']['cash']
                except : pass
        audio = ""
        audio_path = ""
        if using_voice:
            ## instantiating for Azure Speech Service
            speech_key = ""
            service_region = "southeastasia"

            speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

            ## instantiating for ElevenLabs
            elevenlabs.set_api_key(os.environ.get("ELEVENLABS_KEY"))

            text = response
            text = re.sub(r'_+', '_', text)

            if req.language != "":
                if req.voice.code == "en-SG":
                    p = inflect.engine()

                    numbers = re.findall(r'\b\d+\b', text)

                    for number in numbers:
                        words = p.number_to_words(number)
                        text = text.replace(number, words)

                    audio = elevenlabs.generate(
                        text = text,
                        voice = Voice(
                            voice_id = "7AyNFUIDwk2vkj8A8pt1",
                            settings = VoiceSettings(
                                stability = 0.5,
                                similarity_boost = 1,
                                style = 0,
                                use_speaker_boost = True
                            )
                        ),
                        model = "eleven_multilingual_v2"
                    )
                    audio_id = rname()
                    ## elevenlabs.save(audio, os.path.join(os.environ.get("TMP_PATH"),f"{audio_id}.mp3"))
                    elevenlabs.save(audio, os.path.join(config.TMPPATH, f"{audio_id}.mp3"))
                    audio_path = utils.return_mediaURL(filename=audio_id, extension='mp3', url=str(request.url))
                    audio = audio_path.split("/")[-1]
                else:
                    text = f"""
                    <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{req.voice.code}">
                        <voice name="{req.voice.name}">
                            {text}
                        </voice>
                    </speak>"""
                    result = speech_synthesizer.speak_ssml_async(ssml=text).get()
                    audio_id = rname()
                    stream = speechsdk.AudioDataStream(result)
                    stream.save_to_wav_file(os.path.join(config.TMPPATH, f"{audio_id}.mp3"))
                    audio_path = utils.return_mediaURL(filename=audio_id, extension='mp3', url=str(request.url))
                    audio = audio_path.split("/")[-1]
            else:
                text = f"""
                <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
                    <voice name="en-US-JennyNeural">
                        {text}
                    </voice>
                </speak>"""
                result = speech_synthesizer.speak_ssml_async(ssml=text).get()
                audio_id = rname()
                stream = speechsdk.AudioDataStream(result)
                stream.save_to_wav_file(os.path.join(config.TMPPATH, f"{audio_id}.mp3"))
                audio_path = utils.return_mediaURL(filename=audio_id, extension='mp3', url=str(request.url))
                audio = audio_path.split("/")[-1]
    except Exception as e:
        raise HTTPException(status_code=420, detail=str(e))
    # print('audio',audio, "\taudio_url:", audio_path)
    return {"text":response, "audio":audio, "audio-url": audio_path, 'total_token':total_token,'total_cost':float(total_cost),'img':checkImageLibrary}

@app.post("/global/rubric-score")
async def rubric_mark(req:RubricRequest):
    # parse pdf url to text and chunks
    chain = Chain.rubric_marking()
    parser = Chain.rubric_parser()
    total_token = 0
    total_cost = 0
    try:
        with get_openai_callback() as cb:
            response = await chain.arun({"question": req.question,"student_answer":req.student_answer,"model_answer":req.model_answer})
            print(response)
            parsed_output = parser.parse(response)
            total_token = cb.total_tokens 
            total_cost = cb.total_cost
    except Exception as e:
        raise HTTPException(status_code=420, detail=str(e))
    
    return {"result":parsed_output,"total_token":total_token,"total_cost":total_cost}

@app.post("/global/auto-answer")
async def auto_answer():
    # parse pdf url to text and chunks
    result = read_csv_to_array("chinese-paper-new.csv")
    result_json_array = []

    for row in result[1:]:
        question = row[0]
        answer_options = row[1]
        
        # Create a dictionary for the current row and append it to the result JSON array
        result_json_array.append({
            'question': question,
            'answer_options': answer_options
        })
    
    chain = Chain.auto_answer()
    result = []
    for res in result_json_array:
        detail_result = []
        response = await chain.arun({"question": "Anser this question below:\nQuestion: "+res['question']+"\nAnswer options: "+res["answer_options"]})
        res["correct_answer"] =  response
        detail_result.append(res['question'])
        detail_result.append(res['answer_options'])
        detail_result.append(res['correct_answer'])
        result.append(detail_result)
    

    workbook = xlsxwriter.Workbook('arrays.xlsx')
    worksheet = workbook.add_worksheet()

    # Define a format for the font color red
    red_font_format = workbook.add_format({'font_color': 'red'})

    dataanswer = {
        "Q1": 1, "Q2": 2, "Q3": 2, "Q4": 4, "Q5": 1, "Q6": 4, "Q7": 2, "Q8": 3, "Q9": 2, "Q10": 3,
        "Q11": 4, "Q12": 2, "Q13": 2, "Q14": 1, "Q15": 4, "Q16": 3, "Q17": 1, "Q18": 2, "Q19": 1, "Q20": 4,
        # Add all your questions and answers here in the same format
        "Q21": 2, "Q22": 4, "Q23": 3, "Q24": 2, "Q25": 2, "Q26": 1, "Q27": 3, "Q28": 3, "Q29": 1, "Q30": 2,
        "Q31": 1, "Q32": 4, "Q33": 4, "Q34": 2, "Q35": 2, "Q36": 2, "Q37": 3, "Q38": 1, "Q39": 4, "Q40": 1,
        "Q41": 2, "Q42": 3, "Q43": 3, "Q44": 1, "Q45": 1, "Q46": 2, "Q47": 2, "Q48": 2, "Q49": 3, "Q50": 2,
        "Q51": 1, "Q52": 3, "Q53": 4, "Q54": 3, "Q55": 2, "Q56": 3, "Q57": 4, "Q58": 4, "Q59": 2, "Q60": 2,
        "Q61": 2, "Q62": 4, "Q63": 2, "Q64": 2, "Q65": 4, "Q66": 3, "Q67": 1, "Q68": 2, "Q69": 1, "Q70": 3,
        "Q71": 3, "Q72": 3, "Q73": 1, "Q74": 3, "Q75": 1, "Q76": 3, "Q77": 3, "Q78": 2, "Q79": 3, "Q80": 1,
        "Q81": 4, "Q82": 3, "Q83": 2, "Q84": 4, "Q85": 3, "Q86": 3, "Q87": 4, "Q88": 1, "Q89": 2, "Q90": 4,
        "Q91": 2, "Q92": 3, "Q93": 4, "Q94": 2, "Q95": 3, "Q96": 3, "Q97": 1, "Q98": 4, "Q99": 2, "Q100": 1,
        "Q101": 2, "Q102": 4, "Q103": 4, "Q104": 2, "Q105": 3, "Q106": 1, "Q107": 2, "Q108": 3, "Q109": 2, "Q110": 3,
        "Q111": 4, "Q112": 4, "Q113": 1, "Q114": 3, "Q115": 2, "Q116": 3, "Q117": 4, "Q118": 1, "Q119": 2, "Q120": 3,
    }
    row = 0

    for row, data in enumerate(result):
        for col, cell_value in enumerate(data):
            # Check the condition for each cell value
            index = row + 1
            if col == 2 and int(cell_value) != dataanswer["Q"+str(index)]:  # This is where you specify your condition
                # If the condition is false, use the red font format
                worksheet.write(row, col, cell_value, red_font_format)
            else:
                # If the condition is true, write normally without formatting
                worksheet.write(row, col, cell_value)
    
    for row, (key, value) in enumerate(dataanswer.items()):
        print(row,key,value)
        worksheet.write(row,3,value)

    workbook.close()

    return {"result":result_json_array}

@app.post("/rubric-marking/get-score")
def rubric_marking(req:RubricMarkingRequest):
    response = rubric_mark(req.high_level_criteria, req.question, req.correct, req.student, req.student_class)
    return response

@app.post("/compo-marking/get-score")
def compo_marking_api(req:CompoMarkingRequest):
    response = compo_marking(req.question_statement, req.rubric_table, req.student_composition, req.model_composition, req.student_class)
    return response

@app.post("/compo-marking/get-score-v2")
async def compo_marking_api_v2(req:CompoMarkingRequest):
    print('start compo marking')
    response = await compo_marking_v2(req.question_statement, req.rubric_table, req.student_composition, req.model_composition, req.student_class)
    return response

@app.post("/ai-voice/tts")
def text_to_speech(request:Request,req:TTSVoiceRequest):
    input_text = req.text
    elevenlabs.set_api_key(os.environ.get("ELEVENLABS_KEY"))

    audio = elevenlabs.generate(
        text = input_text,
        voice = Voice(
            voice_id = req.voice_id,
            settings = VoiceSettings(
                stability = req.stability,
                similarity_boost = req.similarity_boost,
                style = req.style,
                use_speaker_boost = True
            )
        ),
      model = "eleven_multilingual_v2"
    )
    audio_id = rname()
    elevenlabs.save(audio, os.path.join(os.environ.get("TMP_PATH"),f"{audio_id}.mp3"))
    audio_path = utils.return_mediaURL(filename=audio_id, extension='mp3', url=str(request.url))
    return {"audio":audio_path,"audio_id":audio_id}

@app.post("/ai-voice/merge")
async def tts_merge(request:Request, req:TTSVoiceMerge):
    audio_looped = AudioSegment.silent()
    combined_audio = AudioSegment.silent()

    for item in req.audio_path:
        if item["type"] == "pause":
            silence_segment = AudioSegment.silent(duration = item["value"])
            audio_looped += silence_segment
        else:
            ## logic: get base64 from given url
            response = requests.get(item['value'])
            ## audio_segment = AudioSegment.from_mp3(item["value"])
            audio_segment = AudioSegment.from_file(io.BytesIO(response.content), format="mp3")

            """ audio_segment = AudioSegment.from_file(io.BytesIO(response.content), format="mp3", codec="mp3",
                                       parameters=["-analyzeduration", "2147483647", "-probesize", "2147483647"])"""
                                       
            audio_looped += audio_segment

    combined_audio += audio_looped
    audio_id = rname()
    combined_audio.export(os.path.join(os.environ.get("TMP_PATH"), f"{audio_id}.mp3"), format='mp3')
    audio_path = utils.return_mediaURL(filename=audio_id, extension='mp3', url=str(request.url))
    return {"audio":audio_path}

@app.post("/global/image-descriptor")
async def image_descriptor(req:KBChatAlsResultRequest):
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = Chain.image_description_chain(req.chat_history,req.reference)
    total_token = 0
    total_cost = 0
    parsed_output = []
    try:
        with get_openai_callback() as cb:
            response = await chain.arun({"question": req.question, "reference": req.reference,"chat_history":chat_history})
            total_token = cb.total_tokens 
            total_cost = cb.total_cost
    except Exception as e:
        raise HTTPException(status_code=420, detail=str(e))
    
    return {"result":response,"total_token":total_token,"total_cost":total_cost}

app.include_router(api_gen_ai.router)
#please dont add any code line below here
app.include_router(api_KB_pipeline.router)
app.include_router(api_adaptive_feedback.router)
app.include_router(yt_kb_pipeline.router)
app.include_router(api_learning_module.router)
app.include_router(api_lamini.router)


handler = Mangum(app)
if __name__ == "__main__":
    uvicorn.run(app, port=8008)
