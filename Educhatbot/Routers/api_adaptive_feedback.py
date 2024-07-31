import pandas as pd
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Request
from typing import List
import os, re, time, random
from Modules.helper import Helper
from Modules.chain import Chain
import Modules.autogen.autogen as autogen
from langchain_core.output_parsers import JsonOutputParser
from langchain.callbacks import get_openai_callback
import azure.cognitiveservices.speech as speechsdk
import elevenlabs
from elevenlabs import Voice, VoiceSettings, generate
import inflect
from Modules import utils
import config

router = APIRouter(
    prefix="/adaptive-feedback",
    tags=['Adaptive Feedback']
)

class VoiceObject(BaseModel):
    code: str 
    name: str

class ChatItem(BaseModel):
    question: str 
    answer: str

class KBChatQuizResultRequest(BaseModel):
    reference: str
    question: str
    language: str
    voice: VoiceObject
    chat_history: List[ChatItem]
    autogenver: float = 4.0
    image_description: List[dict]

def rname(): return time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(0, 99)) 

@router.post("/chat")
async def af_generate_text(request:Request,req:KBChatQuizResultRequest):
    request_body = await request.json()
    using_voice = request_body.get('is_using_voice', False)
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    total_token = 0
    total_cost = 0
    baseLanguage = 'English'
    if req.language != '':
        baseLanguage = req.language 
    try:
        with get_openai_callback() as cb:
            ## predict question prompt -> 10 question in ref -> 1 question depends on the question that student want to ask
            preditChain = await Chain.check_image_library(req.chat_history[-4:],req.reference,question=req.question, image_library=req.image_description)
            checkImageLibrary = preditChain["image_library"]
            chain = await Chain.chabot_chain_quiz_result(req.chat_history,preditChain["related_question"],req.language, question=req.question, image_library=preditChain["related_image"])
            response = await chain.arun({"question": req.question,"chat_history":chat_history,"reference":preditChain["related_question"],"baseLanguage":baseLanguage})
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
            speech_key = "5baca4a1326e4c19a2e01090452018b5"
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

@router.post("/tts")
async def af_generate_audio(request:Request):
    request_body = await request.json()
    text = request_body.get('message')
    lang = request_body.get('language')
    voice_code = request_body.get('voice_code')
    voice_name = request_body.get('voice_name')

    try:
        audio = ""
        audio_path = ""

        ## instantiating for Azure Speech Service
        speech_key = "5baca4a1326e4c19a2e01090452018b5"
        service_region = "southeastasia"

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

        ## instantiating for ElevenLabs
        elevenlabs.set_api_key(os.environ.get("ELEVENLABS_KEY"))

        text = re.sub(r'_+', '_', text)

        if lang != "":
            if voice_code == "en-SG":
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
                <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{voice_code}">
                    <voice name="{voice_name}">
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
    return {"text":text, "audio":audio, "audio-url": audio_path}
