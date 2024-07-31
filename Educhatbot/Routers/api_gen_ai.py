import pandas as pd
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Request
from typing import List
import json
import time
from Modules.helper import Helper
from Modules.chain import Chain
from Modules.question_generator.chain import QuestionGeneratorChain
from Modules.question_generator.agents import create_thread_and_run, wait_on_run, get_response, submit_message_kb, create_thread_and_run_kb, wait_on_run_kb, get_response_kb, submit_message_kb_simple
import Modules.autogen.autogen as autogen
from Modules.extentions.automarker_processor import clean_question_text
from langchain_core.output_parsers import JsonOutputParser
from langchain.callbacks import get_openai_callback
import requests, os, config, random

router = APIRouter(
    prefix="/gen-ai",
    tags=['Generative AI']
)

class ChatItem(BaseModel):
    question: str 
    answer: str

class ChatWithReferenceRequest(BaseModel):
    reference: str
    question: str
    chat_history: List[ChatItem]

class ChatWithKBRequest(BaseModel):
    is_passage: bool = False 
    kb_ids: List[str]
    question: str
    passage: str = ""
    chat_history: List[ChatItem]
    example: str
    version: float = 2
    passage_thread_id: str = ""
    question_thread_id: str = ""

class QgenRequest(BaseModel):
    reference: str
    question: str
    question_type_id: int
    chat_history: List[ChatItem]
    version: float = 2
    type: str = 'rephrase'
    rephrase: str  = 'object' #full/object[including noun, number]/noun[noun only]

class QgenAutomateRequest(BaseModel):
    model_version: float
    question: str = "Generate a new question based on the given question"
    chat_history: List[ChatItem]
    input_path: str = "/tmp/test.json"
    output_path: str = "/tmp/out-test.json"

def json_to_dataframe_decoder(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        json_string = file.read()
    
    try:
        json_object = json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"Error decding JSON: {e}")

    json_object = json.loads(json_string)
    df = pd.DataFrame(json_object['question'])

    ### df = df.iloc[0:1]

    return df

@router.post("/generate-question")
async def generate_question(req:ChatWithReferenceRequest):
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = Chain.question_v3_chain_chat(req.chat_history,req.reference)
    parser = Chain.question_parser()
    total_tokens = 0
    total_cost = 0
    with get_openai_callback() as cb:
        response = await chain.arun(question=req.question,chat_history=chat_history,reference=req.reference)
        total_tokens = cb.total_tokens
        total_cost = cb.total_cost
    try:
        parsed_output = parser.parse(response)
    except :
        parsed_output = ""
        print("ERR in parsing output to json -->",response)
    return {"result":parsed_output,"total_tokens":total_tokens,"total_cost":total_cost}

@router.post("/generate-question-kb")
async def generate_question_kb(req:ChatWithKBRequest):
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = Chain.question_kb_chain_chat(req.chat_history,req.kb_ids)
    chain_json = QuestionGeneratorChain.convert_to_json_chain_kb()
    chain_json_simple = QuestionGeneratorChain.convert_to_json_chain_kb_simple()
    structure_json = QuestionGeneratorChain.restructure_question()
    parser = QuestionGeneratorChain.question_parser_kb()
    total_tokens = 0
    total_cost = 0
    with get_openai_callback() as cb:
        if req.version == 2:
            if req.is_passage == False:
                thread, run = await create_thread_and_run_kb(
                    req.question, req.kb_ids[0]
                )
                run = await wait_on_run(run, thread)
                # pretty_print(get_response(thread))
                response = await get_response(thread)
                response = response.data[0].content[0].text.value
                # response = await structure_json.arun({"question":response})
                response = await chain_json.arun({"question":response})
                response = response.replace("```json",'')
                response = response.replace("```",'')
                response = response.strip()
                print(response)
            else:
                PASSAGE_ASSISTANT_ID="asst_qHhilfLmrDsjCwso7AFTKc1t"
                # QUESTION_ASSISTANT_ID="asst_esyvCTtaihrMZM3eKroY9Lb7"
                # QUESTION_ASSISTANT_ID="asst_gaQdAhv8IzlIMnqGYGV2RasM"
                QUESTION_ASSISTANT_ID="asst_vEpIaKr75HpyRDRcfFWGAdnz"

                # run passage assistant
                print("passage")
                print(req.passage_thread_id)
                run = await submit_message_kb_simple(
                    PASSAGE_ASSISTANT_ID, req.passage_thread_id
                )

                run = await wait_on_run_kb(run, req.passage_thread_id)
                print("done run")
                # pretty_print(get_response(thread))
                passage = await get_response_kb(req.passage_thread_id)
                passage = passage.data[0].content[0].text.value

                 # run question assistant
                qn = req.question+f" \n\n Passage: {passage}"
                run = await submit_message_kb(
                    QUESTION_ASSISTANT_ID, req.question_thread_id, qn
                )
                run = await wait_on_run_kb(run, req.question_thread_id)
                print("done run")
                # pretty_print(get_response_kb(thread))
                question = await get_response_kb(req.question_thread_id)
                question = question.data[0].content[0].text.value

                response = f"Questions: {question}"

                # response = await structure_json.arun({"question":response})
                response = await chain_json_simple.arun({"question":response})
                response = response.replace("```json",'')
                response = response.replace("```",'')
                response = response.strip()
        else:
            response = chain({"question":req.question,"chat_history":chat_history})
            print(response["answer"])
            response = await chain_json.arun({"question":response["answer"]})
            response = response.replace("```json",'')
            response = response.replace("```",'')
            response = response.strip()
            # print(response)
            total_tokens = cb.total_tokens
            total_cost = cb.total_cost
    try:
        parsed_output = parser.parse(response)
        if req.is_passage == True:
            for qn in parsed_output["questions"]:
                qn["question_header"] = passage+' \n\n '+qn["question_header"]
    except :
        parsed_output = ""
        print("ERR in parsing output to json -->",response)
    return {"result":parsed_output,"total_tokens":total_tokens,"total_cost":total_cost}

@router.post("/generate-assistant")
async def generate_assistant(request:Request):
    # parse pdf url to text and chunks
    request_body = await request.json()
    thread_id = request_body.get('thread_id')
    assistant_id = request_body.get('assistant_id')
    message = request_body.get('message')

    print("initiate")
    run = await submit_message_kb_simple(
        assistant_id, thread_id, message
    )

    run = await wait_on_run_kb(run, thread_id)
    # pretty_print(get_response(thread))
    response = await get_response_kb(thread_id)
    print("done run")
    response = response.data[0].content[0].text.value

    return {"result":response}

@router.post("/question-generator")
async def question_generator(req:QgenRequest):
    # parse pdf url to text and chunks
    chat_history = Helper.normalized_history(req.chat_history)
    chain = QuestionGeneratorChain.question_chain(req.chat_history,req.reference,req.question_type_id,req.version,req.type)
    chainJson = QuestionGeneratorChain.convert_to_json_chain(req.question_type_id)
    parser = QuestionGeneratorChain.question_parser(req.question_type_id)
    total_token = 0
    total_cost = 0
    response_new = ""
    is_image = False
    refsquest = json.loads(req.reference)

    for item in refsquest:
        if item["type"] == "image":
            is_image = True
    try:
        use_old_version = True
        if 5> req.version >=4: 
            #check type of question
            chain_2classifier = QuestionGeneratorChain.categorize_question(req.reference)
            response         = await chain_2classifier.arun({})
            # question_subject = parser.parse(response.replace("```json",'').replace("```",'').strip())
            # if question_subject['subject'].lower().find('math') > -1:
            if response.lower().find('math') > -1:
                #call autogen
                parsed_output = await question_generator_mathonly(req) #version is only used for math problem
                use_old_version = False
            else: 
                #with other subject ==> call the old version (1,2,3), default = 2
                req.version = 2
                use_old_version = True
                pass

        if use_old_version:
            if req.version == 3 and is_image == False: 
                thread, run = await create_thread_and_run(
                    req.reference, req.question_type_id
                )
                run = await wait_on_run(run, thread)
                # pretty_print(get_response(thread))
                response_new = await get_response(thread)
                response_new = response_new.data[0].content[0].text.value
                response = await chainJson.arun(question=response_new)
                parsed_output = parser.parse(response)
                parsed_output['total_token'] = total_token
                parsed_output['total_cost'] = total_cost
            
            else:
                with get_openai_callback() as cb:
                    new_question = await chain.arun({"question":req.question,"chat_history":chat_history,"reference":req.reference})
                    total_token += cb.total_tokens 
                    total_cost += cb.total_cost
                    # print(cb)
                    # thread, run = await validate(new_question)
                    # run = await wait_on_run(run, thread)
                    # validated_question = await get_response(thread)
                    # validated_question = validated_question.data[0].content[0].text.value
                with get_openai_callback() as cb:
                    response = await chainJson.arun(question=new_question)
                    total_token += cb.total_tokens 
                    total_cost += cb.total_cost
                    # print(cb)
                response = response.replace("```json",'')
                response = response.replace("```",'')
                response = response.strip()
                parsed_output = parser.parse(response)
                parsed_output['total_token'] = total_token
                parsed_output['total_cost'] = total_cost
    except Exception as e:
        raise HTTPException(status_code=420, detail=str(e))

    return parsed_output


@router.post("/question-generator-math")
async def question_generator_mathonly(req:QgenRequest):
    chat_history = Helper.normalized_history(req.chat_history)
    parser= QuestionGeneratorChain.question_parser(question_type_id=-1)
    total_token = total_cost = 0
    #GEN QUESTION
    
    with get_openai_callback() as cb:
        chain = QuestionGeneratorChain.gen_only_question_chain(req.chat_history, req.reference, rephrase=req.rephrase)
        new_question_gen = await chain.arun(question=req.question, chat_history=chat_history)
        new_question_gen = new_question_gen.replace("```json",'').replace("```",'').strip()
        try: new_questions = parser[0].parse(new_question_gen)
        except: new_questions = JsonOutputParser().parse(new_question_gen)
        # print(cb)
        total_token += cb.total_tokens 
        total_cost  += cb.total_cost  
        
    # print(new_questions)
    # Gen solution + answer by autoGen
    is_image = False
    questNsolution = []
    flag = False
    question = ""
    try:
        for idx, item in enumerate(json.loads(req.reference)):
            if  item["type"] == "image":
                is_image = True
            else:
                if item['text'].lower().find("question") > -1:
                    question += clean_question_text(item['text']) + " \n "
                elif item['text'].lower().find('solution') > -1:
                    flag = True; continue
                elif flag:
                    flag = False
                    solution = clean_question_text(item['text'])
                    if len(solution) > 10: 
                        questNsolution += [(question, solution)]
                    question = ""
    except : pass
    reference = ""
    for idx, (q, s) in enumerate(questNsolution):
        reference += f"""
        + [Ref Question #{idx}] : {q}
        + [Ref Solution #{idx}] : {s}
        ==================================

        """

    for question in new_questions['questions']: 
        if req.version == 4: solution = await autogen.find_solution_4mathprob(question['question'], reference, language=question['lang'] ,use_image=is_image)
        elif req.version == 4.1: solution = await autogen.find_solution_4mathprobv2(question['question'], reference, language=question['lang'] ,use_image=is_image)
        elif req.version == 4.2: solution = await autogen.find_solution_4mathprobv3(question['question'], reference, language=question['lang'] ,use_image=is_image)
        # print(solution)
        for k, v in solution.items():
            question[k] = v
    if 'cost' in solution:
        total_token += solution['cost']['tokens']
        total_cost  += solution['cost']['cash']
    new_questions['total_token'] = total_token
    new_questions['total_cost']  = total_cost
    return new_questions


@router.post("/question-math-rephraser")
async def question_generator_rephraser(req:QgenRequest):
    chat_history = Helper.normalized_history(req.chat_history)
    parser= QuestionGeneratorChain.question_parser(question_type_id=-1)
    total_token = total_cost = 0
    #GEN QUESTION
    
    with get_openai_callback() as cb:
        chain = QuestionGeneratorChain.gen_only_question_chain(req.chat_history, req.reference, rephrase=req.rephrase)
        new_question_gen = await chain.arun(question=req.question, chat_history=chat_history)
        new_question_gen = new_question_gen.replace("```json",'').replace("```",'').strip()
        try: new_questions = parser[0].parse(new_question_gen)
        except: new_questions = JsonOutputParser().parse(new_question_gen)
        
        total_token += cb.total_tokens 
        total_cost  += cb.total_cost  
        
    new_questions['total_token'] = total_token
    new_questions['total_cost']  = total_cost
    return new_questions

@router.post("/question-generator-automate")
async def question_generator_automation(req:QgenAutomateRequest):
    input_path = req.input_path #"../tmp/P5-PB.json"
    dataframe = json_to_dataframe_decoder(input_path)
    temp_df = pd.DataFrame(columns = ['original_question', 'question_header', 'question', 'difficulty_level', 'mark', 'solution', 'answer_options', 'correct_answer'])
    count = 0
    for index, row in dataframe.iterrows():
        if count == 100: continue
        else : count += 1
        time.sleep(2)
        try:
            row_dict = dataframe.iloc[index].to_dict()
            print("Processing question number " + str(index + 1))

            question_type = row_dict['question_type_id']
            refprompt = row_dict['prompt']
            
            qgen_req = QgenRequest(
                reference = refprompt,
                question = req.question,
                chat_history = req.chat_history,
                question_type_id = question_type,
                version = req.model_version
            )
            
            output = await question_generator(qgen_req)
            response = output

            data = response['questions'][0]
            flag = False
            for item in json.loads(refprompt):
                if flag : 
                    data['original_question'] = item['text']
                    break
                if 'text' in item and item['text'].find("question :") > -1: flag=True

            answer_options = data.pop('answer_options')
            print(data)
            df = pd.DataFrame([data])
            df['answer_options'] = [answer_options]

            temp_df = pd.concat([temp_df, df], ignore_index = True)

        except json.JSONDecodeError as e:
            # Handle the JSON decoding error
            print(f"Skipping item '{index}' due to JSON decoding error: {e}")
        except Exception as e:
            # Handle other types of errors if necessary
            print(f"Skipping item '{index}' due to an error: {e}")
    output_path = req.output_path
    output_path = "./result-qgen.csv"
    temp_df.to_csv(output_path, index = True)

# @router.post("/generate-question-pdf")
# async def generate_question_pdf(request:Request):
#     request_body = await request.json()
#     url = request_body.get('file_url')
#     print(url)
#     contents = requests.get(url).content
    
#     root, randname = os.path.join(config.TMPPATH) , time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(0, 999))
#     filename = randname #if filename is None else filename   
#     path = os.path.join(root, filename)
#     imgspath = os.path.join(root, "imgs", filename)
#     if not os.path.exists(path): os.makedirs(path)
#     if not os.path.exists(imgspath): os.makedirs(imgspath)
#     pdf_path = os.path.join(root, f"{filename}.pdf")
#     txt_path = os.path.join(path, f"{filename}.txt")
#     with open(pdf_path, "wb") as f: f.write(contents)

#     pdf_to_docx(pdf_path)

#     return {"result":True}