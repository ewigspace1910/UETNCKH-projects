import os
import re 
import json
import cv2
import io
import requests
import time
import random
import boto3
from botocore import UNSIGNED
from botocore.client import Config as AWSConfig
import multiprocessing as mp
import concurrent.futures

# CONSTANTs
CATEGORY_MAP = {
    "heading": 0, 
    "question": 1,  
    "subquestion": 2,
    "choice"  : 3,
    "image"   : 4,
    "table"   : 5,
    "blank"   : 6,
    "auxillary_text"   : 7
}
CATEGORY_LEVEL = {
    "heading": 5, 
    "question": 4,  
    "subquestion": 3,
    "choice"  : 1,
    "image"   : 1,
    "table"   : 1,
    "blank"   : 1,
    "auxillary_text"   : 1
}
REVERSE_MAP = {CATEGORY_MAP[k]:k for k in CATEGORY_MAP}
class Config:
    def __init__(self):
        
        #key-id
        self.matpix_id = os.getenv("mathpix_id") 
        self.matpix_key = os.getenv("mathpix_key") 
        self.s3_bucket_name = os.getenv("aws_s3_buckname")

        #detection model
        # self.store_path =  "data/model/rt-dert" #os.getenv("path_weight_store") 
        # self.cfg_path   =   "data/model/rt-dert/infer_cfg.yml" #os.getenv("path_model_cfg") 
        self.store_path =  os.getenv("path_weight_store") 
        self.cfg_path   =   os.getenv("path_model_cfg") 
        #same args
        self.threshold  = 0.5 if os.getenv("infer_threshold") is None else float(os.getenv("infer_threshold"))
        self.device     = "cpu" if os.getenv("infer_device") is None else os.getenv("infer_device")
        self.n_processes = 1  if os.getenv("infer_n_process") is None else int(os.getenv("infer_n_process")) #for aws
        self.batchsize  = 1  if os.getenv("infer_batchsize") is None else int(os.getenv("infer_batchsize")) #for aws

        self.ocr_req_deplay = 10 if os.getenv("mathpix_delay") is None else int(os.getenv("mathpix_delay"))
        self.ocr_req_max = 20 if os.getenv("mathpix_max_req_per_min") is None else int(os.getenv("mathpix_max_req_per_min"))

################## REQUEST #####################
def request2mathpix(io_buffer, app_id="", app_key=""):
    try:
        r = requests.post("https://api.mathpix.com/v3/text",
            files={"file": io_buffer},
            data={
            "options_json": json.dumps({
                "math_inline_delimiters": ['<span class="math-tex" id="0">\(', "\)</span>"],
                "rm_spaces": True
            })
            },
            headers={
                "app_id": app_id ,
                "app_key":app_key 
            }
        )
        r = r.json()
        if "text" in r.keys():
            return r['text'].strip()
        else :
            return ""
    except Exception as e:
        print("Mathpix_ocr_requestion got error :", e)
        return "???"

def request2s3(file, s3name):
        file_name = "image/" + time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(0, 99)) + ".png"
        try:
            s3 = boto3.client('s3', config=AWSConfig(signature_version=UNSIGNED))
            s3.upload_fileobj(file, s3name, file_name) #upload to s3.
            text = "https://{}.s3.amazonaws.com/{}".format(s3name, file_name)
            return text
        except Exception as e:
            print("S3 error", e)
            return ""  
        

#######################################################
def subprocess_ocr_request(index, cls, cropped_image, mathpix_id, mathpix_key, s3name, conn):
    """
    "inputs" contains:
        page_index  : index of page containing box
        box         : detectedbox list from yolov8 results
        cropped_image : (numpy array): original imaege
        app_id, app_key
    """
    _, buffer = cv2.imencode(".png", cropped_image)
    io_buffer = io.BytesIO(buffer)
    io_buffer.seek(0)
    if cls in ["image", "table"]: 
        text = request2s3(file=io_buffer, s3name=s3name)
    else: 
        text = request2mathpix(io_buffer=io_buffer, app_id=mathpix_id, app_key=mathpix_key)
    obj = { "name":cls, 
            "text": text,
            "category": "OEQ",
            "idx": index}
    
    conn.send(obj)
    conn.close()

def submit_ocr_request(detection_results,  config:Config):
    """
    This function submits OCR requests for detected text boxes using Mathpix OCR API and returns the OCR
    results.
    
    :param detection_results: A list of tuples containing the page index, class label, and cropped image
    of detected objects in a document
    :param config: The `config` parameter is an instance of the `Config` class, which contains various
    configuration settings for the OCR request process
    :type config: Config
    :return: a list of OCR results for each box in the detection results.
    """
    parent_connections = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        mathpix_requests=0
        for box in detection_results: #box = [page_index (tuple), cls(str), cropped_image(np.narray)]
            if not box[1] in ["image", "table"]:
                mathpix_requests += 1
                if mathpix_requests % config.ocr_req_max == 0: time.sleep(config.ocr_req_deplay)
            parent_conn, child_conn = mp.Pipe()
            parent_connections += [parent_conn]      
            executor.submit(subprocess_ocr_request, box[0], box[1], box[2], config.matpix_id, config.matpix_key, config.s3_bucket_name, child_conn)
    elements = []
    for parent_connection in parent_connections:
        elements += [parent_connection.recv()]
    return elements


def group_element(elements):
    """Args:
        - orders : (list[int])   Contain category_weights corresponding elements
        - elements: (list[dict]) Contain all detected element in order from top to bottom of page, from page 0 to page n
        Return: json object like html document object
    """
    def __combine_min_elements(orders, elements):
        '''
        order (list) --> rank of elements
        elements (list) --> containing dictionary objects  contain information of question, must be containt key["child"] = []
        '''
        arr = orders
        idx_e = [i for i in range(len(elements))]
        order = min(orders)
        #Group min elements in orders list : [2 0 0 3 0 1 0 0 1 0 3 2 0 1 0  ] -> [2, [0, 0], 3 [0], 1 [0, 0] 1 [0] 3 2 [0] 1 [0] ]
        tmp, tmp_e, new_arr, new_arr_e =[], [], [], []
        i = len(arr)-1
        while i >= 0:
            if arr[i] > order:
                new_arr = [arr[i], tmp] + new_arr if len(tmp) > 0 else [arr[i]] + new_arr
                new_arr_e = [idx_e[i], tmp_e[::-1]] + new_arr_e if len(tmp_e) > 0 else [idx_e[i]] + new_arr_e
                tmp, tmp_e = [], []
            else:
                tmp += [arr[i]]
                tmp_e += [idx_e[i]]
            i -= 1
        #reduce array for [2, [0, 0], 3 [0], 1 [0, 0] 1 [0] 3 2 [0] 1 [0] ] -> [2, 3, 1, 1, 3, 2, 1]
        new_arr_2, new_arr_2_e = [], []
        i = 0
        while i < len(new_arr):
            if type(new_arr[i]) == int:
                new_arr_2 += [new_arr[i]]
                new_arr_2_e += [new_arr_e[i]]
            else:
                for j in new_arr_e[i]:
                    if elements[j]["name"] in ("image", "table"):
                        if "stimulus" in elements[new_arr_e[i-1]].keys():
                            elements[new_arr_e[i-1]]["stimulus"] += [elements[j]['text']]
                        else: elements[new_arr_e[i-1]]["stimulus"] = [elements[j]['text']]

                    elif elements[j]["name"] in ("auxillary_text", "blank") and len(elements[j]["text"]) > 6:
                        elements[new_arr_e[i-1]]["text"] += f"\n{elements[j]['text']}"

                    elif elements[j]["name"] in ("choice"):
                        if "choices" in elements[new_arr_e[i-1]].keys():
                            elements[new_arr_e[i-1]]["choices"] += parse_choice2dict(elements[j]['text'])#f"\n{elements[j]['text']}"
                        else: elements[new_arr_e[i-1]]["choices"] = parse_choice2dict(elements[j]['text']) #f"\n{elements[j]['text']}"
                        elements[new_arr_e[i-1]]["category"] = "MCQ"

                    elif elements[j]["name"] in ("subquestion", "question"):
                        if "subquestions" in elements[new_arr_e[i-1]].keys():
                            elements[new_arr_e[i-1]]["subquestions"] += [ elements[j] ]
                        else: elements[new_arr_e[i-1]]["subquestions"] = [ elements[j] ]
                        elements[new_arr_e[i-1]]["category"] = "MSQ"
                    else: pass 
            i += 1

        new_arr_2_e = [elements[i] for i in new_arr_2_e]
        return new_arr_2, new_arr_2_e
    
    #######################
    ## exclude unessesary headings --> example: 3322110310322302121023 --> 3221101022302121102
    new_elements=[]
    orders   =[]
    for i, item in enumerate(elements):
        if item["name"] == "heading" and i > 0:
            if elements[i-1]['name'] == "heading":
                elements[i-1]['text'] += f"\n{item['text'].strip()}"
                continue
            elif i== len(elements)-1:continue
            elif elements[i+1]['name'] in ["subquestion"]: continue

        # if item["name"] == "auxillary_text" and i > 0 and len(item['name'].split()) > 5: #hard rule
        #     if elements[i-1]["name"] in ['choice', 'blank']: item["name"]="heading" #after choice/blank ---> heading

        del item["idx"]
        orders += [CATEGORY_LEVEL[item['name']]]
        new_elements += [item]

    ## Grouping items
    if len(orders) == 0 : return []
    if orders[0] < max(CATEGORY_LEVEL.values()):
        orders = [max(CATEGORY_LEVEL.values())] + orders
        new_elements = [{"name":'heading', 
                    "text": "Section - 1",
                    "category": "MSQ"}] + new_elements
    o, e = orders, new_elements
    while len(set(o)) > 1:
        o, e = __combine_min_elements(o,e)
    return e

def parse_choice2dict(text):
    items = [line.strip() for line in text.splitlines() if line.strip()] 
    results = []

    for item in items:
        match = re.match(r"\(\w+\)\s*(.+)", item)

        if match:
            value = match.group(1)
            results += [value]
    if len(results) == 0 : return items
    else: return results