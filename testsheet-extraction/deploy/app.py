import uvicorn
from mangum import Mangum
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile, FastAPI, Request
import os
import time
from typing import List
import random
from PIL import Image
import io
import subprocess
# from deploy.detector_yolo import DetectorSJ
# from detector_yolo import DetectorSJ
from deploy.detector_paddle import DetectorSJ
# from detector_paddle import DetectorSJ
from deploy.detector_doc import DocExtraction
# from detector_doc import DocExtraction

app=FastAPI()
DETECTOR=DetectorSJ()
tmproot = os.getenv('path_tmp')
###############################################################################
#   Routers configuration                                                     #
###############################################################################

@app.get("/")
async def root(request: Request):
    return {"message":  f"experiment API in: {str(request.url)}docs",
            "Machine configs-CPU": os.cpu_count(),
            "Model ready!": DETECTOR.ready(),
            "Json Documents"   : "https://github.com/ewigspace1910/DetectQuestForm4pdf/blob/main/deploy/model/api_doc.md" }


@app.post("/layout", summary="extract layout of pdf file by detection model")
async def post_pdf_extraction(images:List[UploadFile]=File(...)):
    tstart = time.time()
    try:
        objs = []
        for img in images:
            contents = await img.read()
            objs += [Image.open(io.BytesIO(contents)).convert("RGB")]

        Detector =  DETECTOR
        layout = Detector.infer(objs)

        res = JSONResponse({
            "status": "success",
            "data": layout,
            "excution time" : time.time() - tstart
        })
        return res
    except Exception as e:
        print("ERROR in /pdf-layout router-->",e)
        return JSONResponse({
            "status": "false",
            "data": {},
            "excution time" : time.time() - tstart,
            "error": e
        })
    

@app.post("/layout-doc", summary="(trial) extract layout of doc file by rule ")
async def post_doc_extraction(docfile:UploadFile=File(...)):
    tstart = time.time()
    try:
        layout = []
        contents = await docfile.read()
        root, name = tmproot , time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(0, 99)) + ".docx"
        path = os.path.join(root, name)
        with open(path, "wb") as f: f.write(contents)
        detector = DocExtraction(root)
        layout = detector.infer(path)


        res = JSONResponse({
            "status": "success",
            "data": layout,
            "excution time" : time.time() - tstart
        })
        return res
    except Exception as e:
        print("ERROR in /layout-doc router-->",e)
        return JSONResponse({
            "status": "false",
            "excution time" : time.time() - tstart,
            "error": e
        })
    
@app.post("/pandoc", summary="for convert docx/dox to marddown and html")
async def post_pandoc(docfile:UploadFile=File(...)):
    tstart = time.time()
    try:
        contents = await docfile.read()
        root, name = tmproot , time.strftime("%Y%m%d-%H%M%S") + "-" + str(random.randint(0, 99)) + ".docx"
        path = os.path.join(root, name)
        with open(path, "wb") as f: f.write(contents)
        filename = path.split("/")[-1]
        file1 = os.path.join(root, f"{filename}.md")
        file2 = os.path.join(root, f"{filename}.html")
        subprocess.run(["pandoc", path, '-o', file1])        
        subprocess.run(["pandoc", path, '-o', file2])  

        markdown = "convert error!"
        html = "convert error"
        if os.path.exists(file1):
            with open(file1, "r") as f: markdown = "".join([line for line in  f.readlines() if line.strip()])
        if os.path.exists(file2):
            with open(file2, "r") as f: html     = "".join([line for line in  f.readlines() if line.strip()])

        try:
            os.remove(path)
            os.remove(file1)
            os.remove(file2)
        except: pass
        res = JSONResponse({
            "status": "success",
            "data": {"markdown":markdown, "html":html},
            "excution time" : time.time() - tstart
        })
        return res
    except Exception as e:
        print("ERROR in /pandoc router-->",e)
        return JSONResponse({
            "status": "false",
            "excution time" : time.time() - tstart,
            "error": e
        })

###############################################################################
#   Handler for AWS Lambda                                                    #
###############################################################################

handler = Mangum(app)

###############################################################################
#   Run the self contained application                                        #
###############################################################################

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=9000)
