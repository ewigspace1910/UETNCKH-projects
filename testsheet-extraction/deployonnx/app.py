from mangum import Mangum
from fastapi.responses import JSONResponse
from fastapi import File, UploadFile, FastAPI, Request
import os
import time
from typing import List
from PIL import Image
import io


from deploy.detector_onnx import get_detector
# from detector_onnx import get_detector

app=FastAPI()
DETECTOR=get_detector()


###############################################################################
#   Routers configuration                                                     #
###############################################################################

@app.get("/")
async def root(request: Request):
    return {"message":  f"experiment API in: {str(request.url)}docs",
            "Machine configs-CPU": os.cpu_count(),
            "Model ready!": DETECTOR.ready(),
            "Json Documents"   : "https://github.com/ewigspace1910/DetectQuestForm4pdf/blob/main/deploy/model/api_doc.md" }


@app.post("/layout")
async def post_imgs_parallel(images:List[UploadFile]=File(...)):
    tstart = time.time()
    try:
        objs = []
        for img in images:
            contents = await img.read()
            objs += [Image.open(io.BytesIO(contents)).convert("RGB")]

        Detector =  DETECTOR
        layout = Detector.infer_concurrent(objs)

        res = JSONResponse({
            "status": "success",
            "data": layout,
            "excution time" : time.time() - tstart
        })
        return res
    except Exception as e:
        print(e)
        return JSONResponse({
            "status": "false",
            "data": {},
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
# import uvicorn
# if __name__ == '__main__':
#     uvicorn.run(app, host="localhost", port=9000)
