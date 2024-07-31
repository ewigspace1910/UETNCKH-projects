import uvicorn
import os
import sys
sys.path.insert(0, os.getcwd())
from mangum import Mangum
from fastapi.responses import JSONResponse
from fastapi import  FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routers import graph_submit, api_kg_interaction, api_kt_interaction, api_wsg, api_analysis, api_practice_mode, api_practice_mode_teacher, api_db_adapter, api_learning_path

import warnings
warnings.filterwarnings('ignore')
app=FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS
origins = [
    "http://localhost",
    "http://localhost:8080",
    "null",
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


###############################################################################
#   Routers configuration                                                     #
###############################################################################

@app.get("/")
async def root(request: Request):
    return JSONResponse({"message":  f"experiment API in: {str(request.url)}docs"})


##### OTHER SIDE ########
app.include_router(api_kt_interaction.router)
app.include_router(api_practice_mode.router)
app.include_router(api_practice_mode_teacher.router)
app.include_router(api_kg_interaction.router)
app.include_router(api_wsg.router)
app.include_router(api_analysis.router)
app.include_router(api_learning_path.router)

app.include_router(api_db_adapter.router)
app.include_router(graph_submit.router)



###############################################################################
#   Handler for AWS Lambda                                                    #
###############################################################################

handler = Mangum(app)

###############################################################################
#   Run the self contained application                                        #
###############################################################################

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=9090)
