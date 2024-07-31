import logging
import os
from dotenv import load_dotenv
# Load the .env file
load_dotenv()


logging.basicConfig(level=logging.ERROR,format='%(levelname) -10s %(asctime)s %(module)s:%(lineno)s %(funcName)s %(message)s')
LOGGER = logging.getLogger()
TMPPATH = "../tmp" if os.getenv("TMP_PATH") is None else os.getenv("TMP_PATH")
S3BUCKNAME = "pdf-digitalization" if os.getenv("aws_s3_buckname") is None else os.getenv("aws_s3_buckname") 
# OPENAI_API_KEY = "" if os.getenv("OPENAI_API_KEY") is None else os.environ.get("OPENAI_API_KEY")
OPENAI_API_KEY_DICT = {
     "default": "" if os.getenv("OPENAI_API_DKEY") is None else os.environ.get("OPENAI_API_DKEY"),
     "AUTOGEN": "" if os.getenv("OPENAI_API_KEY_AUTOGEN") is None else os.environ.get("OPENAI_API_KEY_AUTOGEN"),
     "AF-KB"  : "" if os.getenv("OPENAI_API_KEY_AFKB") is None else os.environ.get("OPENAI_API_KEY_AFKB"),
     "AF"     : "" if os.getenv("OPENAI_API_KEY_AF") is None else os.environ.get("OPENAI_API_KEY_AF"),
     "QG"     : "" if os.getenv("OPENAI_API_KEY_QGEN") is None else os.environ.get("OPENAI_API_KEY_QGEN"),
     "CONTENT_CHECKING" : "" if os.getenv("OPENAI_API_KEY_CONTENT_CHECKING") is None else os.environ.get("OPENAI_API_KEY_CONTENT_CHECKING"),
     "RUBRIC_MARKING"   : "" if os.getenv("OPENAI_API_KEY_RUBRIC_MARKING") is None else os.environ.get("OPENAI_API_KEY_RUBRIC_MARKING"),
     "VISION"   : "" if os.getenv("OPENAI_API_KEY_VISION") is None else os.environ.get("OPENAI_API_KEY_VISION"),

     ""       : ""
}
CHROMA_DB_DIRECTORY = "../tmp/vector" if os.getenv("dbvector_path") is None else os.getenv("dbvector_path") 

OPENSEARCH_ACCOUNT = "ducanh:ducanh@abJU4LD" if os.getenv("OPENSEARCH_ACCOUNT") is None else os.environ.get("OPENSEARCH_ACCOUNT")
OPENSEARCH_URL = "" if os.getenv("OPENSEARCH_URL") is None else os.environ.get("OPENSEARCH_URL")
LOGVERBOSE=True if os.getenv("log_verbose") is None else bool(os.environ.get("log_verbose"))
if not os.path.exists(CHROMA_DB_DIRECTORY) : os.makedirs(CHROMA_DB_DIRECTORY)

#OCR PIPELINE
BATCH_SIZE = 4 if os.getenv("KBB-OCR-BATCHSIZE") is None else int(os.environ.get("KBB-OCR-BATCHSIZE"))
SLEEP_PER_PAGE = 8 if os.getenv("KBB_OCR_SLEEPINTERVAL") is None else int(os.environ.get("KBB_OCR_SLEEPINTERVAL"))
SLEEP_INTERVAL = 25 if os.getenv("SLEEPINTERVAL_TIME") is None else int(os.environ.get("SLEEPINTERVAL_TIME"))

######KB SETING #######
CHUNK_SIZE = 550
CHUNK_OVERLAP = 50
SPLITTER = "R"
k = 5

# LLM MODEL
temperature = 0.1
max_tokens=2000
model_name = "gpt-4-1106-preview" #gpt-4/gpt-4-1106-preview
chain_type = "stuff" #['stuff', 'refine', 'reduce'/'map_reduce']



######### DB ######
URL_QUERY = "" #backup database host
db_config = {
    "indb": {
        "hostname": "127.0.0.1",
        "user": "ducanh",
        "password": "jasklfducanh44",
        "schema_name": "saas_main",
        "port": 3306
    }
}
DID_API_KEY = "" if os.getenv("DID_API_KEY") is None else os.getenv("DID_API_KEY")