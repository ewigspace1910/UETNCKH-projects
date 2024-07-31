import os
KGsupported_subjects = [2, 1]
#folder to store temporary file
TMPROOT = "" if os.getenv("tmp_path") is None else os.getenv("tmp_path")
#min number questions in topic.
MIN_QUESTS_IN_TOPIC=25
#S3 bucket
BUCKER_NAME = "pdf-digitalization"

ERROR = {  

    200: "This student doesn't have enough records in DB to estimate his proficiency",
    201: "This student hasn't been updated in DB yet, please call update API",

    #related to KT-model
    400: "This subject is not supported by current KT-based recommender",


    #related new topic/substrand id
    300: "Topic/substrand id is incorrect",
    301: "Topic/substrand id is not supported by current KT-based recommender",
    303: "This topic have no questions at same student level",
    304: "This substrand have no topic that KT model support",
    #Nan
    -1: "No error"
}

MASTERY_LEVEL = {
    'c2': 0.75,
    'c1': 0.60,
    'b2': 0.40,
    'b1': 0.25,
    'a2': 0.15,
    'a1': 0.05
}

#link HTTP 
OPENAI_API_KEY= "" if os.getenv("OPENAI_API_KEY") is None else os.getenv("OPENAI_API_KEY")
URL_QUERY = "" #backup database host
URL_KTmodel = {
    "update_user_inDB": "",
    "infer_user_topic": ""
    }

db_config = {
    "pals": {
        "hostname": "127.0.0.1",
        "user": "admin",
        "password": "qllESKF7SZu9rK7C499G",
        "schema_name": "saas_main",
        # "port": 3306
        "port": 3308
    },
    "alias4urDB": {
        "hostname": "<new_host>",
        "user": "user",
        "password": "password",
        "schema_name": "saas_main",
        "port": "port" #3307
    }
}