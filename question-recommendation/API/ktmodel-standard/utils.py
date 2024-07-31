import base64
import requests
import json
import pandas as pd
import random
import itertools


class Config():
     # model_file_name
    def __init__(self, datacfg=None):

        # basic arguments
        self.gpu_id      = -1
        self.train_ratio =.8
        self.valid_ratio =.1
        self.batch_size  = 32
        self.n_epochs    = 55#55
        self.default     = 2
        self.learning_rate = 0.001

        # model, opt, dataset, crit arguments
        self.model_name='monacobert_ctt'
        self.optimizer  = 'adam'
        self.dataset_name = 'assist2017_pid_diff' #dont change
        self.crit = 'binary_cross_entropy'

        # bidkt's arguments
        self.max_seq_len = datacfg['model_cfg']['max_seq_len']
        self.num_encoder = datacfg['model_cfg']['num_encoder']
        self.hidden_size = datacfg['model_cfg']['hidden_size']
        self.num_head    = datacfg['model_cfg']['num_head']
        self.output_size = datacfg['model_cfg']['output_size']
        self.dropout_p   = datacfg['model_cfg']['dropout_p']
        self.use_leakyrelu =datacfg['model_cfg']['use_leakyrelu']

        # grad_accumulation
        self.grad_acc:bool=False
        self.grad_acc_iter:int=2


def get_data_from_db(db, query, url, file_name):

    body = {"db":db, "query":query}
    headers = {'Accept': 'application/json'}

    response = requests.get(url, params=body, headers=headers)
    output = json.loads(response.text)

    # Assuming you have received the JSON response and stored it in a variable called 'response'
    file_name = response['file_name']
    file_content_base64 = output['body']

    # Decode the base64-encoded file content
    file_content = base64.b64decode(file_content_base64)

    # Write the decoded file content to a file
    with open(file_name, 'wb') as file:
        file.write(file_content)
    
    return file_name

def process_data_4infer(df:pd.DataFrame, valid_topic:list, valid_skill:list, topic2strand:dict, max_seq_len:int, topic_id:int=None, substrand_id:int=None):
    diffs= [33, 50, 66, 90]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    df = df.dropna()
    df = df[-max_seq_len+1:]
    #create new df
    if not topic_id is None: #by topic
        topic_diff = [(topic, diff) for topic, diff in itertools.product([topic_id], diffs)]
    else : #by multi topics
        topic_diff = [(topic, diff) for topic, diff in itertools.product(valid_topic, diffs)]
    len_df = len(df)
    new_df= pd.concat([
                pd.concat([df.copy().assign(user_id=id), 
                        pd.DataFrame({len_df+id+1:{'item_id': topic, 
                                                   'skill_id': topic2strand[topic], 
                                                   'difficulty': diff, 
                                                   'correct': random.choice([0,1]), 
                                                   'user_id': id} }).T
                        ], ignore_index=True)
            for id, (topic, diff) in enumerate(topic_diff)])
    return new_df[['user_id','item_id', 'skill_id', 'difficulty', 'correct']]

