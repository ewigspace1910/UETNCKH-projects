import requests
import json
import pandas as pd
from datetime import datetime
import time
from kafka import KafkaConsumer, KafkaClient

API_ENDPOIN = ""
TOPIC = "bi-stream"    
#some statistical variable
GLOBAL_TOTAL = 0
start_timestamp = last_timestamp = time.time()


def send2API(msg):
    global start_timestamp
    global GLOBAL_TOTAL
    global last_timestamp 

    if time.time() - start_timestamp > 86400:
        GLOBAL_TOTAL = 0
        start_timestamp = time.time()
    msg['total_global'] = GLOBAL_TOTAL = GLOBAL_TOTAL +  msg["total_per_second"]
    #datetime '2022-04-26T06:32:56.000+07:00
    timestamp = datetime.strptime(msg['timestamp'][:19], '%Y-%m-%dT%H:%M:%S').timestamp()    
    if timestamp < last_timestamp:
        return
    print(msg)
    last_timestamp = timestamp
    msg = bytes(json.dumps(msg).encode('utf-8'))
    requests.post(API_ENDPOIN, msg)


if __name__ == "__main__":
    consumer = KafkaConsumer(TOPIC)
    for data_json in consumer:
        msg = data_json.value.decode("utf-8")
        msg = json.loads(msg)
        send2API(msg)

        