ID=1  
DATA_PATH="./_data/residen1.csv"
DELAY_TIME=1
LOOP_TIMES=3
TOPIC="test"
IP="localhost:9092"

source venv/bin/activate
python ./producer/producer.py --id $ID --d $DATA_PATH --s $DELAY_TIME --l $LOOP_TIMES --t $TOPIC --ip $IP
deactivate