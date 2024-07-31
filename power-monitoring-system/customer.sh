TOPIC="test"
OUTPUT="bi-stream"
IP="localhost:9092"

#source venv/bin/activate
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.1 ./customer/processor.py  --t $TOPIC --ip $IP --o $OUTPUT 
#deactivate