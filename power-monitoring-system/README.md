# power-monitoring-system
Simple project with Kafka, Spark streaming, PowerBI

## Data
Detailed household load in minutely resolution in [here](https://data.open-power-system-data.org/household_data/)

We reduce some features, then schema like: 
| timestemp | pv/min |

## Requirements
	1. Kafka
	2. Hadoop + Spark
	3. PowerBI
	
## Workflow
	1. Client
	in client device, run file producer.sh or producer.py to load file csv and send msg to kafka pipeline.
	
	2. Server(Kafka + Spark)
	in server, run file processor.sh to submit job to Spark Job Manager.
	
	3. BI
	Finaly, we using PowerBI to visualize and monitoring the streamings. Spark will define new msg(json) to send PowerBI datahouse by API
	Run postman.py to get msg from kafka then send to PowerBI
