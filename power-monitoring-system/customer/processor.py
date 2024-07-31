import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

from schema import c_schema
from postman import send2API
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=1, help="iterval, time(s) to split streams")
    parser.add_argument("--t", type=str, help="topic for input")
    parser.add_argument("--o", type=str, help="topic for ouput")
    parser.add_argument("--ip", type=str, default="localhost:9092", help="ip server:port")
    return parser.parse_args()


class Processor():
    def __init__(self, args):
        self.topic = args.t
        self.server = args.ip
        self.topic_o = args.o
        self.global_total = 0

    def run(self):
        spark = SparkSession \
        .builder \
        .appName("power-monitoring") \
        .master("local[*]") \
        .getOrCreate()

        spark.sparkContext.setLogLevel("ERROR")
        df = spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.server) \
            .option("subscribe", self.topic) \
            .option("startingOffsets", "latest") \
            .load()

        print("Printing Schema of orders_df: ")
        df.printSchema()
        new_df = df.selectExpr("CAST(value AS STRING)")
        new_df = new_df.select(from_json(col('value'), c_schema).alias("consumption")) #{2015-01-12 00:24:00, NaN, 2}
        new_df = new_df.select("consumption.*")
        new_df.printSchema()

        #validate data: fill null,...


        #Transform
        new_df = new_df.groupBy("utc_timestamp") \
                .agg(count("resident_id"), 
                    sum("grid_import"), 
                    avg("grid_import")) \
                .select(col("utc_timestamp").alias("timestamp"), 
                    col("sum(grid_import)").alias("total_per_second"),
                    col("avg(grid_import)").alias("avg_per_resident"),
                    col("count(resident_id)").alias("activate_user"))                 
        
        new_df = new_df.selectExpr("to_json(struct(*)) AS value")


        #write to output stream, then consumer kafka will send to power BI      
        write_stream = new_df \
        .writeStream \
        .outputMode("update") \
        .format("kafka") \
        .option("kafka.bootstrap.servers", self.server) \
        .option("checkpointLocation", "./__checkpoint__") \
        .option("topic", self.topic_o) \
        .option("startingOffsets", "earliest") \
        .start()

        write_stream.awaitTermination()

if __name__ == "__main__":
    args = get_args()

    #init spark streaming processor
    processor = Processor(args)
    processor.run()