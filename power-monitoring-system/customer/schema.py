from pyspark.sql.types import *

c_schema = StructType() \
        .add("utc_timestamp", TimestampType()) \
        .add("grid_import", FloatType()) \
        .add("resident_id", IntegerType())
