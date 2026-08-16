from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-bronze-to-silver")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

DATA_PATH = "data/raw/aqua/2022/*.csv"

df = (
    spark.read
    .option("header", True)
    .option("sep", ";")
    .option("encoding", "UTF-8")
    .csv(DATA_PATH)
    .withColumn("source_file", input_file_name())
)

print("\nSCHEMA")
df.printSchema()

print(f"\nTOTAL FILAS: {df.count():,}")

print("\nPRIMERAS 20 FILAS")
df.show(20, truncate=False)

spark.stop()