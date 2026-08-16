from pyspark.sql import SparkSession, functions as F


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-validate-silver")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

SILVER_PATH = "data/silver/transactions"

df = spark.read.parquet(SILVER_PATH)

print("\n" + "=" * 70)
print("VALIDACIÓN SILVER")
print("=" * 70)

# 1. Volumen general
print(f"\nFilas totales: {df.count():,}")
print(f"Documentos únicos: {df.select('document_id').distinct().count():,}")
print(f"Clientes únicos: {df.select('customer_id').distinct().count():,}")
print(f"Productos únicos: {df.select('product_id').distinct().count():,}")

# 2. Rango temporal
df.select(
    F.min("transaction_date").alias("fecha_min"),
    F.max("transaction_date").alias("fecha_max")
).show()

# 3. Nulos
print("\nNULOS:")
df.select([
    F.sum(F.col(c).isNull().cast("int")).alias(c)
    for c in [
        "transaction_date",
        "operation_type",
        "document_id",
        "customer_id",
        "product_id",
        "units",
        "sales_amount",
    ]
]).show(vertical=True)

# 4. Operaciones
print("\nOPERACIONES:")
df.groupBy("operation_type") \
    .agg(
        F.count("*").alias("lineas"),
        F.countDistinct("document_id").alias("documentos"),
        F.countDistinct("customer_id").alias("clientes")
    ) \
    .orderBy("operation_type") \
    .show()

# 5. Importes negativos, cero y positivos
print("\nDISTRIBUCIÓN DE IMPORTES:")
df.select(
    F.sum((F.col("sales_amount") < 0).cast("int")).alias("ventas_negativas"),
    F.sum((F.col("sales_amount") == 0).cast("int")).alias("ventas_cero"),
    F.sum((F.col("sales_amount") > 0).cast("int")).alias("ventas_positivas"),
).show()

# 6. Ejemplos negativos
print("\nEJEMPLOS DE IMPORTES NEGATIVOS:")
df.filter(
    F.col("sales_amount") < 0
).select(
    "transaction_date",
    "operation_type",
    "document_id",
    "customer_id",
    "product_id",
    "units",
    "sales_amount"
).show(20, truncate=False)

# 7. Posibles duplicados de negocio
business_columns = [
    "transaction_date",
    "operation_type",
    "document_id",
    "customer_id",
    "product_id",
    "units",
    "sales_amount",
]

duplicates = (
    df.groupBy(business_columns)
    .count()
    .filter(F.col("count") > 1)
)

print("\nGRUPOS DE POSIBLES DUPLICADOS:")
print(duplicates.count())

print("\nEJEMPLOS DE POSIBLES DUPLICADOS:")
duplicates.orderBy(F.desc("count")).show(20, truncate=False)

print("\nRESUMEN POR AÑO:")

df.groupBy("source_year") \
    .agg(
        F.count("*").alias("lineas"),
        F.countDistinct("document_id").alias("documentos"),
        F.countDistinct("customer_id").alias("clientes"),
        F.min("transaction_date").alias("fecha_min"),
        F.max("transaction_date").alias("fecha_max")
    ) \
    .orderBy("source_year") \
    .show(truncate=False)

spark.stop()