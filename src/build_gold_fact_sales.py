from pyspark.sql import SparkSession, functions as F


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-build-gold-fact-sales")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

SILVER_PATH = "data/silver/transactions"
GOLD_PATH = "data/gold/fact_sales"

silver = spark.read.parquet(SILVER_PATH)


# ---------------------------------------------------------
# 1. Solo FACTURAS
# ---------------------------------------------------------

invoice_lines = (
    silver
    .filter(F.col("operation_type") == "FACTURA")
    .filter(F.col("customer_id") != "000000")
)


# ---------------------------------------------------------
# 2. Una fila por factura
# ---------------------------------------------------------

fact_sales = (
    invoice_lines
    .groupBy(
        "document_id",
        "customer_id",
        "transaction_date"
    )
    .agg(
        F.sum("sales_amount").alias("invoice_amount"),
        F.sum("units").alias("total_units"),
        F.count("*").alias("number_of_lines"),
        F.countDistinct("product_id").alias("different_products")
    )
)


# ---------------------------------------------------------
# 3. Clasificar el tipo de evento económico
# ---------------------------------------------------------

fact_sales = (
    fact_sales
    .withColumn(
        "event_type",
        F.when(
            F.col("invoice_amount") > 0,
            F.lit("PURCHASE")
        )
        .when(
            F.col("invoice_amount") < 0,
            F.lit("CREDIT_NOTE")
        )
        .otherwise(F.lit("ZERO"))
    )
    .withColumn(
        "year",
        F.year("transaction_date")
    )
    .withColumn(
        "month",
        F.month("transaction_date")
    )
    .withColumn(
        "year_month",
        F.date_format("transaction_date", "yyyy-MM")
    )
)


# ---------------------------------------------------------
# 4. Orden lógico de columnas
# ---------------------------------------------------------

fact_sales = fact_sales.select(
    F.col("document_id").alias("invoice_id"),
    "customer_id",
    "transaction_date",
    "invoice_amount",
    "event_type",
    "total_units",
    "number_of_lines",
    "different_products",
    "year",
    "month",
    "year_month"
)


# ---------------------------------------------------------
# 5. Guardar Gold
# ---------------------------------------------------------

fact_sales.write \
    .mode("overwrite") \
    .parquet(GOLD_PATH)


# ---------------------------------------------------------
# 6. Validaciones
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("GOLD FACT_SALES CREADA")
print("=" * 70)

print(f"\nFilas: {fact_sales.count():,}")

print(
    f"Clientes: "
    f"{fact_sales.select('customer_id').distinct().count():,}"
)

print("\nEVENTOS:")
fact_sales.groupBy("event_type") \
    .count() \
    .orderBy("event_type") \
    .show()

print("\nREVENUE NETO:")
fact_sales.select(
    F.sum("invoice_amount").alias("net_revenue")
).show()

print("\nRESUMEN POR AÑO:")
fact_sales.groupBy("year") \
    .agg(
        F.count("*").alias("facturas"),
        F.countDistinct("customer_id").alias("clientes"),
        F.sum("invoice_amount").alias("revenue")
    ) \
    .orderBy("year") \
    .show()

print("\nPRIMERAS 20 FILAS:")
fact_sales.orderBy(
    "transaction_date",
    "invoice_id"
).show(20, truncate=False)

spark.stop()