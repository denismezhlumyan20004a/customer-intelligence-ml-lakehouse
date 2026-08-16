from pyspark.sql import SparkSession, functions as F


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-validate-invoices")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet("data/silver/transactions")

# Para churn/CLV nos centramos en FACTURA
invoices_lines = df.filter(
    F.col("operation_type") == "FACTURA"
)

# Una fila por factura
invoices = (
    invoices_lines
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

print("\n" + "=" * 70)
print("VALIDACIÓN DE FACTURAS")
print("=" * 70)

print(f"\nFacturas: {invoices.count():,}")
print(
    f"Clientes con factura: "
    f"{invoices.select('customer_id').distinct().count():,}"
)

print("\nIMPORTE TOTAL FACTURADO:")
invoices.select(
    F.sum("invoice_amount").alias("total_revenue")
).show()

print("\nFACTURAS POSITIVAS / CERO / NEGATIVAS:")
invoices.select(
    F.sum(
        (F.col("invoice_amount") > 0).cast("int")
    ).alias("positivas"),

    F.sum(
        (F.col("invoice_amount") == 0).cast("int")
    ).alias("cero"),

    F.sum(
        (F.col("invoice_amount") < 0).cast("int")
    ).alias("negativas")
).show()

print("\nEJEMPLOS DE FACTURAS NEGATIVAS:")
invoices.filter(
    F.col("invoice_amount") < 0
).orderBy(
    "invoice_amount"
).show(20, truncate=False)

print("\nCLIENTE 000000:")
invoices.filter(
    F.col("customer_id") == "000000"
).agg(
    F.count("*").alias("facturas"),
    F.sum("invoice_amount").alias("revenue")
).show()

print("\nFACTURAS CON MÁS DE 1 CLIENTE:")
(
    invoices_lines
    .groupBy("document_id")
    .agg(
        F.countDistinct("customer_id").alias("clientes")
    )
    .filter(F.col("clientes") > 1)
    .show()
)

spark.stop()