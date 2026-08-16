from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-build-customer-monthly")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

FACT_SALES_PATH = "data/gold/fact_sales"
OUTPUT_PATH = "data/gold/customer_monthly"

fact_sales = spark.read.parquet(FACT_SALES_PATH)


# ---------------------------------------------------------
# 1. Fecha máxima disponible
# ---------------------------------------------------------

max_date = (
    fact_sales
    .agg(F.max("transaction_date").alias("max_date"))
    .first()["max_date"]
)

max_month = max_date.replace(day=1)

print(f"\nÚltima fecha disponible: {max_date}")


# ---------------------------------------------------------
# 2. Solo clientes que han tenido al menos una compra real
# ---------------------------------------------------------

purchases = fact_sales.filter(
    F.col("event_type") == "PURCHASE"
)

customer_start = (
    purchases
    .groupBy("customer_id")
    .agg(
        F.trunc(
            F.min("transaction_date"),
            "month"
        ).alias("first_purchase_month")
    )
)


# ---------------------------------------------------------
# 3. Crear TODOS los meses para cada cliente
#    incluso cuando no compró
# ---------------------------------------------------------

calendar = (
    customer_start
    .withColumn(
        "month_start",
        F.explode(
            F.sequence(
                F.col("first_purchase_month"),
                F.lit(max_month),
                F.expr("interval 1 month")
            )
        )
    )
    .select(
        "customer_id",
        "month_start"
    )
)


# ---------------------------------------------------------
# 4. Métricas mensuales reales
# ---------------------------------------------------------

monthly_actual = (
    fact_sales
    .withColumn(
        "month_start",
        F.trunc("transaction_date", "month")
    )
    .groupBy(
        "customer_id",
        "month_start"
    )
    .agg(

        # Revenue neto: compras + abonos
        F.sum("invoice_amount")
        .alias("net_revenue"),

        # Revenue solo de compras positivas
        F.sum(
            F.when(
                F.col("event_type") == "PURCHASE",
                F.col("invoice_amount")
            ).otherwise(0)
        ).alias("purchase_revenue"),

        # Número de compras reales
        F.sum(
            (F.col("event_type") == "PURCHASE")
            .cast("int")
        ).alias("purchase_count"),

        # Número de abonos
        F.sum(
            (F.col("event_type") == "CREDIT_NOTE")
            .cast("int")
        ).alias("credit_note_count"),

        F.count("*")
        .alias("invoice_count"),

        # Última compra positiva del mes
        F.max(
            F.when(
                F.col("event_type") == "PURCHASE",
                F.col("transaction_date")
            )
        ).alias("last_purchase_in_month")
    )
)


# ---------------------------------------------------------
# 5. Unir calendario + actividad
# ---------------------------------------------------------

monthly = (
    calendar
    .join(
        monthly_actual,
        ["customer_id", "month_start"],
        "left"
    )
    .fillna(
        {
            "net_revenue": 0,
            "purchase_revenue": 0,
            "purchase_count": 0,
            "credit_note_count": 0,
            "invoice_count": 0,
        }
    )
)


# ---------------------------------------------------------
# 6. Recency: última compra conocida hasta ese mes
# ---------------------------------------------------------

window = (
    Window
    .partitionBy("customer_id")
    .orderBy("month_start")
    .rowsBetween(
        Window.unboundedPreceding,
        0
    )
)

monthly = (
    monthly
    .withColumn(
        "last_purchase_date",
        F.last(
            "last_purchase_in_month",
            ignorenulls=True
        ).over(window)
    )
    .withColumn(
    "observation_date",
    F.least(
        F.last_day("month_start"),
        F.lit(max_date)
        )
    )
    .withColumn(
    "days_since_last_purchase",
    F.datediff(
        F.col("observation_date"),
        F.col("last_purchase_date")
        )
    )
)


# ---------------------------------------------------------
# 7. Variables temporales
# ---------------------------------------------------------

monthly = (
    monthly
    .withColumn(
        "year",
        F.year("month_start")
    )
    .withColumn(
        "month",
        F.month("month_start")
    )
    .withColumn(
        "year_month",
        F.date_format("month_start", "yyyy-MM")
    )
    .withColumn(
        "is_active_month",
        (F.col("purchase_count") > 0).cast("int")
    )
)


# ---------------------------------------------------------
# 8. Guardar Gold
# ---------------------------------------------------------

monthly.write \
    .mode("overwrite") \
    .parquet(OUTPUT_PATH)


# ---------------------------------------------------------
# 9. Validaciones
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CUSTOMER MONTHLY CREADA")
print("=" * 70)

print(f"\nFilas: {monthly.count():,}")

print(
    f"Clientes: "
    f"{monthly.select('customer_id').distinct().count():,}"
)

print("\nRANGO:")
monthly.select(
    F.min("month_start").alias("min_month"),
    F.max("month_start").alias("max_month")
).show()

print("\nMESES CON / SIN COMPRA:")
monthly.groupBy("is_active_month") \
    .count() \
    .orderBy("is_active_month") \
    .show()

print("\nEJEMPLO DE HISTORIAL:")
example_customer = (
    purchases
    .groupBy("customer_id")
    .count()
    .orderBy(F.desc("count"))
    .first()["customer_id"]
)

print(f"Cliente ejemplo: {example_customer}")

monthly.filter(
    F.col("customer_id") == example_customer
).orderBy(
    "month_start"
).show(60, truncate=False)

spark.stop()