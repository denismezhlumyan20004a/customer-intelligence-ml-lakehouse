from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-build-churn-target")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

MONTHLY_PATH = "data/gold/customer_monthly"
OUTPUT_PATH = "data/gold/churn_target"

df = spark.read.parquet(MONTHLY_PATH)


# ---------------------------------------------------------
# 1. Ventanas temporales
# ---------------------------------------------------------

order_window = (
    Window
    .partitionBy("customer_id")
    .orderBy("month_start")
)

past_3m_window = (
    order_window
    .rowsBetween(-2, 0)
)

future_3m_window = (
    order_window
    .rowsBetween(1, 3)
)


# ---------------------------------------------------------
# 2. Crear métricas pasado / futuro
# ---------------------------------------------------------

target = (
    df
    .withColumn(
        "months_of_history",
        F.row_number().over(order_window)
    )
    .withColumn(
        "past_3m_revenue",
        F.sum("net_revenue").over(past_3m_window)
    )
    .withColumn(
        "future_3m_revenue",
        F.sum("net_revenue").over(future_3m_window)
    )
    .withColumn(
        "past_3m_purchases",
        F.sum("purchase_count").over(past_3m_window)
    )
    .withColumn(
        "future_3m_purchases",
        F.sum("purchase_count").over(future_3m_window)
    )
)


# ---------------------------------------------------------
# 3. Último mes que podemos etiquetar con 3 meses completos
# ---------------------------------------------------------

max_data_date = (
    df
    .agg(F.max("observation_date"))
    .first()[0]
)

# Agosto 2026 está incompleto.
# Si max_data_date = 2026-08-10:
# último mes completo = julio
# último snapshot con 3 meses futuros completos = abril
max_target_month = (
    spark
    .range(1)
    .select(
        F.add_months(
            F.trunc(F.lit(max_data_date), "month"),
            -4
        ).alias("max_target_month")
    )
    .first()["max_target_month"]
)

print(f"\nÚltimo mes etiquetable: {max_target_month}")


# ---------------------------------------------------------
# 4. Filtrar observaciones válidas
# ---------------------------------------------------------

target = target.filter(
    (F.col("months_of_history") >= 6) &
    (F.col("past_3m_revenue") > 0) &
    (F.col("days_since_last_purchase") <= 120) &
    (F.col("month_start") <= F.lit(max_target_month))
)


# ---------------------------------------------------------
# 5. Target: caída >= 30% en los próximos 3 meses
# ---------------------------------------------------------

target = (
    target
    .withColumn(
        "future_vs_past_ratio",
        F.col("future_3m_revenue") /
        F.col("past_3m_revenue")
    )
    .withColumn(
        "revenue_drop_pct",
        1 - F.col("future_vs_past_ratio")
    )
    .withColumn(
        "target_drop_30",
        (
            F.col("future_3m_revenue")
            <= F.col("past_3m_revenue") * 0.70
        ).cast("int")
    )
)


# ---------------------------------------------------------
# 6. Guardar
# ---------------------------------------------------------

target.write \
    .mode("overwrite") \
    .parquet(OUTPUT_PATH)


# ---------------------------------------------------------
# 7. Validaciones
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CHURN TARGET CREADO")
print("=" * 70)

print(f"\nObservaciones: {target.count():,}")

print(
    f"Clientes: "
    f"{target.select('customer_id').distinct().count():,}"
)

print("\nDISTRIBUCIÓN TARGET:")
target.groupBy("target_drop_30") \
    .count() \
    .orderBy("target_drop_30") \
    .show()

print("\nPORCENTAJE DE CHURN:")
target.select(
    (
        F.avg("target_drop_30") * 100
    ).alias("churn_rate_pct")
).show()

print("\nEJEMPLOS DE CAÍDA:")
target.filter(
    F.col("target_drop_30") == 1
).select(
    "customer_id",
    "month_start",
    "past_3m_revenue",
    "future_3m_revenue",
    "revenue_drop_pct",
    "past_3m_purchases",
    "future_3m_purchases",
    "days_since_last_purchase"
).orderBy(
    F.desc("past_3m_revenue")
).show(20, truncate=False)

spark.stop()