from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-build-churn-target-v2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

MONTHLY_PATH = "data/gold/customer_monthly"
OUTPUT_PATH = "data/gold/churn_target_v2"

df = spark.read.parquet(MONTHLY_PATH)


# ---------------------------------------------------------
# 1. Ventanas temporales
# ---------------------------------------------------------

order_window = (
    Window
    .partitionBy("customer_id")
    .orderBy("month_start")
)

past_6m_window = (
    order_window
    .rowsBetween(-5, 0)
)

future_3m_window = (
    order_window
    .rowsBetween(1, 3)
)


# ---------------------------------------------------------
# 2. Métricas de comportamiento histórico
# ---------------------------------------------------------

target = (
    df
    .withColumn(
        "months_of_history",
        F.row_number().over(order_window)
    )
    .withColumn(
        "past_6m_revenue",
        F.sum("net_revenue").over(past_6m_window)
    )
    .withColumn(
        "past_6m_purchase_revenue",
        F.sum("purchase_revenue").over(past_6m_window)
    )
    .withColumn(
        "past_6m_purchases",
        F.sum("purchase_count").over(past_6m_window)
    )
    .withColumn(
        "active_months_last_6m",
        F.sum("is_active_month").over(past_6m_window)
    )
)


# ---------------------------------------------------------
# 3. Comportamiento futuro a 3 meses
# ---------------------------------------------------------

target = (
    target
    .withColumn(
        "future_3m_revenue",
        F.sum("net_revenue").over(future_3m_window)
    )
    .withColumn(
        "future_3m_purchases",
        F.sum("purchase_count").over(future_3m_window)
    )
    .withColumn(
        "future_active_months",
        F.sum("is_active_month").over(future_3m_window)
    )
)


# ---------------------------------------------------------
# 4. Último snapshot etiquetable
# ---------------------------------------------------------

max_data_date = (
    df
    .agg(F.max("observation_date"))
    .first()[0]
)

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
# 5. Filtrar clientes con comportamiento suficientemente estable
# ---------------------------------------------------------

target = target.filter(
    (F.col("months_of_history") >= 6) &
    (F.col("past_6m_revenue") > 0) &
    (F.col("active_months_last_6m") >= 3) &
    (F.col("days_since_last_purchase") <= 120) &
    (F.col("month_start") <= F.lit(max_target_month))
)


# ---------------------------------------------------------
# 6. Nivel esperado para próximos 3 meses
#
# Si en 6 meses factura 12.000 €,
# expectativa 3 meses ≈ 6.000 €
# ---------------------------------------------------------

target = (
    target
    .withColumn(
        "expected_3m_revenue",
        F.col("past_6m_revenue") / 2.0
    )
    .withColumn(
        "future_vs_expected_ratio",
        F.col("future_3m_revenue") /
        F.col("expected_3m_revenue")
    )
    .withColumn(
        "revenue_drop_pct",
        1 - F.col("future_vs_expected_ratio")
    )
)


# ---------------------------------------------------------
# 7. Target: caída futura >= 30%
# ---------------------------------------------------------

target = target.withColumn(
    "target_drop_30",
    (
        F.col("future_3m_revenue")
        <= F.col("expected_3m_revenue") * 0.70
    ).cast("int")
)


# ---------------------------------------------------------
# 8. Guardar
# ---------------------------------------------------------

target.write \
    .mode("overwrite") \
    .parquet(OUTPUT_PATH)


# ---------------------------------------------------------
# 9. Validaciones
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CHURN TARGET V2 CREADO")
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

print("\nPORCENTAJE DE RIESGO:")
target.select(
    (
        F.avg("target_drop_30") * 100
    ).alias("risk_rate_pct")
).show()

print("\nTARGET POR AÑO:")
target.groupBy(
    F.year("month_start").alias("year")
).agg(
    F.count("*").alias("observaciones"),
    F.sum("target_drop_30").alias("positivos"),
    (
        F.avg("target_drop_30") * 100
    ).alias("risk_rate_pct")
).orderBy("year").show()

print("\nEJEMPLOS DE CAÍDA:")
target.filter(
    F.col("target_drop_30") == 1
).select(
    "customer_id",
    "month_start",
    "past_6m_revenue",
    "active_months_last_6m",
    "expected_3m_revenue",
    "future_3m_revenue",
    "revenue_drop_pct",
    "past_6m_purchases",
    "future_3m_purchases",
    "days_since_last_purchase"
).orderBy(
    F.desc("past_6m_revenue")
).show(20, truncate=False)

spark.stop()