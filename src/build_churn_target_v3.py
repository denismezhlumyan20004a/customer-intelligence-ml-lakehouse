from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-build-churn-target-v3")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

MONTHLY_PATH = "data/gold/customer_monthly"
OUTPUT_PATH = "data/gold/churn_target_v3"

df = spark.read.parquet(MONTHLY_PATH)


# ---------------------------------------------------------
# 1. Ventana temporal por cliente
# ---------------------------------------------------------

order_window = (
    Window
    .partitionBy("customer_id")
    .orderBy("month_start")
)


# Meses t-5, t-4, t-3
baseline_3m_window = (
    order_window
    .rowsBetween(-5, -3)
)


# Meses t-2, t-1, t
recent_3m_window = (
    order_window
    .rowsBetween(-2, 0)
)


# Meses t+1, t+2, t+3
future_3m_window = (
    order_window
    .rowsBetween(1, 3)
)


# Últimos 6 meses para medir estabilidad
past_6m_window = (
    order_window
    .rowsBetween(-5, 0)
)


# ---------------------------------------------------------
# 2. Construir comportamiento pasado, actual y futuro
# ---------------------------------------------------------

target = (
    df
    .withColumn(
        "months_of_history",
        F.row_number().over(order_window)
    )

    # Baseline: 3 meses anteriores al periodo reciente
    .withColumn(
        "baseline_3m_revenue",
        F.sum("net_revenue").over(baseline_3m_window)
    )

    .withColumn(
        "baseline_3m_purchases",
        F.sum("purchase_count").over(baseline_3m_window)
    )

    # Comportamiento reciente disponible en t
    .withColumn(
        "recent_3m_revenue",
        F.sum("net_revenue").over(recent_3m_window)
    )

    .withColumn(
        "recent_3m_purchases",
        F.sum("purchase_count").over(recent_3m_window)
    )

    # Comportamiento futuro: SOLO para construir la etiqueta
    .withColumn(
        "future_3m_revenue",
        F.sum("net_revenue").over(future_3m_window)
    )

    .withColumn(
        "future_3m_purchases",
        F.sum("purchase_count").over(future_3m_window)
    )

    # Estabilidad del cliente
    .withColumn(
        "active_months_last_6m",
        F.sum("is_active_month").over(past_6m_window)
    )
)


# ---------------------------------------------------------
# 3. Último snapshot con 3 meses futuros completos
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
# 4. Calcular caída que YA es visible en el snapshot
# ---------------------------------------------------------

target = (
    target
    .withColumn(
        "recent_vs_baseline_ratio",
        F.col("recent_3m_revenue") /
        F.col("baseline_3m_revenue")
    )
    .withColumn(
        "current_drop_pct",
        1 - F.col("recent_vs_baseline_ratio")
    )
)


# ---------------------------------------------------------
# 5. Filtrar snapshots elegibles para EARLY WARNING
#
# - mínimo 6 meses de historia
# - baseline económico real
# - actividad suficiente
# - no estar ya caído >= 30%
# - tener 3 meses futuros completos
# ---------------------------------------------------------

target = target.filter(
    (F.col("months_of_history") >= 6) &
    (F.col("baseline_3m_revenue") > 0) &
    (F.col("active_months_last_6m") >= 3) &
    (F.col("recent_3m_revenue") > 0) &
    (
        F.col("recent_3m_revenue")
        > F.col("baseline_3m_revenue") * 0.70
    ) &
    (F.col("month_start") <= F.lit(max_target_month))
)


# ---------------------------------------------------------
# 6. Caída FUTURA respecto al baseline
# ---------------------------------------------------------

target = (
    target
    .withColumn(
        "future_vs_baseline_ratio",
        F.col("future_3m_revenue") /
        F.col("baseline_3m_revenue")
    )
    .withColumn(
        "future_drop_pct",
        1 - F.col("future_vs_baseline_ratio")
    )
)


# ---------------------------------------------------------
# 7. Target de early warning
#
# Hoy NO ha caído >=30%.
# ¿En los siguientes 3 meses caerá >=30%?
# ---------------------------------------------------------

target = target.withColumn(
    "target_drop_30",
    (
        F.col("future_3m_revenue")
        <= F.col("baseline_3m_revenue") * 0.70
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
print("CHURN TARGET V3 - EARLY WARNING CREADO")
print("=" * 70)

print(f"\nObservaciones elegibles: {target.count():,}")

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


print("\nCOMPROBACIÓN: CAÍDA ACTUAL DE CLIENTES ELEGIBLES:")
target.select(
    F.min("current_drop_pct").alias("min_current_drop"),
    F.max("current_drop_pct").alias("max_current_drop"),
    F.avg("current_drop_pct").alias("avg_current_drop")
).show()


print("\nEJEMPLOS DE EARLY WARNING:")
target.filter(
    F.col("target_drop_30") == 1
).select(
    "customer_id",
    "month_start",
    "baseline_3m_revenue",
    "recent_3m_revenue",
    "current_drop_pct",
    "future_3m_revenue",
    "future_drop_pct",
    "baseline_3m_purchases",
    "recent_3m_purchases",
    "future_3m_purchases",
    "days_since_last_purchase"
).orderBy(
    F.desc("baseline_3m_revenue")
).show(20, truncate=False)


spark.stop()