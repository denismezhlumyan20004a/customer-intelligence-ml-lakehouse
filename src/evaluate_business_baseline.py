from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-evaluate-business-baseline")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

MONTHLY_PATH = "data/gold/customer_monthly"

df = spark.read.parquet(MONTHLY_PATH)


# ---------------------------------------------------------
# 1. Ventanas temporales
# ---------------------------------------------------------

order_window = (
    Window
    .partitionBy("customer_id")
    .orderBy("month_start")
)

# t-5, t-4, t-3
baseline_3m_window = (
    order_window
    .rowsBetween(-5, -3)
)

# t-2, t-1, t
recent_3m_window = (
    order_window
    .rowsBetween(-2, 0)
)

# t+1, t+2, t+3
future_3m_window = (
    order_window
    .rowsBetween(1, 3)
)

past_6m_window = (
    order_window
    .rowsBetween(-5, 0)
)


# ---------------------------------------------------------
# 2. Construir dataset de evaluación
# ---------------------------------------------------------

evaluation = (
    df
    .withColumn(
        "months_of_history",
        F.row_number().over(order_window)
    )
    .withColumn(
        "baseline_3m_revenue",
        F.sum("net_revenue").over(baseline_3m_window)
    )
    .withColumn(
        "recent_3m_revenue",
        F.sum("net_revenue").over(recent_3m_window)
    )
    .withColumn(
        "future_3m_revenue",
        F.sum("net_revenue").over(future_3m_window)
    )
    .withColumn(
        "active_months_last_6m",
        F.sum("is_active_month").over(past_6m_window)
    )
)


# ---------------------------------------------------------
# 3. Último mes con futuro completo
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

print(f"\nÚltimo mes evaluable: {max_target_month}")


# ---------------------------------------------------------
# 4. Universo comparable
# ---------------------------------------------------------

evaluation = evaluation.filter(
    (F.col("months_of_history") >= 6) &
    (F.col("baseline_3m_revenue") > 0) &
    (F.col("active_months_last_6m") >= 3) &
    (F.col("month_start") <= F.lit(max_target_month))
)


# ---------------------------------------------------------
# 5. Regla actual del negocio
#
# Ya existe caída >=30% en el momento t
# ---------------------------------------------------------

evaluation = (
    evaluation
    .withColumn(
        "current_drop_pct",
        1 -
        (
            F.col("recent_3m_revenue") /
            F.col("baseline_3m_revenue")
        )
    )
    .withColumn(
        "business_rule_30",
        (
            F.col("recent_3m_revenue")
            <= F.col("baseline_3m_revenue") * 0.70
        ).cast("int")
    )
)


# ---------------------------------------------------------
# 6. Ground truth:
# ¿Los próximos 3 meses estarán >=30% por debajo
# del baseline histórico?
# ---------------------------------------------------------

evaluation = (
    evaluation
    .withColumn(
        "future_drop_pct",
        1 -
        (
            F.col("future_3m_revenue") /
            F.col("baseline_3m_revenue")
        )
    )
    .withColumn(
        "future_drop_30",
        (
            F.col("future_3m_revenue")
            <= F.col("baseline_3m_revenue") * 0.70
        ).cast("int")
    )
)


# ---------------------------------------------------------
# 7. Matriz de confusión
# ---------------------------------------------------------

metrics = (
    evaluation
    .agg(
        F.count("*").alias("observations"),

        F.sum("future_drop_30")
        .alias("future_positive"),

        F.sum("business_rule_30")
        .alias("rule_flags"),

        F.sum(
            F.when(
                (F.col("business_rule_30") == 1) &
                (F.col("future_drop_30") == 1),
                1
            ).otherwise(0)
        ).alias("tp"),

        F.sum(
            F.when(
                (F.col("business_rule_30") == 1) &
                (F.col("future_drop_30") == 0),
                1
            ).otherwise(0)
        ).alias("fp"),

        F.sum(
            F.when(
                (F.col("business_rule_30") == 0) &
                (F.col("future_drop_30") == 1),
                1
            ).otherwise(0)
        ).alias("fn"),

        F.sum(
            F.when(
                (F.col("business_rule_30") == 0) &
                (F.col("future_drop_30") == 0),
                1
            ).otherwise(0)
        ).alias("tn")
    )
    .first()
)


tp = metrics["tp"]
fp = metrics["fp"]
fn = metrics["fn"]
tn = metrics["tn"]

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0 else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0 else 0
)

early_warning_share = (
    fn / (tp + fn)
    if (tp + fn) > 0 else 0
)


# ---------------------------------------------------------
# 8. Resultados
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("BASELINE DE NEGOCIO - REGLA 30%")
print("=" * 70)

print(f"\nObservaciones: {metrics['observations']:,}")
print(f"Caídas futuras reales: {metrics['future_positive']:,}")
print(f"Alertas generadas por regla: {metrics['rule_flags']:,}")

print("\nMATRIZ DE CONFUSIÓN:")
print(f"TP - caída ya visible y continúa: {tp:,}")
print(f"FP - alerta pero después mejora: {fp:,}")
print(f"FN - caída futura aún NO visible: {fn:,}")
print(f"TN - estable correctamente: {tn:,}")

print("\nMÉTRICAS:")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")

print(
    "\nPorcentaje de caídas futuras que todavía "
    "NO eran visibles con la regla del 30%:"
)
print(f"{early_warning_share * 100:.2f}%")


# ---------------------------------------------------------
# 9. Categorías de negocio
# ---------------------------------------------------------

evaluation = evaluation.withColumn(
    "business_situation",
    F.when(
        (F.col("business_rule_30") == 1) &
        (F.col("future_drop_30") == 1),
        "ALREADY_VISIBLE"
    )
    .when(
        (F.col("business_rule_30") == 0) &
        (F.col("future_drop_30") == 1),
        "EARLY_WARNING_OPPORTUNITY"
    )
    .when(
        (F.col("business_rule_30") == 1) &
        (F.col("future_drop_30") == 0),
        "FALSE_ALARM"
    )
    .otherwise("STABLE")
)

print("\nSITUACIONES DE NEGOCIO:")
evaluation.groupBy(
    "business_situation"
).count().orderBy(
    F.desc("count")
).show(truncate=False)


print("\nEJEMPLOS DE OPORTUNIDAD DE EARLY WARNING:")

evaluation.filter(
    F.col("business_situation") ==
    "EARLY_WARNING_OPPORTUNITY"
).select(
    "customer_id",
    "month_start",
    "baseline_3m_revenue",
    "recent_3m_revenue",
    "current_drop_pct",
    "future_3m_revenue",
    "future_drop_pct",
    "days_since_last_purchase"
).orderBy(
    F.desc("baseline_3m_revenue")
).show(20, truncate=False)


spark.stop()