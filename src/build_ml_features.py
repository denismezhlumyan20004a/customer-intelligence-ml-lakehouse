from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-build-ml-features")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

MONTHLY_PATH = "data/gold/customer_monthly"
TARGET_PATH = "data/gold/churn_target_v3"
OUTPUT_PATH = "data/gold/ml_features"

monthly = spark.read.parquet(MONTHLY_PATH)

# Del target SOLO cogemos identificadores + label.
labels = (
    spark.read.parquet(TARGET_PATH)
    .select(
        "customer_id",
        "month_start",
        "target_drop_30"
    )
)


# ---------------------------------------------------------
# 1. Ventanas históricas
# ---------------------------------------------------------

w = (
    Window
    .partitionBy("customer_id")
    .orderBy("month_start")
)

w3 = w.rowsBetween(-2, 0)
w6 = w.rowsBetween(-5, 0)
w12 = w.rowsBetween(-11, 0)

# Para tendencia:
# primeros 3 meses de los últimos 6
w_prev3 = w.rowsBetween(-5, -3)

# últimos 3 meses
w_recent3 = w.rowsBetween(-2, 0)


# ---------------------------------------------------------
# 2. Features monetarias
# ---------------------------------------------------------

features = (
    monthly

    # Revenue del mes actual
    .withColumn(
        "revenue_1m",
        F.col("net_revenue")
    )

    .withColumn(
        "revenue_3m",
        F.sum("net_revenue").over(w3)
    )

    .withColumn(
        "revenue_6m",
        F.sum("net_revenue").over(w6)
    )

    .withColumn(
        "revenue_12m",
        F.sum("net_revenue").over(w12)
    )

    .withColumn(
        "avg_monthly_revenue_3m",
        F.avg("net_revenue").over(w3)
    )

    .withColumn(
        "avg_monthly_revenue_6m",
        F.avg("net_revenue").over(w6)
    )

    .withColumn(
        "avg_monthly_revenue_12m",
        F.avg("net_revenue").over(w12)
    )
)


# ---------------------------------------------------------
# 3. Frecuencia de compra
# ---------------------------------------------------------

features = (
    features
    .withColumn(
        "purchases_3m",
        F.sum("purchase_count").over(w3)
    )
    .withColumn(
        "purchases_6m",
        F.sum("purchase_count").over(w6)
    )
    .withColumn(
        "purchases_12m",
        F.sum("purchase_count").over(w12)
    )
    .withColumn(
        "active_months_3m",
        F.sum("is_active_month").over(w3)
    )
    .withColumn(
        "active_months_6m",
        F.sum("is_active_month").over(w6)
    )
    .withColumn(
        "active_months_12m",
        F.sum("is_active_month").over(w12)
    )
)


# ---------------------------------------------------------
# 4. Recency
# ---------------------------------------------------------

features = (
    features
    .withColumn(
        "recency_days",
        F.col("days_since_last_purchase")
    )
)


# ---------------------------------------------------------
# 5. Tendencia reciente
#
# Comparamos los últimos 3 meses contra los
# 3 meses inmediatamente anteriores.
# Solo usa información disponible hasta t.
# ---------------------------------------------------------

features = (
    features
    .withColumn(
        "previous_3m_revenue",
        F.sum("net_revenue").over(w_prev3)
    )
    .withColumn(
        "recent_3m_revenue_feature",
        F.sum("net_revenue").over(w_recent3)
    )
    .withColumn(
        "revenue_change_3m",
        F.when(
            F.col("previous_3m_revenue") > 0,
            (
                F.col("recent_3m_revenue_feature")
                - F.col("previous_3m_revenue")
            ) / F.col("previous_3m_revenue")
        ).otherwise(F.lit(0.0))
    )
)


# ---------------------------------------------------------
# 6. Volatilidad
#
# Clientes muy irregulares pueden parecer "churn"
# cuando simplemente tienen compras variables.
# ---------------------------------------------------------

features = (
    features
    .withColumn(
        "revenue_std_6m",
        F.stddev_pop("net_revenue").over(w6)
    )
    .withColumn(
        "revenue_std_12m",
        F.stddev_pop("net_revenue").over(w12)
    )
    .withColumn(
        "revenue_cv_6m",
        F.when(
            F.avg("net_revenue").over(w6) > 0,
            F.stddev_pop("net_revenue").over(w6)
            / F.avg("net_revenue").over(w6)
        ).otherwise(F.lit(0.0))
    )
)


# ---------------------------------------------------------
# 7. Intensidad / ticket medio
# ---------------------------------------------------------

features = (
    features
    .withColumn(
        "avg_ticket_3m",
        F.when(
            F.col("purchases_3m") > 0,
            F.col("revenue_3m") /
            F.col("purchases_3m")
        ).otherwise(F.lit(0.0))
    )
    .withColumn(
        "avg_ticket_6m",
        F.when(
            F.col("purchases_6m") > 0,
            F.col("revenue_6m") /
            F.col("purchases_6m")
        ).otherwise(F.lit(0.0))
    )
)


# ---------------------------------------------------------
# 8. Abonos / créditos
# ---------------------------------------------------------

features = (
    features
    .withColumn(
        "credit_notes_3m",
        F.sum("credit_note_count").over(w3)
    )
    .withColumn(
        "credit_notes_6m",
        F.sum("credit_note_count").over(w6)
    )
)


# ---------------------------------------------------------
# 9. Antigüedad del cliente
# ---------------------------------------------------------

features = (
    features
    .withColumn(
        "customer_age_months",
        F.row_number().over(w)
    )
)


# ---------------------------------------------------------
# 10. Unir SOLO con snapshots válidos del target V3
# ---------------------------------------------------------

ml = (
    features
    .join(
        labels,
        ["customer_id", "month_start"],
        "inner"
    )
)


# ---------------------------------------------------------
# 11. Selección final
# ---------------------------------------------------------

ml = ml.select(
    "customer_id",
    "month_start",

    "revenue_1m",
    "revenue_3m",
    "revenue_6m",
    "revenue_12m",

    "avg_monthly_revenue_3m",
    "avg_monthly_revenue_6m",
    "avg_monthly_revenue_12m",

    "purchases_3m",
    "purchases_6m",
    "purchases_12m",

    "active_months_3m",
    "active_months_6m",
    "active_months_12m",

    "recency_days",

    "previous_3m_revenue",
    "recent_3m_revenue_feature",
    "revenue_change_3m",

    "revenue_std_6m",
    "revenue_std_12m",
    "revenue_cv_6m",

    "avg_ticket_3m",
    "avg_ticket_6m",

    "credit_notes_3m",
    "credit_notes_6m",

    "customer_age_months",

    "target_drop_30"
)


# ---------------------------------------------------------
# 12. Guardar
# ---------------------------------------------------------

ml.write \
    .mode("overwrite") \
    .parquet(OUTPUT_PATH)


# ---------------------------------------------------------
# 13. Validaciones
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ML FEATURES CREADAS")
print("=" * 70)

print(f"\nFilas: {ml.count():,}")

print(
    f"Clientes: "
    f"{ml.select('customer_id').distinct().count():,}"
)

print(f"Columnas: {len(ml.columns)}")

print("\nTARGET:")
ml.groupBy("target_drop_30") \
    .count() \
    .orderBy("target_drop_30") \
    .show()

print("\nRANGO TEMPORAL:")
ml.select(
    F.min("month_start").alias("min_month"),
    F.max("month_start").alias("max_month")
).show()

print("\nNULOS EN FEATURES:")

feature_columns = [
    c for c in ml.columns
    if c not in [
        "customer_id",
        "month_start",
        "target_drop_30"
    ]
]

ml.select(
    [
        F.sum(
            F.col(c).isNull().cast("int")
        ).alias(c)
        for c in feature_columns
    ]
).show(vertical=True)

print("\nEJEMPLO:")
ml.orderBy(
    F.desc("month_start")
).show(10, truncate=False)

spark.stop()