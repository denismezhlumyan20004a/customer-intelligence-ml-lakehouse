from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array


# =========================================================
# 0. SPARK
# =========================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-score-current-customers")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. PATHS
# =========================================================

SCORING_FEATURES_PATH = (
    "data/gold/current_scoring_features"
)

MODEL_PATH = (
    "models/random_forest_production"
)

OUTPUT_SCORES_PATH = (
    "data/gold/current_customer_scores"
)

OUTPUT_TOP25_PATH = (
    "data/gold/current_top25_sales"
)

OUTPUT_TOP30_PATH = (
    "data/gold/current_top30_sales"
)


# =========================================================
# 2. FEATURES DEL MODELO
# =========================================================

BASE_FEATURE_COLUMNS = [
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
]


NEW_FEATURE_COLUMNS = [
    "expected_days_between_purchases_12m",

    "recency_ratio_12m",
    "recency_ratio_6m",

    "active_rate_3m",
    "active_rate_6m",
    "active_rate_12m",

    "purchases_per_active_month_12m",
    "revenue_per_active_month_12m",

    "revenue_momentum_1m_vs_6m",
    "revenue_momentum_3m_vs_12m",

    "purchase_momentum_3m_vs_12m",

    "avg_ticket_change_3m_vs_6m",

    "credit_note_rate_6m",

    "revenue_slope_6m",
    "purchase_slope_6m",

    "revenue_trend_6m_normalized",
    "purchase_trend_6m_normalized",

    "inactivity_streak_months",
]


FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    +
    NEW_FEATURE_COLUMNS
)


# =========================================================
# 3. CARGAR CLIENTES ACTUALES
# =========================================================

scoring = spark.read.parquet(
    SCORING_FEATURES_PATH
)


print("\n" + "=" * 70)
print("CURRENT CUSTOMER SCORING")
print("=" * 70)


scoring_count = scoring.count()


print(
    f"\nClientes elegibles recibidos: "
    f"{scoring_count:,}"
)


# =========================================================
# 4. VALIDACIÓN ESPERADA
#
# El paso anterior produjo 99 clientes elegibles.
# =========================================================

EXPECTED_CUSTOMERS = 99


if scoring_count != EXPECTED_CUSTOMERS:

    print(
        f"\nWARNING: esperábamos "
        f"{EXPECTED_CUSTOMERS} clientes, "
        f"pero encontramos {scoring_count}."
    )


# =========================================================
# 5. CAST NUMÉRICO
# =========================================================

for column_name in FEATURE_COLUMNS:

    scoring = scoring.withColumn(
        column_name,
        F.col(column_name).cast("double")
    )


# =========================================================
# 6. CARGAR MODELO PRODUCTIVO
# =========================================================

print(
    "\nCargando Random Forest "
    "production..."
)


model = PipelineModel.load(
    MODEL_PATH
)


# =========================================================
# 7. SCORE
# =========================================================

print(
    "Calculando probabilidades "
    "de caída futura..."
)


predictions = (
    model
    .transform(scoring)

    .withColumn(
        "risk_probability",

        vector_to_array(
            "probability"
        )[1]
    )

    .cache()
)


predictions.count()


# =========================================================
# 8. RANKING POR RIESGO
#
# ESTE es el ranking cuya lógica hemos validado
# históricamente.
#
# Rank 1 = mayor probabilidad de caída.
# =========================================================

risk_window = (
    Window
    .orderBy(
        F.desc("risk_probability"),
        F.asc("customer_id")
    )
)


predictions = predictions.withColumn(
    "risk_rank",

    F.row_number().over(
        risk_window
    )
)


# =========================================================
# 9. RISK × VALUE SCORE
#
# No interpretarlo literalmente como expected loss.
#
# Nuestro modelo predice probabilidad de EVENTO:
# caída >=30% en los próximos 3 meses.
#
# revenue_12m sirve como proxy de valor económico.
#
# Por tanto:
#
# risk_probability × revenue_12m
#
# = score de priorización riesgo-valor.
#
# Sirve para ordenar comercialmente los clientes
# dentro del conjunto de riesgo seleccionado.
# =========================================================

predictions = predictions.withColumn(
    "risk_value_score",

    F.col("risk_probability")
    *
    F.greatest(
        F.col("revenue_12m"),
        F.lit(0.0)
    )
)


# =========================================================
# 10. VALUE RANK
# =========================================================

value_window = (
    Window
    .orderBy(
        F.desc("revenue_12m"),
        F.asc("customer_id")
    )
)


predictions = predictions.withColumn(
    "value_rank",

    F.row_number().over(
        value_window
    )
)


# =========================================================
# 11. RISK-VALUE RANK
#
# Importante:
#
# Este ranking NO sustituye todavía al risk_rank
# validado.
#
# Lo utilizaremos para priorizar visitas dentro
# del Top 25 / Top 30.
# =========================================================

risk_value_window = (
    Window
    .orderBy(
        F.desc("risk_value_score"),
        F.desc("risk_probability"),
        F.asc("customer_id")
    )
)


predictions = predictions.withColumn(
    "risk_value_rank",

    F.row_number().over(
        risk_value_window
    )
)


# =========================================================
# 12. VALUE PERCENTILE
# =========================================================

predictions = predictions.withColumn(
    "value_percentile",

    F.percent_rank().over(
        Window.orderBy(
            F.col("revenue_12m")
        )
    )
)


# =========================================================
# 13. VALUE TIER
#
# Segmentación sencilla para lectura comercial.
# =========================================================

predictions = predictions.withColumn(
    "value_tier",

    F.when(
        F.col("value_percentile") >= 0.80,
        F.lit("VERY_HIGH")
    )

    .when(
        F.col("value_percentile") >= 0.60,
        F.lit("HIGH")
    )

    .when(
        F.col("value_percentile") >= 0.40,
        F.lit("MEDIUM")
    )

    .otherwise(
        F.lit("LOW")
    )
)


# =========================================================
# 14. RISK BAND
#
# No sustituye al ranking.
# Es solo una etiqueta explicativa.
# =========================================================

predictions = predictions.withColumn(
    "risk_band",

    F.when(
        F.col("risk_rank") <= 10,
        F.lit("VERY_HIGH")
    )

    .when(
        F.col("risk_rank") <= 25,
        F.lit("HIGH")
    )

    .when(
        F.col("risk_rank") <= 50,
        F.lit("MEDIUM")
    )

    .otherwise(
        F.lit("LOW")
    )
)


# =========================================================
# 15. SELECCIÓN OPERATIVA
# =========================================================

predictions = (
    predictions

    .withColumn(
        "selected_top25",

        F.when(
            F.col("risk_rank") <= 25,
            1
        ).otherwise(0)
    )

    .withColumn(
        "selected_top30",

        F.when(
            F.col("risk_rank") <= 30,
            1
        ).otherwise(0)
    )
)


# =========================================================
# 16. PRIORIDAD DE VISITA DENTRO DEL TOP 25
#
# El conjunto Top25 sigue determinado por risk_rank.
#
# Dentro de esos 25, ordenamos por risk_value_score
# para dar prioridad a cuentas económicamente mayores.
# =========================================================

top25_priority_window = (
    Window
    .orderBy(
        F.desc("risk_value_score"),
        F.desc("risk_probability"),
        F.asc("customer_id")
    )
)


top25 = (
    predictions

    .filter(
        F.col("risk_rank") <= 25
    )

    .withColumn(
        "sales_visit_priority",

        F.row_number().over(
            top25_priority_window
        )
    )
)


# =========================================================
# 17. TOP 30 AMPLIADO
# =========================================================

top30_priority_window = (
    Window
    .orderBy(
        F.desc("risk_value_score"),
        F.desc("risk_probability"),
        F.asc("customer_id")
    )
)


top30 = (
    predictions

    .filter(
        F.col("risk_rank") <= 30
    )

    .withColumn(
        "sales_visit_priority",

        F.row_number().over(
            top30_priority_window
        )
    )
)


# =========================================================
# 18. OUTPUT COLUMNS
# =========================================================

OUTPUT_COLUMNS = [

    "customer_id",
    "month_start",

    # Ranking validado
    "risk_rank",
    "risk_probability",
    "risk_band",

    # Valor
    "revenue_12m",
    "revenue_6m",
    "revenue_3m",
    "value_rank",
    "value_tier",

    # Priorización comercial
    "risk_value_score",
    "risk_value_rank",

    # Situación actual
    "current_drop_pct",
    "revenue_change_3m",

    "purchases_12m",
    "active_months_12m",

    "recency_days",
    "recency_ratio_12m",

    "revenue_trend_6m_normalized",

    # Flags
    "selected_top25",
    "selected_top30",
]


all_scores = predictions.select(
    *OUTPUT_COLUMNS
)


# =========================================================
# 19. RESULTADOS GENERALES
# =========================================================

print("\n" + "=" * 70)
print("SCORING SUMMARY")
print("=" * 70)


all_scores.select(

    F.count("*")
    .alias("customers"),

    F.min(
        "risk_probability"
    )
    .alias("min_risk"),

    F.avg(
        "risk_probability"
    )
    .alias("avg_risk"),

    F.max(
        "risk_probability"
    )
    .alias("max_risk"),

    F.sum(
        "revenue_12m"
    )
    .alias("total_revenue_12m"),

).show(
    truncate=False
)


# =========================================================
# 20. TOP 25 POR RIESGO
#
# Esta es la selección validada.
# =========================================================

print("\n" + "=" * 70)
print("TOP 25 - VALIDATED RISK RANKING")
print("=" * 70)


(
    all_scores

    .filter(
        F.col("risk_rank") <= 25
    )

    .select(
        "risk_rank",
        "customer_id",
        "risk_probability",

        "revenue_12m",
        "value_tier",

        "risk_value_score",

        "current_drop_pct",

        "recency_days",
        "recency_ratio_12m",

        "revenue_trend_6m_normalized",
    )

    .orderBy(
        "risk_rank"
    )

    .show(
        25,
        truncate=False
    )
)


# =========================================================
# 21. TOP 25 - ORDEN DE VISITA COMERCIAL
#
# MISMO conjunto de 25 clientes.
#
# Cambia únicamente el orden interno para favorecer
# cuentas económicamente relevantes.
# =========================================================

print("\n" + "=" * 70)
print("TOP 25 - SALES VISIT PRIORITY")
print("=" * 70)


(
    top25

    .select(
        "sales_visit_priority",

        "risk_rank",
        "customer_id",

        "risk_probability",

        "revenue_12m",
        "value_tier",

        "risk_value_score",

        "current_drop_pct",

        "recency_days",
        "recency_ratio_12m",

        "revenue_trend_6m_normalized",
    )

    .orderBy(
        "sales_visit_priority"
    )

    .show(
        25,
        truncate=False
    )
)


# =========================================================
# 22. TOP 30 AMPLIADO
# =========================================================

print("\n" + "=" * 70)
print("TOP 30 - EXTENDED SALES LIST")
print("=" * 70)


(
    top30

    .select(
        "sales_visit_priority",

        "risk_rank",
        "customer_id",

        "risk_probability",

        "revenue_12m",
        "value_tier",

        "risk_value_score",

        "current_drop_pct",
    )

    .orderBy(
        "sales_visit_priority"
    )

    .show(
        30,
        truncate=False
    )
)


# =========================================================
# 23. RESUMEN ECONÓMICO TOP 25
# =========================================================

print("\n" + "=" * 70)
print("TOP 25 - ECONOMIC SUMMARY")
print("=" * 70)


top25.select(

    F.count("*")
    .alias("customers"),

    F.sum(
        "revenue_12m"
    )
    .alias(
        "revenue_12m_top25"
    ),

    F.avg(
        "risk_probability"
    )
    .alias(
        "avg_risk_probability"
    ),

    F.sum(
        "risk_value_score"
    )
    .alias(
        "aggregate_risk_value_score"
    ),

).show(
    truncate=False
)


# =========================================================
# 24. RESUMEN ECONÓMICO TOP 30
# =========================================================

print("\n" + "=" * 70)
print("TOP 30 - ECONOMIC SUMMARY")
print("=" * 70)


top30.select(

    F.count("*")
    .alias("customers"),

    F.sum(
        "revenue_12m"
    )
    .alias(
        "revenue_12m_top30"
    ),

    F.avg(
        "risk_probability"
    )
    .alias(
        "avg_risk_probability"
    ),

    F.sum(
        "risk_value_score"
    )
    .alias(
        "aggregate_risk_value_score"
    ),

).show(
    truncate=False
)


# =========================================================
# 25. GUARDAR TODOS LOS SCORES
# =========================================================

(
    all_scores

    .write

    .mode("overwrite")

    .parquet(
        OUTPUT_SCORES_PATH
    )
)


# =========================================================
# 26. GUARDAR TOP 25
# =========================================================

top25_output = (
    top25

    .select(
        "sales_visit_priority",
        *OUTPUT_COLUMNS
    )

    .orderBy(
        "sales_visit_priority"
    )
)


(
    top25_output

    .write

    .mode("overwrite")

    .parquet(
        OUTPUT_TOP25_PATH
    )
)


# =========================================================
# 27. GUARDAR TOP 30
# =========================================================

top30_output = (
    top30

    .select(
        "sales_visit_priority",
        *OUTPUT_COLUMNS
    )

    .orderBy(
        "sales_visit_priority"
    )
)


(
    top30_output

    .write

    .mode("overwrite")

    .parquet(
        OUTPUT_TOP30_PATH
    )
)


# =========================================================
# 28. RESULTADO FINAL
# =========================================================

print("\n" + "=" * 70)
print("CURRENT SCORING COMPLETED")
print("=" * 70)


print(
    f"\nTodos los scores:"
)

print(
    OUTPUT_SCORES_PATH
)


print(
    f"\nTop 25 ventas:"
)

print(
    OUTPUT_TOP25_PATH
)


print(
    f"\nTop 30 ventas:"
)

print(
    OUTPUT_TOP30_PATH
)


print(
    "\nPolítica:"
)

print(
    "1. risk_rank determina quién entra "
    "en Top 25 / Top 30."
)

print(
    "2. sales_visit_priority ordena esos "
    "clientes por riesgo × valor económico."
)

print(
    "3. risk_value_score NO se interpreta "
    "como pérdida monetaria esperada."
)


# =========================================================
# 29. FIN
# =========================================================

predictions.unpersist()

spark.stop()