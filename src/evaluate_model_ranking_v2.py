from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

from xgboost.spark import SparkXGBClassifierModel


# =========================================================
# 0. SPARK
# =========================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-evaluate-model-ranking-v2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. PATHS
# =========================================================

VALIDATION_PATH = "data/gold/ml_validation_v2"

RF_MODEL_PATH = "models/random_forest_v2"
XGB_MODEL_PATH = "models/xgboost_v2"


# =========================================================
# 2. FEATURES V1
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


# =========================================================
# 3. FEATURES V2
# =========================================================

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
# 4. CARGAR SOLO VALIDATION
#
# TEST NO SE CARGA.
# =========================================================

validation = spark.read.parquet(
    VALIDATION_PATH
)


# =========================================================
# 5. CAST NUMÉRICO
#
# Dejamos el mismo dataset preparado para los dos modelos.
# =========================================================

for column_name in FEATURE_COLUMNS:

    validation = validation.withColumn(
        column_name,
        F.col(column_name).cast("double")
    )


validation = validation.withColumn(
    "target_drop_30",
    F.col("target_drop_30").cast("double")
)


validation = validation.cache()

validation_count = validation.count()


print("\n" + "=" * 70)
print("MODEL RANKING V2 - VALIDATION")
print("=" * 70)

print(
    f"\nValidation observations: "
    f"{validation_count:,}"
)

print(
    f"Validation months: "
    f"{validation.select('month_start').distinct().count()}"
)

print(
    "\nTEST NO se ha cargado."
)


# =========================================================
# 6. CARGAR MODELOS
# =========================================================

print("\nCargando modelos...")


rf_model = PipelineModel.load(
    RF_MODEL_PATH
)


xgb_model = SparkXGBClassifierModel.load(
    XGB_MODEL_PATH
)


# =========================================================
# 7. RANDOM FOREST SCORES
# =========================================================

print(
    "\nGenerando scores Random Forest V2..."
)


rf_scores = (
    rf_model
    .transform(validation)
    .withColumn(
        "probability_drop",
        vector_to_array("probability")[1]
    )
    .select(
        "customer_id",
        "month_start",
        "target_drop_30",
        "probability_drop"
    )
    .withColumn(
        "model",
        F.lit("RANDOM_FOREST_V2")
    )
)


# =========================================================
# 8. XGBOOST SCORES
# =========================================================

print(
    "Generando scores XGBoost V2..."
)


xgb_scores = (
    xgb_model
    .transform(validation)
    .withColumn(
        "probability_drop",
        vector_to_array("probability")[1]
    )
    .select(
        "customer_id",
        "month_start",
        "target_drop_30",
        "probability_drop"
    )
    .withColumn(
        "model",
        F.lit("XGBOOST_V2")
    )
)


# =========================================================
# 9. UNIR SCORES
# =========================================================

scores = (
    rf_scores
    .unionByName(
        xgb_scores
    )
    .cache()
)


scores.count()


# =========================================================
# 10. RANKING DE RIESGO POR MES
#
# Rank 1 = cliente con mayor probabilidad estimada
# de sufrir la caída futura.
#
# customer_id se usa únicamente para desempate estable.
# =========================================================

ranking_window = (
    Window
    .partitionBy(
        "model",
        "month_start"
    )
    .orderBy(
        F.desc("probability_drop"),
        F.asc("customer_id")
    )
)


scores = scores.withColumn(
    "risk_rank",
    F.row_number().over(
        ranking_window
    )
)


# =========================================================
# 11. SITUACIÓN REAL POR MES
#
# Calculamos:
#
# - clientes elegibles
# - positivos reales
# - base rate
#
# El base rate representa qué precisión conseguiríamos
# aproximadamente seleccionando clientes sin ranking.
# =========================================================

monthly_totals = (
    scores
    .groupBy(
        "model",
        "month_start"
    )
    .agg(

        F.count("*")
        .alias(
            "eligible_customers"
        ),

        F.sum(
            "target_drop_30"
        )
        .alias(
            "actual_positives"
        ),
    )
    .withColumn(
        "monthly_base_rate",

        F.col("actual_positives")
        /
        F.col("eligible_customers")
    )
)


# =========================================================
# 12. FUNCIÓN PRECISION / RECALL / LIFT @ K
# =========================================================

def calculate_at_k(
    ranked_scores,
    k
):

    selected = (
        ranked_scores

        .filter(
            F.col("risk_rank")
            <=
            F.lit(k)
        )

        .groupBy(
            "model",
            "month_start"
        )

        .agg(

            F.count("*")
            .alias(
                "selected_customers"
            ),

            F.sum(
                "target_drop_30"
            )
            .alias(
                "true_positives_at_k"
            ),

            F.avg(
                "probability_drop"
            )
            .alias(
                "avg_score_top_k"
            ),

            F.min(
                "probability_drop"
            )
            .alias(
                "min_score_top_k"
            ),
        )
    )


    result = (
        selected

        .join(
            monthly_totals,
            [
                "model",
                "month_start"
            ],
            "inner"
        )

        .withColumn(
            "k",
            F.lit(k)
        )

        .withColumn(
            "precision_at_k",

            F.when(
                F.col("selected_customers") > 0,

                F.col("true_positives_at_k")
                /
                F.col("selected_customers")
            )
            .otherwise(
                F.lit(0.0)
            )
        )

        .withColumn(
            "recall_at_k",

            F.when(
                F.col("actual_positives") > 0,

                F.col("true_positives_at_k")
                /
                F.col("actual_positives")
            )
            .otherwise(
                F.lit(0.0)
            )
        )

        .withColumn(
            "lift_at_k",

            F.when(
                F.col("monthly_base_rate") > 0,

                F.col("precision_at_k")
                /
                F.col("monthly_base_rate")
            )
            .otherwise(
                F.lit(0.0)
            )
        )
    )


    return result


# =========================================================
# 13. TOP 25 Y TOP 30
# =========================================================

metrics_25 = calculate_at_k(
    scores,
    25
)


metrics_30 = calculate_at_k(
    scores,
    30
)


ranking_metrics = (
    metrics_25
    .unionByName(
        metrics_30
    )
    .cache()
)


ranking_metrics.count()


# =========================================================
# 14. CANDIDATOS DISPONIBLES POR MES
#
# Antes de interpretar Top 25/30 queremos comprobar que
# realmente existen suficientes clientes elegibles.
# =========================================================

print("\n" + "=" * 70)
print("CANDIDATOS ELEGIBLES POR MES")
print("=" * 70)


(
    monthly_totals

    .filter(
        F.col("model")
        ==
        "XGBOOST_V2"
    )

    .select(
        "month_start",
        "eligible_customers",
        "actual_positives",
        "monthly_base_rate"
    )

    .orderBy(
        "month_start"
    )

    .show(
        100,
        truncate=False
    )
)


# =========================================================
# 15. RESULTADOS MES A MES
# =========================================================

print("\n" + "=" * 70)
print("PRECISION / RECALL / LIFT POR MES")
print("=" * 70)


(
    ranking_metrics

    .select(
        "model",
        "month_start",
        "k",

        "eligible_customers",
        "actual_positives",

        "selected_customers",
        "true_positives_at_k",

        "precision_at_k",
        "recall_at_k",
        "lift_at_k"
    )

    .orderBy(
        "month_start",
        "k",
        "model"
    )

    .show(
        200,
        truncate=False
    )
)


# =========================================================
# 16. MACRO AVERAGE
#
# Cada mes pesa exactamente lo mismo.
#
# Esto evita que un mes con más clientes domine
# completamente el resultado.
# =========================================================

macro_summary = (
    ranking_metrics

    .groupBy(
        "model",
        "k"
    )

    .agg(

        F.countDistinct(
            "month_start"
        )
        .alias(
            "months"
        ),

        F.avg(
            "precision_at_k"
        )
        .alias(
            "avg_precision_at_k"
        ),

        F.avg(
            "recall_at_k"
        )
        .alias(
            "avg_recall_at_k"
        ),

        F.avg(
            "lift_at_k"
        )
        .alias(
            "avg_lift_at_k"
        ),

        F.min(
            "precision_at_k"
        )
        .alias(
            "worst_month_precision"
        ),

        F.max(
            "precision_at_k"
        )
        .alias(
            "best_month_precision"
        ),
    )
)


print("\n" + "=" * 70)
print("MACRO AVERAGE - CADA MES PESA IGUAL")
print("=" * 70)


(
    macro_summary

    .orderBy(
        "k",
        F.desc(
            "avg_precision_at_k"
        )
    )

    .show(
        truncate=False
    )
)


# =========================================================
# 17. GLOBAL / MICRO AVERAGE
#
# Aquí juntamos todas las decisiones realizadas durante
# los nueve meses.
#
# Ejemplo Top 25:
# 25 visitas x 9 meses = 225 acciones comerciales,
# siempre que existan >=25 candidatos cada mes.
# =========================================================

global_summary = (
    ranking_metrics

    .groupBy(
        "model",
        "k"
    )

    .agg(

        F.sum(
            "selected_customers"
        )
        .alias(
            "total_selected"
        ),

        F.sum(
            "true_positives_at_k"
        )
        .alias(
            "total_true_positives"
        ),

        F.sum(
            "actual_positives"
        )
        .alias(
            "total_actual_positives"
        ),

        F.sum(
            "eligible_customers"
        )
        .alias(
            "total_eligible"
        ),
    )

    .withColumn(
        "global_precision_at_k",

        F.col("total_true_positives")
        /
        F.col("total_selected")
    )

    .withColumn(
        "global_recall_at_k",

        F.col("total_true_positives")
        /
        F.col("total_actual_positives")
    )

    .withColumn(
        "global_base_rate",

        F.col("total_actual_positives")
        /
        F.col("total_eligible")
    )

    .withColumn(
        "global_lift_at_k",

        F.col("global_precision_at_k")
        /
        F.col("global_base_rate")
    )
)


print("\n" + "=" * 70)
print("GLOBAL / MICRO AVERAGE")
print("=" * 70)


(
    global_summary

    .select(
        "model",
        "k",

        "total_selected",
        "total_true_positives",
        "total_actual_positives",

        "global_precision_at_k",
        "global_recall_at_k",
        "global_base_rate",
        "global_lift_at_k"
    )

    .orderBy(
        "k",
        F.desc(
            "global_precision_at_k"
        )
    )

    .show(
        truncate=False
    )
)


# =========================================================
# 18. RESUMEN COMPACTO
#
# Este será el bloque más útil para decidir champion.
# =========================================================

final_summary = (
    macro_summary

    .join(
        global_summary,
        [
            "model",
            "k"
        ],
        "inner"
    )

    .select(
        "model",
        "k",

        "avg_precision_at_k",
        "avg_recall_at_k",
        "avg_lift_at_k",

        "global_precision_at_k",
        "global_recall_at_k",
        "global_lift_at_k",

        "worst_month_precision",
        "best_month_precision",

        "total_selected",
        "total_true_positives",
    )
)


print("\n" + "=" * 70)
print("RESUMEN FINAL - RF V2 VS XGBOOST V2")
print("=" * 70)


(
    final_summary

    .orderBy(
        "k",
        F.desc(
            "avg_precision_at_k"
        )
    )

    .show(
        truncate=False
    )
)


# =========================================================
# 19. TOP 30 ÚLTIMO MES DE VALIDATION
#
# Nos permite ver un ejemplo cercano a la lista que
# finalmente recibiría ventas.
# =========================================================

last_month = (
    validation
    .agg(
        F.max("month_start")
        .alias("last_month")
    )
    .first()["last_month"]
)


print(
    "\nÚLTIMO MES DE VALIDATION:",
    last_month
)


print("\nTOP 30 RANDOM FOREST V2:")


(
    scores

    .filter(
        (
            F.col("model")
            ==
            "RANDOM_FOREST_V2"
        )
        &
        (
            F.col("month_start")
            ==
            F.lit(last_month)
        )
        &
        (
            F.col("risk_rank")
            <=
            30
        )
    )

    .select(
        "risk_rank",
        "customer_id",
        "probability_drop",
        "target_drop_30"
    )

    .orderBy(
        "risk_rank"
    )

    .show(
        30,
        truncate=False
    )
)


print("\nTOP 30 XGBOOST V2:")


(
    scores

    .filter(
        (
            F.col("model")
            ==
            "XGBOOST_V2"
        )
        &
        (
            F.col("month_start")
            ==
            F.lit(last_month)
        )
        &
        (
            F.col("risk_rank")
            <=
            30
        )
    )

    .select(
        "risk_rank",
        "customer_id",
        "probability_drop",
        "target_drop_30"
    )

    .orderBy(
        "risk_rank"
    )

    .show(
        30,
        truncate=False
    )
)


# =========================================================
# 20. OVERLAP TOP 30
#
# Opcional pero útil:
# ¿RF y XGBoost están identificando aproximadamente
# a los mismos clientes o modelos muy distintos?
# =========================================================

rf_top30 = (
    scores

    .filter(
        (
            F.col("model")
            ==
            "RANDOM_FOREST_V2"
        )
        &
        (
            F.col("risk_rank")
            <=
            30
        )
    )

    .select(
        "month_start",
        "customer_id"
    )

    .withColumn(
        "rf_selected",
        F.lit(1)
    )
)


xgb_top30 = (
    scores

    .filter(
        (
            F.col("model")
            ==
            "XGBOOST_V2"
        )
        &
        (
            F.col("risk_rank")
            <=
            30
        )
    )

    .select(
        "month_start",
        "customer_id"
    )

    .withColumn(
        "xgb_selected",
        F.lit(1)
    )
)


overlap = (
    rf_top30

    .join(
        xgb_top30,
        [
            "month_start",
            "customer_id"
        ],
        "inner"
    )

    .groupBy(
        "month_start"
    )

    .agg(
        F.count("*")
        .alias(
            "common_clients_top30"
        )
    )

    .withColumn(
        "overlap_pct",

        F.col("common_clients_top30")
        /
        F.lit(30.0)
    )
)


print("\n" + "=" * 70)
print("OVERLAP RF V2 VS XGBOOST V2 - TOP 30")
print("=" * 70)


(
    overlap
    .orderBy(
        "month_start"
    )
    .show(
        100,
        truncate=False
    )
)


# =========================================================
# 21. FIN
# =========================================================

ranking_metrics.unpersist()
scores.unpersist()
validation.unpersist()

spark.stop()