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
    .appName("retailco-evaluate-model-ranking")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. PATHS
# =========================================================

VALIDATION_PATH = "data/gold/ml_validation"

LOGISTIC_MODEL_PATH = "models/logistic_regression"
RF_MODEL_PATH = "models/random_forest"
XGB_MODEL_PATH = "models/xgboost"


# =========================================================
# 2. FEATURES
#
# Las necesitamos especialmente para XGBoost,
# porque sus columnas deben ser numéricas.
# =========================================================

FEATURE_COLUMNS = [
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
# 3. CARGAR VALIDATION
#
# TEST sigue completamente sin tocarse.
# =========================================================

validation = spark.read.parquet(VALIDATION_PATH)

print("\nCargando modelos...")


# =========================================================
# 4. CARGAR MODELOS YA ENTRENADOS
# =========================================================

logistic_model = PipelineModel.load(
    LOGISTIC_MODEL_PATH
)

rf_model = PipelineModel.load(
    RF_MODEL_PATH
)

xgb_model = SparkXGBClassifierModel.load(
    XGB_MODEL_PATH
)


# =========================================================
# 5. SCORE LOGISTIC REGRESSION
# =========================================================

logistic_scores = (
    logistic_model
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
        F.lit("LOGISTIC")
    )
)


# =========================================================
# 6. SCORE RANDOM FOREST
# =========================================================

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
        F.lit("RANDOM_FOREST")
    )
)


# =========================================================
# 7. PREPARAR VALIDATION PARA XGBOOST
#
# SparkXGBClassifier necesita integral / float / double
# cuando features_col es una lista de columnas.
# =========================================================

validation_xgb = validation

for col_name in FEATURE_COLUMNS:

    validation_xgb = validation_xgb.withColumn(
        col_name,
        F.col(col_name).cast("double")
    )

validation_xgb = validation_xgb.withColumn(
    "target_drop_30",
    F.col("target_drop_30").cast("double")
)


# =========================================================
# 8. SCORE XGBOOST
# =========================================================

xgb_scores = (
    xgb_model
    .transform(validation_xgb)
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
        F.lit("XGBOOST")
    )
)


# =========================================================
# 9. UNIR SCORES
# =========================================================

scores = (
    logistic_scores
    .unionByName(rf_scores)
    .unionByName(xgb_scores)
)


# =========================================================
# 10. RANKING MENSUAL
#
# El nº 1 es el cliente con mayor riesgo estimado
# dentro de ese mes.
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
    F.row_number().over(ranking_window)
)


# =========================================================
# 11. Nº DE POSITIVOS REALES POR MES
#
# Necesario para Recall@K.
# =========================================================

monthly_totals = (
    scores
    .groupBy(
        "model",
        "month_start"
    )
    .agg(
        F.count("*").alias(
            "eligible_customers"
        ),

        F.sum(
            "target_drop_30"
        ).alias(
            "actual_positive_customers"
        )
    )
)


# =========================================================
# 12. FUNCIÓN PARA PRECISION@K / RECALL@K
# =========================================================

def calculate_at_k(scores_df, k):

    selected = (
        scores_df
        .filter(
            F.col("risk_rank") <= k
        )
        .groupBy(
            "model",
            "month_start"
        )
        .agg(
            F.count("*").alias(
                "selected_customers"
            ),

            F.sum(
                "target_drop_30"
            ).alias(
                "true_positives_at_k"
            ),

            F.avg(
                "probability_drop"
            ).alias(
                "avg_probability_top_k"
            )
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
            ).otherwise(0.0)
        )
        .withColumn(
            "recall_at_k",
            F.when(
                F.col("actual_positive_customers") > 0,
                F.col("true_positives_at_k")
                /
                F.col("actual_positive_customers")
            ).otherwise(0.0)
        )
    )

    return result


# =========================================================
# 13. CALCULAR @25 Y @30
# =========================================================

ranking_25 = calculate_at_k(
    scores,
    25
)

ranking_30 = calculate_at_k(
    scores,
    30
)

ranking_metrics = (
    ranking_25
    .unionByName(ranking_30)
)


# =========================================================
# 14. RESULTADO MES A MES
# =========================================================

print("\n" + "=" * 70)
print("MODEL RANKING - VALIDATION")
print("=" * 70)


print("\nPRECISION / RECALL POR MES:")

ranking_metrics.select(
    "model",
    "month_start",
    "k",
    "eligible_customers",
    "actual_positive_customers",
    "selected_customers",
    "true_positives_at_k",
    "precision_at_k",
    "recall_at_k"
).orderBy(
    "month_start",
    "k",
    "model"
).show(
    200,
    truncate=False
)


# =========================================================
# 15. RESUMEN PROMEDIO POR MODELO
#
# Macro average:
# cada mes pesa lo mismo.
# =========================================================

summary = (
    ranking_metrics
    .groupBy(
        "model",
        "k"
    )
    .agg(
        F.countDistinct(
            "month_start"
        ).alias(
            "months"
        ),

        F.avg(
            "precision_at_k"
        ).alias(
            "avg_precision_at_k"
        ),

        F.avg(
            "recall_at_k"
        ).alias(
            "avg_recall_at_k"
        ),

        F.sum(
            "true_positives_at_k"
        ).alias(
            "total_true_positives"
        ),

        F.sum(
            "selected_customers"
        ).alias(
            "total_selected"
        ),

        F.sum(
            "actual_positive_customers"
        ).alias(
            "total_actual_positives"
        )
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
)


print("\n" + "=" * 70)
print("RESUMEN DE RANKING")
print("=" * 70)

summary.select(
    "model",
    "k",
    "months",
    "avg_precision_at_k",
    "avg_recall_at_k",
    "global_precision_at_k",
    "global_recall_at_k",
    "total_true_positives",
    "total_selected"
).orderBy(
    "k",
    F.desc("avg_precision_at_k")
).show(
    truncate=False
)


# =========================================================
# 16. LIFT VS SELECCIÓN ALEATORIA
#
# Si un mes tiene ~39% de positivos,
# seleccionar clientes al azar daría ~39% precision.
#
# El lift mide cuánto mejor ordena el modelo.
# =========================================================

base_rate = (
    validation
    .agg(
        F.avg(
            "target_drop_30"
        ).alias("base_rate")
    )
    .first()["base_rate"]
)

summary = summary.withColumn(
    "lift_vs_random",
    F.col("global_precision_at_k")
    /
    F.lit(base_rate)
)


print("\nBASE RATE VALIDATION:")
print(
    f"{base_rate * 100:.2f}%"
)


print("\nLIFT VS SELECCIÓN ALEATORIA:")

summary.select(
    "model",
    "k",
    "global_precision_at_k",
    "global_recall_at_k",
    "lift_vs_random"
).orderBy(
    "k",
    F.desc("lift_vs_random")
).show(
    truncate=False
)


# =========================================================
# 17. TOP 30 DEL ÚLTIMO MES DE VALIDATION
#
# Esto nos permite ver cómo sería una lista real
# para ventas.
# =========================================================

last_validation_month = (
    validation
    .agg(
        F.max("month_start")
    )
    .first()[0]
)

print(
    "\nÚLTIMO MES DE VALIDATION:",
    last_validation_month
)


print("\nTOP 30 RANDOM FOREST - ÚLTIMO MES:")

scores.filter(
    (F.col("model") == "RANDOM_FOREST") &
    (F.col("month_start") == last_validation_month) &
    (F.col("risk_rank") <= 30)
).select(
    "risk_rank",
    "customer_id",
    "probability_drop",
    "target_drop_30"
).orderBy(
    "risk_rank"
).show(
    30,
    truncate=False
)


print("\nTOP 30 XGBOOST - ÚLTIMO MES:")

scores.filter(
    (F.col("model") == "XGBOOST") &
    (F.col("month_start") == last_validation_month) &
    (F.col("risk_rank") <= 30)
).select(
    "risk_rank",
    "customer_id",
    "probability_drop",
    "target_drop_30"
).orderBy(
    "risk_rank"
).show(
    30,
    truncate=False
)


# =========================================================
# 18. FIN
# =========================================================

spark.stop()