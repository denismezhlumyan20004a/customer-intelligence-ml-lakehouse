from pyspark.sql import SparkSession, functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier


# =========================================================
# 0. SPARK
# =========================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-train-production-model")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. PATHS
# =========================================================

TRAINING_DATA_PATH = "data/gold/ml_features_v2"

PRODUCTION_MODEL_PATH = (
    "models/random_forest_production"
)


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


# =========================================================
# 4. TODAS LAS FEATURES
# =========================================================

FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    +
    NEW_FEATURE_COLUMNS
)


# =========================================================
# 5. CARGAR TODO EL HISTÓRICO ETIQUETADO
#
# IMPORTANTE:
#
# Este dataset contiene todas las observaciones
# etiquetadas hasta abril de 2026.
#
# Ya hemos terminado la evaluación out-of-time.
# Por eso ahora podemos aprovechar:
#
# TRAIN
# VALIDATION
# EMBARGO
# antiguo TEST
#
# para construir el modelo operativo definitivo.
# =========================================================

training = (
    spark.read
    .parquet(TRAINING_DATA_PATH)
)


print("\n" + "=" * 70)
print("TRAIN PRODUCTION MODEL")
print("=" * 70)

print(
    f"\nFeatures utilizadas: "
    f"{len(FEATURE_COLUMNS)}"
)


# =========================================================
# 6. CAST NUMÉRICO
# =========================================================

for column_name in FEATURE_COLUMNS:

    training = training.withColumn(
        column_name,
        F.col(column_name).cast("double")
    )


training = training.withColumn(
    "target_drop_30",
    F.col("target_drop_30").cast("double")
)


# =========================================================
# 7. VALIDACIONES DEL DATASET
# =========================================================

training_count = training.count()


print(
    f"Observaciones etiquetadas: "
    f"{training_count:,}"
)


# Esperamos las 4.709 observaciones oficiales V3/V2
EXPECTED_ROWS = 4709


if training_count != EXPECTED_ROWS:

    raise ValueError(
        f"Esperábamos {EXPECTED_ROWS} observaciones "
        f"pero encontramos {training_count}."
    )


# =========================================================
# 8. RANGO TEMPORAL
# =========================================================

date_stats = (
    training
    .agg(
        F.min(
            "month_start"
        ).alias("min_month"),

        F.max(
            "month_start"
        ).alias("max_month"),
    )
    .first()
)


print(
    f"Periodo etiquetado: "
    f"{date_stats['min_month']} "
    f"-> "
    f"{date_stats['max_month']}"
)


# =========================================================
# 9. TARGET DISTRIBUTION
# =========================================================

target_stats = (
    training
    .agg(
        F.sum(
            "target_drop_30"
        ).alias("positives"),

        F.avg(
            "target_drop_30"
        ).alias("positive_rate"),
    )
    .first()
)


print(
    f"Positivos: "
    f"{int(target_stats['positives']):,}"
)

print(
    f"Positive rate: "
    f"{target_stats['positive_rate'] * 100:.2f}%"
)


# =========================================================
# 10. COMPROBAR NULLS
# =========================================================

null_expressions = [

    F.sum(
        F.when(
            F.col(column_name).isNull(),
            1
        ).otherwise(0)
    ).alias(column_name)

    for column_name
    in FEATURE_COLUMNS
]


null_row = (
    training
    .agg(
        *null_expressions
    )
    .first()
)


total_nulls = sum(
    null_row[column_name]
    for column_name
    in FEATURE_COLUMNS
)


print(
    f"Nulls en features: "
    f"{total_nulls}"
)


if total_nulls > 0:

    raise ValueError(
        "El dataset de entrenamiento "
        "contiene nulls en las features."
    )


# =========================================================
# 11. VECTOR ASSEMBLER
# =========================================================

assembler = VectorAssembler(

    inputCols=FEATURE_COLUMNS,

    outputCol="features",

    handleInvalid="error",
)


# =========================================================
# 12. RANDOM FOREST PRODUCTION
#
# HIPERPARÁMETROS CONGELADOS
#
# Son exactamente los del champion:
#
# numTrees = 300
# maxDepth = 6
# minInstancesPerNode = 5
# featureSubsetStrategy = sqrt
#
# No hacemos tuning adicional.
# =========================================================

rf = RandomForestClassifier(

    featuresCol="features",

    labelCol="target_drop_30",

    numTrees=300,

    maxDepth=6,

    minInstancesPerNode=5,

    featureSubsetStrategy="sqrt",

    seed=42,
)


# =========================================================
# 13. PIPELINE
# =========================================================

pipeline = Pipeline(
    stages=[
        assembler,
        rf,
    ]
)


# =========================================================
# 14. ENTRENAMIENTO PRODUCTIVO
#
# No calculamos AUC/F1 sobre training:
# sería una métrica optimista y no representa
# generalización.
#
# La métrica oficial sigue siendo la obtenida
# anteriormente sobre TEST 2026 out-of-time.
# =========================================================

print(
    "\nEntrenando Random Forest "
    "PRODUCTION con todo el histórico etiquetado..."
)


production_model = pipeline.fit(
    training
)


# =========================================================
# 15. GUARDAR MODELO
# =========================================================

production_model.write() \
    .overwrite() \
    .save(
        PRODUCTION_MODEL_PATH
    )


print(
    f"\nModelo production guardado en:"
)

print(
    PRODUCTION_MODEL_PATH
)


# =========================================================
# 16. FEATURE IMPORTANCE
# =========================================================

rf_model = production_model.stages[1]


feature_importances = (
    rf_model
    .featureImportances
    .toArray()
)


importance_rows = sorted(

    zip(
        FEATURE_COLUMNS,
        feature_importances
    ),

    key=lambda x: x[1],

    reverse=True
)


print(
    "\nTOP 15 FEATURES "
    "- PRODUCTION MODEL:"
)


for feature, importance in importance_rows[:15]:

    marker = (
        " [V2]"
        if feature in NEW_FEATURE_COLUMNS
        else ""
    )

    print(
        f"{feature:<38} "
        f"{importance:.4f}"
        f"{marker}"
    )


# =========================================================
# 17. RESUMEN METODOLÓGICO
# =========================================================

print("\n" + "=" * 70)
print("PRODUCTION MODEL READY")
print("=" * 70)


print(
    "\nModelo: Random Forest V2"
)

print(
    "Features: 43"
)

print(
    "Hiperparámetros congelados "
    "antes del TEST final."
)

print(
    f"Training observations: "
    f"{training_count:,}"
)

print(
    f"Training period: "
    f"{date_stats['min_month']} "
    f"-> "
    f"{date_stats['max_month']}"
)

print(
    "\nEste modelo se utilizará para scoring operativo."
)

print(
    "NO debe utilizarse para recalcular "
    "las métricas oficiales de TEST."
)


# =========================================================
# 18. FIN
# =========================================================

spark.stop()