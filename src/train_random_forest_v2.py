from pyspark.sql import SparkSession, functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


# =========================================================
# 0. SPARK
# =========================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-train-random-forest-v2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. PATHS
# =========================================================

TRAIN_PATH = "data/gold/ml_train_v2"
VALIDATION_PATH = "data/gold/ml_validation_v2"

MODEL_PATH = "models/random_forest_v2"


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
# 3. NUEVAS FEATURES V2
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
# 5. CARGAR TRAIN Y VALIDATION
#
# TEST NO SE CARGA.
# =========================================================

train = spark.read.parquet(
    TRAIN_PATH
)

validation = spark.read.parquet(
    VALIDATION_PATH
)


print("\n" + "=" * 70)
print("RANDOM FOREST V2")
print("=" * 70)

print(
    f"\nFeatures utilizadas: "
    f"{len(FEATURE_COLUMNS)}"
)


# =========================================================
# 6. CAST A DOUBLE
#
# Dejamos todas las features numéricamente homogéneas.
# =========================================================

for column_name in FEATURE_COLUMNS:

    train = train.withColumn(
        column_name,
        F.col(column_name).cast("double")
    )

    validation = validation.withColumn(
        column_name,
        F.col(column_name).cast("double")
    )


train = train.withColumn(
    "target_drop_30",
    F.col("target_drop_30").cast("double")
)

validation = validation.withColumn(
    "target_drop_30",
    F.col("target_drop_30").cast("double")
)


# =========================================================
# 7. VALIDACIONES BÁSICAS
# =========================================================

train_count = train.count()
validation_count = validation.count()


print(
    f"TRAIN observations:      "
    f"{train_count:,}"
)

print(
    f"VALIDATION observations: "
    f"{validation_count:,}"
)


# =========================================================
# 8. VECTOR ASSEMBLER
# =========================================================

assembler = VectorAssembler(
    inputCols=FEATURE_COLUMNS,
    outputCol="features",
    handleInvalid="error"
)


# =========================================================
# 9. RANDOM FOREST
#
# EXACTAMENTE LOS MISMOS HIPERPARÁMETROS QUE V1.
#
# Esto es importante:
# queremos medir el efecto de FEATURES V2,
# no mezclar feature engineering + tuning.
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
# 10. PIPELINE
# =========================================================

pipeline = Pipeline(
    stages=[
        assembler,
        rf,
    ]
)


# =========================================================
# 11. ENTRENAR SOLO CON TRAIN
# =========================================================

print("\nEntrenando Random Forest V2...")

model = pipeline.fit(
    train
)


# =========================================================
# 12. GUARDAR MODELO
# =========================================================

model.write() \
    .overwrite() \
    .save(MODEL_PATH)


print(
    f"\nModelo guardado en: "
    f"{MODEL_PATH}"
)


# =========================================================
# 13. PREDICCIONES EN VALIDATION
# =========================================================

predictions = (
    model
    .transform(validation)
    .withColumn(
        "probability_drop",
        vector_to_array("probability")[1]
    )
    .cache()
)


prediction_count = predictions.count()


print(
    f"Predicciones VALIDATION: "
    f"{prediction_count:,}"
)


# =========================================================
# 14. AUC ROC
# =========================================================

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="target_drop_30",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc_roc = roc_evaluator.evaluate(
    predictions
)


# =========================================================
# 15. AUC PR
# =========================================================

pr_evaluator = BinaryClassificationEvaluator(
    labelCol="target_drop_30",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

auc_pr = pr_evaluator.evaluate(
    predictions
)


# =========================================================
# 16. THRESHOLDS
# =========================================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


threshold_results = []


# =========================================================
# 17. EVALUAR THRESHOLDS
# =========================================================

for threshold in thresholds:

    scored = predictions.withColumn(
        "predicted_label",

        (
            F.col("probability_drop")
            >=
            F.lit(threshold)
        ).cast("int")
    )


    row = (
        scored
        .agg(

            F.sum(
                F.when(
                    (
                        F.col("predicted_label") == 1
                    )
                    &
                    (
                        F.col("target_drop_30") == 1
                    ),
                    1
                ).otherwise(0)
            ).alias("tp"),

            F.sum(
                F.when(
                    (
                        F.col("predicted_label") == 1
                    )
                    &
                    (
                        F.col("target_drop_30") == 0
                    ),
                    1
                ).otherwise(0)
            ).alias("fp"),

            F.sum(
                F.when(
                    (
                        F.col("predicted_label") == 0
                    )
                    &
                    (
                        F.col("target_drop_30") == 1
                    ),
                    1
                ).otherwise(0)
            ).alias("fn"),

            F.sum(
                F.when(
                    (
                        F.col("predicted_label") == 0
                    )
                    &
                    (
                        F.col("target_drop_30") == 0
                    ),
                    1
                ).otherwise(0)
            ).alias("tn"),
        )
        .first()
    )


    tp = row["tp"]
    fp = row["fp"]
    fn = row["fn"]
    tn = row["tn"]


    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )


    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )


    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )


    threshold_results.append(
        {
            "threshold": threshold,

            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,

            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


# =========================================================
# 18. MEJOR THRESHOLD SEGÚN F1
# =========================================================

best = max(
    threshold_results,
    key=lambda x: x["f1"]
)


# =========================================================
# 19. FEATURE IMPORTANCE
#
# Pipeline:
# stage 0 = VectorAssembler
# stage 1 = RandomForestClassificationModel
# =========================================================

rf_model = model.stages[1]

feature_importances = (
    rf_model.featureImportances.toArray()
)


importance_rows = list(
    zip(
        FEATURE_COLUMNS,
        feature_importances
    )
)


importance_rows = sorted(
    importance_rows,
    key=lambda x: x[1],
    reverse=True
)


# =========================================================
# 20. RESULTADOS
# =========================================================

print("\n" + "=" * 70)
print("RANDOM FOREST V2 - VALIDATION")
print("=" * 70)


print(
    f"\nAUC ROC: "
    f"{auc_roc:.4f}"
)

print(
    f"AUC PR:  "
    f"{auc_pr:.4f}"
)


# =========================================================
# 21. TABLA THRESHOLDS
# =========================================================

print("\nTHRESHOLDS:")


print(
    f"{'THRESHOLD':<12}"
    f"{'PRECISION':<12}"
    f"{'RECALL':<12}"
    f"{'F1':<12}"
)


for result in threshold_results:

    print(
        f"{result['threshold']:<12.2f}"
        f"{result['precision']:<12.3f}"
        f"{result['recall']:<12.3f}"
        f"{result['f1']:<12.3f}"
    )


# =========================================================
# 22. MEJOR THRESHOLD
# =========================================================

print(
    "\nMEJOR THRESHOLD EN VALIDATION:"
)


print(
    f"Threshold: "
    f"{best['threshold']:.2f}"
)

print(
    f"Precision: "
    f"{best['precision']:.3f}"
)

print(
    f"Recall:    "
    f"{best['recall']:.3f}"
)

print(
    f"F1:        "
    f"{best['f1']:.3f}"
)


# =========================================================
# 23. MATRIZ DE CONFUSIÓN
# =========================================================

print(
    "\nMATRIZ DE CONFUSIÓN:"
)

print(
    f"TP: {best['tp']}"
)

print(
    f"FP: {best['fp']}"
)

print(
    f"FN: {best['fn']}"
)

print(
    f"TN: {best['tn']}"
)


# =========================================================
# 24. TOP FEATURES
# =========================================================

print(
    "\nTOP 20 FEATURES:"
)


for feature, importance in importance_rows[:20]:

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
# 25. CUÁNTA IMPORTANCIA TOTAL APORTAN LAS FEATURES V2
# =========================================================

new_feature_importance = sum(

    importance

    for feature, importance
    in importance_rows

    if feature in NEW_FEATURE_COLUMNS
)


base_feature_importance = sum(

    importance

    for feature, importance
    in importance_rows

    if feature in BASE_FEATURE_COLUMNS
)


print(
    "\nIMPORTANCIA AGREGADA:"
)


print(
    f"Features V1: "
    f"{base_feature_importance:.4f}"
)

print(
    f"Features nuevas V2: "
    f"{new_feature_importance:.4f}"
)


# =========================================================
# 26. PROBABILIDADES
# =========================================================

print(
    "\nPROBABILIDADES EN VALIDATION:"
)


predictions.select(

    F.min(
        "probability_drop"
    ).alias(
        "min_probability"
    ),

    F.avg(
        "probability_drop"
    ).alias(
        "avg_probability"
    ),

    F.max(
        "probability_drop"
    ).alias(
        "max_probability"
    ),

).show(
    truncate=False
)


# =========================================================
# 27. COMPARACIÓN CONTRA RANDOM FOREST V1
# =========================================================

print("\n" + "=" * 70)
print("COMPARACIÓN RF V1 VS RF V2")
print("=" * 70)


print(
    "\nRandom Forest V1:"
)

print(
    "  AUC ROC = 0.6815"
)

print(
    "  AUC PR  = 0.5482"
)

print(
    "  F1      = 0.606"
)


print(
    "\nRandom Forest V2:"
)

print(
    f"  AUC ROC = "
    f"{auc_roc:.4f}"
)

print(
    f"  AUC PR  = "
    f"{auc_pr:.4f}"
)

print(
    f"  F1      = "
    f"{best['f1']:.3f}"
)


print(
    "\nDIFERENCIA V2 - V1:"
)

print(
    f"  Delta AUC ROC = "
    f"{auc_roc - 0.6815:+.4f}"
)

print(
    f"  Delta AUC PR  = "
    f"{auc_pr - 0.5482:+.4f}"
)

print(
    f"  Delta F1      = "
    f"{best['f1'] - 0.606:+.3f}"
)


# =========================================================
# 28. FIN
# =========================================================

predictions.unpersist()

spark.stop()