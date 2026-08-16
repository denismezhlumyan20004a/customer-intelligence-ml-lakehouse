from pyspark.sql import SparkSession, functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

from xgboost.spark import SparkXGBClassifierModel


# =========================================================
# 0. SPARK
# =========================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-evaluate-tuned-xgboost")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. PATHS
# =========================================================

VALIDATION_PATH = "data/gold/ml_validation"
MODEL_PATH = "models/xgboost_tuned"


# =========================================================
# 2. FEATURES
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
# 3. CARGAR SOLO VALIDATION
#
# TEST sigue sin tocarse.
# =========================================================

validation = spark.read.parquet(
    VALIDATION_PATH
)


# =========================================================
# 4. CAST A DOUBLE PARA XGBOOST
# =========================================================

for col_name in FEATURE_COLUMNS:

    validation = validation.withColumn(
        col_name,
        F.col(col_name).cast("double")
    )

validation = validation.withColumn(
    "target_drop_30",
    F.col("target_drop_30").cast("double")
)


# =========================================================
# 5. CARGAR MODELO TUNEADO
# =========================================================

print("\nCargando XGBoost tuned...")

model = SparkXGBClassifierModel.load(
    MODEL_PATH
)


# =========================================================
# 6. PREDICCIONES SOBRE VALIDATION
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
    f"Predicciones validation: "
    f"{prediction_count:,}"
)


# =========================================================
# 7. AUC ROC
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
# 8. AUC PR
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
# 9. THRESHOLDS
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
# 10. EVALUAR CADA THRESHOLD
# =========================================================

for threshold in thresholds:

    scored = predictions.withColumn(
        "predicted_label",
        (
            F.col("probability_drop")
            >= threshold
        ).cast("int")
    )

    row = (
        scored
        .agg(

            F.sum(
                F.when(
                    (F.col("predicted_label") == 1) &
                    (F.col("target_drop_30") == 1),
                    1
                ).otherwise(0)
            ).alias("tp"),

            F.sum(
                F.when(
                    (F.col("predicted_label") == 1) &
                    (F.col("target_drop_30") == 0),
                    1
                ).otherwise(0)
            ).alias("fp"),

            F.sum(
                F.when(
                    (F.col("predicted_label") == 0) &
                    (F.col("target_drop_30") == 1),
                    1
                ).otherwise(0)
            ).alias("fn"),

            F.sum(
                F.when(
                    (F.col("predicted_label") == 0) &
                    (F.col("target_drop_30") == 0),
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
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    threshold_results.append({
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


# =========================================================
# 11. MEJOR THRESHOLD
# =========================================================

best = max(
    threshold_results,
    key=lambda x: x["f1"]
)


# =========================================================
# 12. FEATURE IMPORTANCE
# =========================================================

raw_importances = (
    model.get_feature_importances(
        importance_type="gain"
    )
)

importance_rows = []

for feature, importance in raw_importances.items():

    if (
        feature.startswith("f")
        and feature[1:].isdigit()
    ):

        index = int(feature[1:])

        if index < len(FEATURE_COLUMNS):
            feature_name = FEATURE_COLUMNS[index]
        else:
            feature_name = feature

    else:
        feature_name = feature

    importance_rows.append(
        (
            feature_name,
            float(importance)
        )
    )

importance_rows = sorted(
    importance_rows,
    key=lambda x: x[1],
    reverse=True
)


# =========================================================
# 13. RESULTADOS
# =========================================================

print("\n" + "=" * 70)
print("XGBOOST TUNED - VALIDATION")
print("=" * 70)

print(f"\nAUC ROC: {auc_roc:.4f}")
print(f"AUC PR:  {auc_pr:.4f}")


# =========================================================
# 14. THRESHOLD TABLE
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
# 15. MEJOR RESULTADO
# =========================================================

print("\nMEJOR THRESHOLD EN VALIDATION:")

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


print("\nMATRIZ DE CONFUSIÓN:")

print(f"TP: {best['tp']}")
print(f"FP: {best['fp']}")
print(f"FN: {best['fn']}")
print(f"TN: {best['tn']}")


# =========================================================
# 16. TOP FEATURES
# =========================================================

print("\nTOP 10 FEATURES:")

for feature, importance in importance_rows[:10]:

    print(
        f"{feature:<32} "
        f"{importance:.4f}"
    )


# =========================================================
# 17. PROBABILIDADES
# =========================================================

print("\nPROBABILIDADES EN VALIDATION:")

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
    )

).show(
    truncate=False
)


# =========================================================
# 18. REFERENCIAS DE LOS MODELOS ANTERIORES
# =========================================================

print("\n" + "=" * 70)
print("REFERENCIA VALIDATION")
print("=" * 70)

print(
    "\nLogistic Regression:"
    "\n  AUC ROC = 0.6450"
    "\n  AUC PR  = 0.5357"
    "\n  F1      = 0.583"
)

print(
    "\nRandom Forest:"
    "\n  AUC ROC = 0.6815"
    "\n  AUC PR  = 0.5482"
    "\n  F1      = 0.606"
)

print(
    "\nXGBoost original:"
    "\n  AUC ROC = 0.6674"
    "\n  AUC PR  = 0.5466"
    "\n  F1      = 0.602"
)


# =========================================================
# 19. FIN
# =========================================================

predictions.unpersist()

spark.stop()