from pyspark.sql import SparkSession, functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

from xgboost.spark import SparkXGBClassifier

import xgboost


# =========================================================
# 0. SPARK
# =========================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-train-xgboost")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. PATHS
# =========================================================

TRAIN_PATH = "data/gold/ml_train"
VALIDATION_PATH = "data/gold/ml_validation"

MODEL_PATH = "models/xgboost"


# =========================================================
# 2. CARGAR TRAIN Y VALIDATION
#
# TEST sigue completamente sin tocarse.
# =========================================================

train = spark.read.parquet(TRAIN_PATH)
validation = spark.read.parquet(VALIDATION_PATH)

print(f"\nXGBoost version: {xgboost.__version__}")


# =========================================================
# 3. FEATURES
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
# 4. CONVERTIR FEATURES A DOUBLE
#
# XGBoost Spark no acepta DecimalType cuando features_col
# es una lista de columnas.
# =========================================================

for col_name in FEATURE_COLUMNS:

    train = train.withColumn(
        col_name,
        F.col(col_name).cast("double")
    )

    validation = validation.withColumn(
        col_name,
        F.col(col_name).cast("double")
    )


# También dejamos el label explícitamente como double
train = train.withColumn(
    "target_drop_30",
    F.col("target_drop_30").cast("double")
)

validation = validation.withColumn(
    "target_drop_30",
    F.col("target_drop_30").cast("double")
)


# =========================================================
# 5. COMPROBACIÓN DE TIPOS
# =========================================================

print("\nTipos de columnas preparados para XGBoost:")

train.select(
    FEATURE_COLUMNS + ["target_drop_30"]
).printSchema()


# =========================================================
# 6. MODELO XGBOOST
#
# No ponemos objective manualmente:
# SparkXGBClassifier gestiona la clasificación.
# =========================================================

xgb = SparkXGBClassifier(

    features_col=FEATURE_COLUMNS,
    label_col="target_drop_30",

    num_workers=1,
    device="cpu",

    eval_metric="logloss",

    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=5,

    reg_alpha=0.0,
    reg_lambda=1.0,

    random_state=42,
)


# =========================================================
# 7. ENTRENAR SOLO CON TRAIN
# =========================================================

print("\nEntrenando XGBoost...")

model = xgb.fit(train)


# =========================================================
# 8. GUARDAR MODELO
# =========================================================

model.write() \
    .overwrite() \
    .save(MODEL_PATH)


# =========================================================
# 9. PREDICCIONES SOLO SOBRE VALIDATION
# =========================================================

predictions = (
    model
    .transform(validation)
    .withColumn(
        "probability_drop",
        vector_to_array("probability")[1]
    )
)


# =========================================================
# 10. AUC ROC
# =========================================================

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="target_drop_30",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc_roc = roc_evaluator.evaluate(predictions)


# =========================================================
# 11. AUC PR
# =========================================================

pr_evaluator = BinaryClassificationEvaluator(
    labelCol="target_drop_30",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

auc_pr = pr_evaluator.evaluate(predictions)


# =========================================================
# 12. BUSCAR MEJOR THRESHOLD EN VALIDATION
#
# TEST NO se utiliza para elegir threshold.
# =========================================================

threshold_results = []

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


for threshold in thresholds:

    scored = predictions.withColumn(
        "predicted_label",
        (
            F.col("probability_drop") >= threshold
        ).cast("int")
    )

    row = (
        scored
        .agg(

            # TRUE POSITIVES
            F.sum(
                F.when(
                    (F.col("predicted_label") == 1) &
                    (F.col("target_drop_30") == 1),
                    1
                ).otherwise(0)
            ).alias("tp"),

            # FALSE POSITIVES
            F.sum(
                F.when(
                    (F.col("predicted_label") == 1) &
                    (F.col("target_drop_30") == 0),
                    1
                ).otherwise(0)
            ).alias("fp"),

            # FALSE NEGATIVES
            F.sum(
                F.when(
                    (F.col("predicted_label") == 0) &
                    (F.col("target_drop_30") == 1),
                    1
                ).otherwise(0)
            ).alias("fn"),

            # TRUE NEGATIVES
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
        2 * precision * recall /
        (precision + recall)
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
# 13. MEJOR THRESHOLD SEGÚN F1
# =========================================================

best = max(
    threshold_results,
    key=lambda x: x["f1"]
)


# =========================================================
# 14. FEATURE IMPORTANCE
# =========================================================

importances = model.get_feature_importances(
    importance_type="gain"
)

importance_rows = []


for feature, importance in importances.items():

    # XGBoost puede devolver nombres tipo:
    # f0, f1, f2...
    if (
        feature.startswith("f")
        and feature[1:].isdigit()
    ):

        feature_index = int(feature[1:])

        if feature_index < len(FEATURE_COLUMNS):
            feature_name = FEATURE_COLUMNS[feature_index]
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
# 15. RESULTADOS
# =========================================================

print("\n" + "=" * 70)
print("XGBOOST - VALIDATION")
print("=" * 70)


print(f"\nAUC ROC: {auc_roc:.4f}")
print(f"AUC PR:  {auc_pr:.4f}")


# =========================================================
# 16. TABLA DE THRESHOLDS
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
# 17. MEJOR THRESHOLD
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


# =========================================================
# 18. MATRIZ DE CONFUSIÓN
# =========================================================

print("\nMATRIZ DE CONFUSIÓN:")

print(f"TP: {best['tp']}")
print(f"FP: {best['fp']}")
print(f"FN: {best['fn']}")
print(f"TN: {best['tn']}")


# =========================================================
# 19. TOP FEATURES
# =========================================================

print("\nTOP 10 FEATURES:")


for feature, importance in importance_rows[:10]:

    print(
        f"{feature:<32} "
        f"{importance:.4f}"
    )


# =========================================================
# 20. DISTRIBUCIÓN DE PROBABILIDADES
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
# 21. FIN
# =========================================================

spark.stop()