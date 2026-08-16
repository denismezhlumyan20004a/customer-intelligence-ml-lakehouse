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
    .appName("retailco-train-xgboost-v2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. PATHS
# =========================================================

TRAIN_PATH = "data/gold/ml_train_v2"
VALIDATION_PATH = "data/gold/ml_validation_v2"

MODEL_PATH = "models/xgboost_v2"


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
# 3. FEATURES NUEVAS V2
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
# TEST NO se carga.
# =========================================================

train = spark.read.parquet(
    TRAIN_PATH
)

validation = spark.read.parquet(
    VALIDATION_PATH
)


print("\n" + "=" * 70)
print("XGBOOST V2")
print("=" * 70)

print(
    f"\nXGBoost version: "
    f"{xgboost.__version__}"
)

print(
    f"Features utilizadas: "
    f"{len(FEATURE_COLUMNS)}"
)


# =========================================================
# 6. CAST A DOUBLE
#
# SparkXGBClassifier necesita features integral,
# float o double cuando se pasan como lista.
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
# 8. XGBOOST
#
# EXACTAMENTE la configuración ganadora del tuning
# temporal interno realizado SOLO dentro de TRAIN.
#
# XGB_01:
#
# n_estimators      = 250
# max_depth         = 3
# learning_rate     = 0.05
# min_child_weight  = 5
# subsample         = 0.8
# colsample_bytree  = 0.8
# reg_alpha         = 0
# reg_lambda        = 1
# scale_pos_weight  = 1
#
# Solo cambiamos el feature set V1 -> V2.
# =========================================================

xgb = SparkXGBClassifier(

    features_col=FEATURE_COLUMNS,
    label_col="target_drop_30",

    num_workers=1,
    device="cpu",
    tree_method="hist",

    eval_metric="logloss",

    n_estimators=250,
    max_depth=3,
    learning_rate=0.05,

    min_child_weight=5,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=0.0,
    reg_lambda=1.0,

    scale_pos_weight=1.0,

    random_state=42,
)


# =========================================================
# 9. ENTRENAR SOLO CON TRAIN
# =========================================================

print("\nEntrenando XGBoost V2...")

model = xgb.fit(
    train
)


# =========================================================
# 10. GUARDAR MODELO
# =========================================================

model.write() \
    .overwrite() \
    .save(MODEL_PATH)


print(
    f"\nModelo guardado en: "
    f"{MODEL_PATH}"
)


# =========================================================
# 11. PREDICCIONES SOBRE VALIDATION
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
# 12. AUC ROC
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
# 13. AUC PR
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
# 14. THRESHOLDS
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
# 15. EVALUAR CADA THRESHOLD
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
# 16. MEJOR THRESHOLD SEGÚN F1
# =========================================================

best = max(
    threshold_results,
    key=lambda x: x["f1"]
)


# =========================================================
# 17. FEATURE IMPORTANCE
# =========================================================

raw_importances = (
    model.get_feature_importances(
        importance_type="gain"
    )
)


importance_rows = []


for feature, importance in raw_importances.items():

    # XGBoost puede devolver:
    # f0, f1, f2...
    if (
        feature.startswith("f")
        and feature[1:].isdigit()
    ):

        index = int(
            feature[1:]
        )

        if index < len(FEATURE_COLUMNS):

            feature_name = (
                FEATURE_COLUMNS[index]
            )

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
# 18. RESULTADOS
# =========================================================

print("\n" + "=" * 70)
print("XGBOOST V2 - VALIDATION")
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
# 19. TABLA DE THRESHOLDS
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
# 20. MEJOR THRESHOLD
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
# 21. MATRIZ DE CONFUSIÓN
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
# 22. TOP FEATURES
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
# 23. IMPORTANCIA DE LAS NUEVAS FEATURES
#
# En XGBoost gain no está necesariamente normalizado
# a suma = 1.
#
# Por eso calculamos el porcentaje relativo sobre la
# suma de gains obtenidos.
# =========================================================

total_gain = sum(
    importance
    for _, importance
    in importance_rows
)


new_feature_gain = sum(

    importance

    for feature, importance
    in importance_rows

    if feature in NEW_FEATURE_COLUMNS
)


base_feature_gain = sum(

    importance

    for feature, importance
    in importance_rows

    if feature in BASE_FEATURE_COLUMNS
)


print(
    "\nIMPORTANCIA AGREGADA (GAIN):"
)


print(
    f"Gain total: "
    f"{total_gain:.4f}"
)

print(
    f"Features V1: "
    f"{base_feature_gain:.4f}"
)

print(
    f"Features nuevas V2: "
    f"{new_feature_gain:.4f}"
)


if total_gain > 0:

    print(
        f"% gain features V2: "
        f"{new_feature_gain / total_gain * 100:.2f}%"
    )


# =========================================================
# 24. PROBABILIDADES
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
# 25. COMPARACIÓN
# =========================================================

print("\n" + "=" * 70)
print("COMPARACIÓN DE MODELOS - VALIDATION")
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
    "  AUC ROC = 0.6840"
)

print(
    "  AUC PR  = 0.5547"
)

print(
    "  F1      = 0.610"
)


print(
    "\nXGBoost tuned V1:"
)

print(
    "  AUC ROC = 0.6755"
)

print(
    "  AUC PR  = 0.5617"
)

print(
    "  F1      = 0.599"
)


print(
    "\nXGBoost V2:"
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


# =========================================================
# 26. DELTA XGBOOST V2 VS TUNED V1
# =========================================================

print(
    "\nDIFERENCIA XGBOOST V2 - TUNED V1:"
)


print(
    f"  Delta AUC ROC = "
    f"{auc_roc - 0.6755:+.4f}"
)

print(
    f"  Delta AUC PR  = "
    f"{auc_pr - 0.5617:+.4f}"
)

print(
    f"  Delta F1      = "
    f"{best['f1'] - 0.599:+.3f}"
)


# =========================================================
# 27. FIN
# =========================================================

predictions.unpersist()

spark.stop()