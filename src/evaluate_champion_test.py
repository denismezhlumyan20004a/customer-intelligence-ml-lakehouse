from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window

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
    .appName("retailco-evaluate-champion-test")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. PATHS
# =========================================================

TRAIN_PATH = "data/gold/ml_train_v2"
VALIDATION_PATH = "data/gold/ml_validation_v2"
TEST_PATH = "data/gold/ml_test_v2"

FINAL_MODEL_PATH = "models/random_forest_v2_final"


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
# 4. CONFIGURACIÓN CONGELADA
#
# Champion decidido en VALIDATION:
#
# Random Forest V2
#
# Hiperparámetros:
# numTrees = 300
# maxDepth = 6
# minInstancesPerNode = 5
# featureSubsetStrategy = sqrt
#
# Política principal:
# Top 25 clientes / mes
#
# Política ampliada:
# Top 30 clientes / mes
#
# Threshold secundario:
# 0.30
#
# NADA se seleccionará usando TEST.
# =========================================================

FIXED_THRESHOLD = 0.30


# =========================================================
# 5. CARGAR TRAIN + VALIDATION
#
# VALIDATION ya cumplió su función de selección.
# Ahora puede incorporarse al entrenamiento final.
# =========================================================

train = spark.read.parquet(
    TRAIN_PATH
)

validation = spark.read.parquet(
    VALIDATION_PATH
)

train_final = (
    train
    .unionByName(validation)
)


# =========================================================
# 6. CARGAR TEST
#
# ESTE ES EL PRIMER MOMENTO EN QUE TEST SE UTILIZA
# PARA EVALUAR EL CHAMPION FINAL.
# =========================================================

test = spark.read.parquet(
    TEST_PATH
)


print("\n" + "=" * 70)
print("FINAL CHAMPION TEST")
print("=" * 70)

print(
    f"\nFeatures: "
    f"{len(FEATURE_COLUMNS)}"
)


# =========================================================
# 7. CAST NUMÉRICO
# =========================================================

for column_name in FEATURE_COLUMNS:

    train_final = train_final.withColumn(
        column_name,
        F.col(column_name).cast("double")
    )

    test = test.withColumn(
        column_name,
        F.col(column_name).cast("double")
    )


train_final = train_final.withColumn(
    "target_drop_30",
    F.col("target_drop_30").cast("double")
)

test = test.withColumn(
    "target_drop_30",
    F.col("target_drop_30").cast("double")
)


# =========================================================
# 8. COMPROBAR DATASETS
# =========================================================

train_final_count = train_final.count()
test_count = test.count()


print(
    f"Final training observations: "
    f"{train_final_count:,}"
)

print(
    f"TEST observations:           "
    f"{test_count:,}"
)


print(
    "\nPeriodo TEST:"
)

test.select(
    F.min("month_start").alias("min_month"),
    F.max("month_start").alias("max_month")
).show(
    truncate=False
)


# =========================================================
# 9. VECTOR ASSEMBLER
# =========================================================

assembler = VectorAssembler(
    inputCols=FEATURE_COLUMNS,
    outputCol="features",
    handleInvalid="error"
)


# =========================================================
# 10. RANDOM FOREST FINAL
#
# HIPERPARÁMETROS COMPLETAMENTE CONGELADOS.
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
# 11. PIPELINE
# =========================================================

pipeline = Pipeline(
    stages=[
        assembler,
        rf,
    ]
)


# =========================================================
# 12. ENTRENAMIENTO FINAL
#
# TRAIN + VALIDATION.
#
# TEST NO participa.
# =========================================================

print(
    "\nEntrenando Random Forest V2 FINAL "
    "con TRAIN + VALIDATION..."
)

final_model = pipeline.fit(
    train_final
)


# =========================================================
# 13. GUARDAR MODELO FINAL
# =========================================================

final_model.write() \
    .overwrite() \
    .save(FINAL_MODEL_PATH)


print(
    f"\nModelo final guardado en: "
    f"{FINAL_MODEL_PATH}"
)


# =========================================================
# 14. SCORE TEST
# =========================================================

predictions = (
    final_model
    .transform(test)
    .withColumn(
        "probability_drop",
        vector_to_array("probability")[1]
    )
    .cache()
)


predictions.count()


# =========================================================
# 15. AUC ROC
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
# 16. AUC PR
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
# 17. THRESHOLD FIJO
#
# NO buscamos el mejor threshold en TEST.
#
# Usamos exactamente 0.30,
# elegido previamente en VALIDATION.
# =========================================================

threshold_predictions = predictions.withColumn(

    "predicted_label",

    (
        F.col("probability_drop")
        >=
        F.lit(FIXED_THRESHOLD)
    ).cast("int")
)


confusion = (
    threshold_predictions
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


tp = confusion["tp"]
fp = confusion["fp"]
fn = confusion["fn"]
tn = confusion["tn"]


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
    /
    (precision + recall)

    if (precision + recall) > 0

    else 0.0
)


# =========================================================
# 18. TEST BASE RATE
# =========================================================

test_stats = (
    predictions
    .agg(

        F.count("*")
        .alias("observations"),

        F.sum(
            "target_drop_30"
        )
        .alias("positives"),

        F.avg(
            "target_drop_30"
        )
        .alias("base_rate"),
    )
    .first()
)


# =========================================================
# 19. RANKING MENSUAL
# =========================================================

ranking_window = (
    Window
    .partitionBy(
        "month_start"
    )
    .orderBy(
        F.desc("probability_drop"),
        F.asc("customer_id")
    )
)


ranked = predictions.withColumn(
    "risk_rank",
    F.row_number().over(
        ranking_window
    )
)


# =========================================================
# 20. TOTAL POR MES
# =========================================================

monthly_totals = (
    ranked

    .groupBy(
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
        "base_rate",

        F.col("actual_positives")
        /
        F.col("eligible_customers")
    )
)


# =========================================================
# 21. FUNCIÓN @K
# =========================================================

def calculate_at_k(
    ranked_df,
    k
):

    selected = (
        ranked_df

        .filter(
            F.col("risk_rank")
            <=
            F.lit(k)
        )

        .groupBy(
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
                "true_positives"
            ),
        )
    )


    result = (
        selected

        .join(
            monthly_totals,
            "month_start",
            "inner"
        )

        .withColumn(
            "k",
            F.lit(k)
        )

        .withColumn(
            "precision_at_k",

            F.col("true_positives")
            /
            F.col("selected_customers")
        )

        .withColumn(
            "recall_at_k",

            F.col("true_positives")
            /
            F.col("actual_positives")
        )

        .withColumn(
            "lift_at_k",

            F.col("precision_at_k")
            /
            F.col("base_rate")
        )
    )


    return result


# =========================================================
# 22. TOP 25 / TOP 30
# =========================================================

metrics_25 = calculate_at_k(
    ranked,
    25
)

metrics_30 = calculate_at_k(
    ranked,
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
# 23. RESULTADOS PRINCIPALES
# =========================================================

print("\n" + "=" * 70)
print("RANDOM FOREST V2 FINAL - OUT-OF-TIME TEST")
print("=" * 70)


print(
    f"\nTEST observations: "
    f"{test_stats['observations']:,}"
)

print(
    f"TEST positives: "
    f"{int(test_stats['positives'])}"
)

print(
    f"TEST base rate: "
    f"{test_stats['base_rate'] * 100:.2f}%"
)


print(
    f"\nAUC ROC: "
    f"{auc_roc:.4f}"
)

print(
    f"AUC PR:  "
    f"{auc_pr:.4f}"
)


# =========================================================
# 24. THRESHOLD FIJO 0.30
# =========================================================

print(
    "\nTHRESHOLD FIJO = "
    f"{FIXED_THRESHOLD:.2f}"
)


print(
    f"Precision: "
    f"{precision:.3f}"
)

print(
    f"Recall:    "
    f"{recall:.3f}"
)

print(
    f"F1:        "
    f"{f1:.3f}"
)


print(
    "\nMATRIZ DE CONFUSIÓN:"
)

print(
    f"TP: {tp}"
)

print(
    f"FP: {fp}"
)

print(
    f"FN: {fn}"
)

print(
    f"TN: {tn}"
)


# =========================================================
# 25. CANDIDATOS POR MES
# =========================================================

print("\n" + "=" * 70)
print("TEST - CANDIDATOS POR MES")
print("=" * 70)


(
    monthly_totals

    .orderBy(
        "month_start"
    )

    .show(
        100,
        truncate=False
    )
)


# =========================================================
# 26. RESULTADOS @25 / @30 POR MES
# =========================================================

print("\n" + "=" * 70)
print("TEST - PRECISION / RECALL / LIFT POR MES")
print("=" * 70)


(
    ranking_metrics

    .select(
        "month_start",
        "k",

        "eligible_customers",
        "actual_positives",

        "selected_customers",
        "true_positives",

        "precision_at_k",
        "recall_at_k",
        "lift_at_k"
    )

    .orderBy(
        "month_start",
        "k"
    )

    .show(
        100,
        truncate=False
    )
)


# =========================================================
# 27. RESUMEN GLOBAL
# =========================================================

global_summary = (
    ranking_metrics

    .groupBy(
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

        F.sum(
            "selected_customers"
        )
        .alias(
            "total_selected"
        ),

        F.sum(
            "true_positives"
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
print("FINAL OUT-OF-TIME RANKING SUMMARY")
print("=" * 70)


(
    global_summary

    .select(
        "k",
        "months",

        "avg_precision_at_k",
        "avg_recall_at_k",
        "avg_lift_at_k",

        "global_precision_at_k",
        "global_recall_at_k",
        "global_base_rate",
        "global_lift_at_k",

        "total_selected",
        "total_true_positives",
        "total_actual_positives",
    )

    .orderBy(
        "k"
    )

    .show(
        truncate=False
    )
)


# =========================================================
# 28. TOP 25 DEL ÚLTIMO MES DE TEST
#
# Solo para inspección del ranking final histórico.
# =========================================================

last_test_month = (
    ranked
    .agg(
        F.max("month_start")
        .alias("last_month")
    )
    .first()["last_month"]
)


print(
    "\nÚLTIMO MES TEST:",
    last_test_month
)


print(
    "\nTOP 25 CHAMPION - ÚLTIMO MES TEST:"
)


(
    ranked

    .filter(
        (
            F.col("month_start")
            ==
            F.lit(last_test_month)
        )
        &
        (
            F.col("risk_rank")
            <=
            25
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
        25,
        truncate=False
    )
)


# =========================================================
# 29. FEATURE IMPORTANCE FINAL
# =========================================================

rf_model = final_model.stages[1]

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
    "\nTOP 15 FEATURES - MODELO FINAL:"
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
# 30. RECORDATORIO METODOLÓGICO
# =========================================================

print("\n" + "=" * 70)
print("METODOLOGÍA")
print("=" * 70)


print(
    "\nChampion seleccionado previamente: "
    "Random Forest V2"
)

print(
    "Selección realizada únicamente con VALIDATION 2025."
)

print(
    "Hiperparámetros congelados antes de abrir TEST."
)

print(
    "Política operativa congelada: "
    "Top 25 principal / Top 30 ampliado."
)

print(
    "TEST 2026 utilizado una única vez "
    "para evaluación final out-of-time."
)


# =========================================================
# 31. FIN
# =========================================================

ranking_metrics.unpersist()
predictions.unpersist()

spark.stop()