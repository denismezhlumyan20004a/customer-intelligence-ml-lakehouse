from pyspark.sql import SparkSession, functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-train-random-forest")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


TRAIN_PATH = "data/gold/ml_train"
VALIDATION_PATH = "data/gold/ml_validation"

MODEL_PATH = "models/random_forest"


# ---------------------------------------------------------
# 1. Cargar TRAIN y VALIDATION
#
# TEST sigue sin tocarse.
# ---------------------------------------------------------

train = spark.read.parquet(TRAIN_PATH)
validation = spark.read.parquet(VALIDATION_PATH)


# ---------------------------------------------------------
# 2. Features
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# 3. Pipeline
#
# Random Forest no necesita StandardScaler.
# ---------------------------------------------------------

assembler = VectorAssembler(
    inputCols=FEATURE_COLUMNS,
    outputCol="features"
)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="target_drop_30",

    numTrees=300,
    maxDepth=6,
    minInstancesPerNode=5,

    featureSubsetStrategy="sqrt",

    seed=42
)

pipeline = Pipeline(
    stages=[
        assembler,
        rf
    ]
)


# ---------------------------------------------------------
# 4. Entrenar SOLO con TRAIN
# ---------------------------------------------------------

print("\nEntrenando Random Forest...")

model = pipeline.fit(train)

model.write() \
    .overwrite() \
    .save(MODEL_PATH)


# ---------------------------------------------------------
# 5. Predicciones SOLO sobre VALIDATION
# ---------------------------------------------------------

predictions = (
    model
    .transform(validation)
    .withColumn(
        "probability_drop",
        vector_to_array("probability")[1]
    )
)


# ---------------------------------------------------------
# 6. AUC ROC y AUC PR
# ---------------------------------------------------------

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="target_drop_30",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

pr_evaluator = BinaryClassificationEvaluator(
    labelCol="target_drop_30",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

auc_roc = roc_evaluator.evaluate(predictions)
auc_pr = pr_evaluator.evaluate(predictions)


# ---------------------------------------------------------
# 7. Buscar threshold usando SOLO VALIDATION
# ---------------------------------------------------------

threshold_results = []

for threshold in [
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
]:

    scored = predictions.withColumn(
        "predicted_label",
        (
            F.col("probability_drop") >= threshold
        ).cast("int")
    )

    row = scored.agg(

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
        ).alias("tn")

    ).first()

    tp = row["tp"]
    fp = row["fp"]
    fn = row["fn"]
    tn = row["tn"]

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0 else 0
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


# ---------------------------------------------------------
# 8. Mejor threshold por F1
# ---------------------------------------------------------

best = max(
    threshold_results,
    key=lambda x: x["f1"]
)


# ---------------------------------------------------------
# 9. Feature importances
# ---------------------------------------------------------

rf_model = model.stages[-1]

importances = list(
    zip(
        FEATURE_COLUMNS,
        rf_model.featureImportances.toArray()
    )
)

importances = sorted(
    importances,
    key=lambda x: x[1],
    reverse=True
)


# ---------------------------------------------------------
# 10. Resultados
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("RANDOM FOREST - VALIDATION")
print("=" * 70)

print(f"\nAUC ROC: {auc_roc:.4f}")
print(f"AUC PR:  {auc_pr:.4f}")


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


print("\nMEJOR THRESHOLD EN VALIDATION:")
print(f"Threshold: {best['threshold']:.2f}")
print(f"Precision: {best['precision']:.3f}")
print(f"Recall:    {best['recall']:.3f}")
print(f"F1:        {best['f1']:.3f}")


print("\nMATRIZ DE CONFUSIÓN:")
print(f"TP: {best['tp']}")
print(f"FP: {best['fp']}")
print(f"FN: {best['fn']}")
print(f"TN: {best['tn']}")


print("\nTOP 10 FEATURES:")

for feature, importance in importances[:10]:
    print(
        f"{feature:<32} "
        f"{importance:.4f}"
    )


print("\nPROBABILIDADES EN VALIDATION:")

predictions.select(
    F.min("probability_drop").alias("min_probability"),
    F.avg("probability_drop").alias("avg_probability"),
    F.max("probability_drop").alias("max_probability")
).show(truncate=False)


spark.stop()