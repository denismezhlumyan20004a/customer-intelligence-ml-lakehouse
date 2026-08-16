from pathlib import Path
import json

from pyspark.sql import SparkSession, functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from xgboost.spark import SparkXGBClassifier


# =========================================================
# 0. SPARK
# =========================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-tune-xgboost-temporal")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. PATHS
# =========================================================

TRAIN_PATH = "data/gold/ml_train"

TUNED_MODEL_PATH = "models/xgboost_tuned"
BEST_PARAMS_PATH = "config/best_xgboost_params.json"


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
# 3. CARGAR ÚNICAMENTE TRAIN
#
# No cargamos VALIDATION.
# No cargamos TEST.
# =========================================================

train_full = spark.read.parquet(TRAIN_PATH)


# =========================================================
# 4. CAST A DOUBLE PARA XGBOOST
# =========================================================

for col_name in FEATURE_COLUMNS:

    train_full = train_full.withColumn(
        col_name,
        F.col(col_name).cast("double")
    )

train_full = train_full.withColumn(
    "target_drop_30",
    F.col("target_drop_30").cast("double")
)

train_full = train_full.cache()

train_count = train_full.count()

print("\n" + "=" * 70)
print("XGBOOST TEMPORAL TUNING")
print("=" * 70)

print(f"\nTRAIN total: {train_count:,} observaciones")


# =========================================================
# 5. FOLDS TEMPORALES INTERNOS
#
# Cada validación queda separada del training
# por aproximadamente 3 meses de embargo.
#
# Fold 1
# Train <= 2023-03
# Embargo: 2023-04,05,06
# Validation: 2023-07,08,09
#
# Fold 2
# Train <= 2023-09
# Embargo: 2023-10,11,12
# Validation: 2024-01,02,03
#
# Fold 3
# Train <= 2024-03
# Embargo: 2024-04,05,06
# Validation: 2024-07,08,09
# =========================================================

FOLDS = [
    {
        "name": "FOLD_1",
        "train_end": "2023-03-01",
        "valid_start": "2023-07-01",
        "valid_end": "2023-09-01",
    },
    {
        "name": "FOLD_2",
        "train_end": "2023-09-01",
        "valid_start": "2024-01-01",
        "valid_end": "2024-03-01",
    },
    {
        "name": "FOLD_3",
        "train_end": "2024-03-01",
        "valid_start": "2024-07-01",
        "valid_end": "2024-09-01",
    },
]


# =========================================================
# 6. MOSTRAR TAMAÑOS DE FOLDS
# =========================================================

print("\nFOLDS TEMPORALES:")

for fold in FOLDS:

    fold_train = train_full.filter(
        F.col("month_start") <= F.lit(fold["train_end"])
    )

    fold_valid = train_full.filter(
        (F.col("month_start") >= F.lit(fold["valid_start"])) &
        (F.col("month_start") <= F.lit(fold["valid_end"]))
    )

    train_stats = fold_train.agg(
        F.count("*").alias("n"),
        F.avg("target_drop_30").alias("rate")
    ).first()

    valid_stats = fold_valid.agg(
        F.count("*").alias("n"),
        F.avg("target_drop_30").alias("rate")
    ).first()

    print(
        f"\n{fold['name']}"
        f"\n  Train:      {train_stats['n']:,} obs "
        f"({train_stats['rate'] * 100:.2f}% positivos)"
        f"\n  Validation: {valid_stats['n']:,} obs "
        f"({valid_stats['rate'] * 100:.2f}% positivos)"
    )


# =========================================================
# 7. CONFIGURACIONES A PROBAR
#
# No hacemos una grid enorme.
# Son configuraciones razonables y distintas.
# 8 configs x 3 folds = 24 entrenamientos.
# =========================================================

PARAMETER_SETS = [

    # Config 1 - conservadora
    {
        "name": "XGB_01",
        "n_estimators": 250,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
    },

    # Config 2 - más árboles
    {
        "name": "XGB_02",
        "n_estimators": 500,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
    },

    # Config 3 - baseline parecido al anterior
    {
        "name": "XGB_03",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
    },

    # Config 4 - menos regularizado estructuralmente
    {
        "name": "XGB_04",
        "n_estimators": 350,
        "max_depth": 4,
        "learning_rate": 0.04,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
    },

    # Config 5 - árboles más profundos
    {
        "name": "XGB_05",
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.04,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "scale_pos_weight": 1.0,
    },

    # Config 6 - más regularización
    {
        "name": "XGB_06",
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.04,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 5.0,
        "scale_pos_weight": 1.0,
    },

    # Config 7 - pequeño peso extra a positivos
    {
        "name": "XGB_07",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 2.0,
        "scale_pos_weight": 1.5,
    },

    # Config 8 - shallow + regularizado + positivos
    {
        "name": "XGB_08",
        "n_estimators": 450,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
        "scale_pos_weight": 1.5,
    },
]


# =========================================================
# 8. EVALUADORES
#
# Elegiremos principalmente por AUC-PR.
# AUC-ROC sirve de desempate.
# =========================================================

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


# =========================================================
# 9. TUNING
# =========================================================

all_results = []

for config_index, params in enumerate(
    PARAMETER_SETS,
    start=1
):

    print("\n" + "=" * 70)
    print(
        f"CONFIG {config_index}/{len(PARAMETER_SETS)} "
        f"- {params['name']}"
    )
    print("=" * 70)

    print(
        f"n_estimators={params['n_estimators']}, "
        f"depth={params['max_depth']}, "
        f"lr={params['learning_rate']}, "
        f"min_child={params['min_child_weight']}, "
        f"subsample={params['subsample']}, "
        f"colsample={params['colsample_bytree']}, "
        f"alpha={params['reg_alpha']}, "
        f"lambda={params['reg_lambda']}, "
        f"pos_weight={params['scale_pos_weight']}"
    )

    fold_results = []

    for fold in FOLDS:

        fold_train = train_full.filter(
            F.col("month_start") <=
            F.lit(fold["train_end"])
        )

        fold_valid = train_full.filter(
            (
                F.col("month_start") >=
                F.lit(fold["valid_start"])
            ) &
            (
                F.col("month_start") <=
                F.lit(fold["valid_end"])
            )
        )

        model = SparkXGBClassifier(

            features_col=FEATURE_COLUMNS,
            label_col="target_drop_30",

            num_workers=1,
            device="cpu",
            tree_method="hist",

            eval_metric="logloss",

            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],

            min_child_weight=params["min_child_weight"],

            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],

            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],

            scale_pos_weight=params["scale_pos_weight"],

            random_state=42,
        )

        fitted = model.fit(fold_train)

        predictions = fitted.transform(
            fold_valid
        )

        auc_roc = roc_evaluator.evaluate(
            predictions
        )

        auc_pr = pr_evaluator.evaluate(
            predictions
        )

        fold_results.append({
            "fold": fold["name"],
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
        })

        print(
            f"{fold['name']}: "
            f"AUC-ROC={auc_roc:.4f} | "
            f"AUC-PR={auc_pr:.4f}"
        )


    # -----------------------------------------------------
    # Media de los 3 folds
    # -----------------------------------------------------

    mean_auc_roc = sum(
        x["auc_roc"]
        for x in fold_results
    ) / len(fold_results)

    mean_auc_pr = sum(
        x["auc_pr"]
        for x in fold_results
    ) / len(fold_results)


    # También observamos estabilidad
    auc_pr_values = [
        x["auc_pr"]
        for x in fold_results
    ]

    min_auc_pr = min(auc_pr_values)
    max_auc_pr = max(auc_pr_values)

    auc_pr_range = (
        max_auc_pr -
        min_auc_pr
    )


    result = {
        **params,
        "mean_auc_roc": mean_auc_roc,
        "mean_auc_pr": mean_auc_pr,
        "min_auc_pr": min_auc_pr,
        "auc_pr_range": auc_pr_range,
        "folds": fold_results,
    }

    all_results.append(result)


    print(
        "\nMEDIA:"
        f"\n  AUC-ROC: {mean_auc_roc:.4f}"
        f"\n  AUC-PR:  {mean_auc_pr:.4f}"
        f"\n  Worst fold AUC-PR: {min_auc_pr:.4f}"
        f"\n  AUC-PR range: {auc_pr_range:.4f}"
    )


# =========================================================
# 10. RANKING DE CONFIGURACIONES
#
# Primary metric = mean AUC-PR
# Tie-break = mean AUC-ROC
# =========================================================

all_results = sorted(
    all_results,
    key=lambda x: (
        x["mean_auc_pr"],
        x["mean_auc_roc"]
    ),
    reverse=True
)


print("\n" + "=" * 70)
print("RESULTADO FINAL DEL TUNING")
print("=" * 70)


print(
    "\n"
    f"{'CONFIG':<10}"
    f"{'AUC_PR':<12}"
    f"{'AUC_ROC':<12}"
    f"{'WORST_PR':<12}"
    f"{'PR_RANGE':<12}"
)

for result in all_results:

    print(
        f"{result['name']:<10}"
        f"{result['mean_auc_pr']:<12.4f}"
        f"{result['mean_auc_roc']:<12.4f}"
        f"{result['min_auc_pr']:<12.4f}"
        f"{result['auc_pr_range']:<12.4f}"
    )


# =========================================================
# 11. MEJOR CONFIGURACIÓN
# =========================================================

best = all_results[0]


print("\n" + "=" * 70)
print("MEJOR CONFIGURACIÓN")
print("=" * 70)

print(f"\nNombre: {best['name']}")

print(
    f"Mean AUC-PR:  "
    f"{best['mean_auc_pr']:.4f}"
)

print(
    f"Mean AUC-ROC: "
    f"{best['mean_auc_roc']:.4f}"
)

print(
    f"Worst fold PR: "
    f"{best['min_auc_pr']:.4f}"
)


print("\nPARÁMETROS:")

PARAM_NAMES = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "scale_pos_weight",
]

for param_name in PARAM_NAMES:

    print(
        f"{param_name}: "
        f"{best[param_name]}"
    )


# =========================================================
# 12. GUARDAR PARÁMETROS
# =========================================================

Path("config").mkdir(
    parents=True,
    exist_ok=True
)

params_to_save = {
    param_name: best[param_name]
    for param_name in PARAM_NAMES
}

params_to_save["name"] = best["name"]
params_to_save["mean_auc_pr"] = best[
    "mean_auc_pr"
]
params_to_save["mean_auc_roc"] = best[
    "mean_auc_roc"
]

with open(
    BEST_PARAMS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        params_to_save,
        file,
        indent=4
    )


print(
    f"\nParámetros guardados en: "
    f"{BEST_PARAMS_PATH}"
)


# =========================================================
# 13. REENTRENAR MEJOR XGBOOST SOBRE TODO TRAIN
#
# Ahora usamos TODO TRAIN 2022-06 -> 2024-09.
#
# Seguimos sin tocar VALIDATION.
# =========================================================

print(
    "\nEntrenando mejor configuración "
    "sobre TODO TRAIN..."
)

best_model_estimator = SparkXGBClassifier(

    features_col=FEATURE_COLUMNS,
    label_col="target_drop_30",

    num_workers=1,
    device="cpu",
    tree_method="hist",

    eval_metric="logloss",

    n_estimators=best["n_estimators"],
    max_depth=best["max_depth"],
    learning_rate=best["learning_rate"],

    min_child_weight=best[
        "min_child_weight"
    ],

    subsample=best["subsample"],
    colsample_bytree=best[
        "colsample_bytree"
    ],

    reg_alpha=best["reg_alpha"],
    reg_lambda=best["reg_lambda"],

    scale_pos_weight=best[
        "scale_pos_weight"
    ],

    random_state=42,
)

best_model = best_model_estimator.fit(
    train_full
)

best_model.write() \
    .overwrite() \
    .save(TUNED_MODEL_PATH)


print(
    f"\nModelo tuned guardado en: "
    f"{TUNED_MODEL_PATH}"
)


# =========================================================
# 14. FIN
# =========================================================

train_full.unpersist()

spark.stop()