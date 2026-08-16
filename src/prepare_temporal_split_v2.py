from pyspark.sql import SparkSession, functions as F


# =========================================================
# 0. SPARK
# =========================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-prepare-temporal-split-v2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. PATHS
# =========================================================

INPUT_PATH = "data/gold/ml_features_v2"

OUTPUT_SPLIT_PATH = "data/gold/ml_features_v2_split"
OUTPUT_TRAIN_PATH = "data/gold/ml_train_v2"
OUTPUT_VALIDATION_PATH = "data/gold/ml_validation_v2"
OUTPUT_TEST_PATH = "data/gold/ml_test_v2"


# =========================================================
# 2. CARGAR FEATURES V2
# =========================================================

df = (
    spark.read.parquet(INPUT_PATH)
    .withColumn(
        "month_start",
        F.col("month_start").cast("date")
    )
)


print("\n" + "=" * 70)
print("TEMPORAL SPLIT V2")
print("=" * 70)

print(
    f"\nObservaciones totales: "
    f"{df.count():,}"
)


# =========================================================
# 3. SPLIT TEMPORAL
#
# EXACTAMENTE EL MISMO QUE V1
#
# TRAIN:
#   2022-06 -> 2024-09
#
# EMBARGO:
#   2024-10 -> 2024-12
#
# VALIDATION:
#   2025-01 -> 2025-09
#
# EMBARGO:
#   2025-10 -> 2025-12
#
# TEST:
#   2026-01 -> 2026-04
#
# La separación de 3 meses protege nuestro target
# forward-looking de 3 meses.
# =========================================================

df = df.withColumn(

    "split",

    F.when(
        F.col("month_start") <= F.lit("2024-09-01"),
        F.lit("TRAIN")
    )

    .when(
        (
            F.col("month_start") >=
            F.lit("2024-10-01")
        )
        &
        (
            F.col("month_start") <=
            F.lit("2024-12-01")
        ),
        F.lit("EMBARGO")
    )

    .when(
        (
            F.col("month_start") >=
            F.lit("2025-01-01")
        )
        &
        (
            F.col("month_start") <=
            F.lit("2025-09-01")
        ),
        F.lit("VALIDATION")
    )

    .when(
        (
            F.col("month_start") >=
            F.lit("2025-10-01")
        )
        &
        (
            F.col("month_start") <=
            F.lit("2025-12-01")
        ),
        F.lit("EMBARGO")
    )

    .when(
        (
            F.col("month_start") >=
            F.lit("2026-01-01")
        )
        &
        (
            F.col("month_start") <=
            F.lit("2026-04-01")
        ),
        F.lit("TEST")
    )

    .otherwise(
        F.lit("UNASSIGNED")
    )
)


# =========================================================
# 4. COMPROBAR QUE NO HAY FILAS SIN ASIGNAR
# =========================================================

unassigned_count = (
    df
    .filter(
        F.col("split") == "UNASSIGNED"
    )
    .count()
)

print(
    f"\nFilas UNASSIGNED: "
    f"{unassigned_count}"
)


if unassigned_count > 0:

    print(
        "\nFechas UNASSIGNED:"
    )

    (
        df
        .filter(
            F.col("split") == "UNASSIGNED"
        )
        .groupBy("month_start")
        .count()
        .orderBy("month_start")
        .show(
            100,
            truncate=False
        )
    )

    raise ValueError(
        "Hay observaciones fuera del split temporal esperado."
    )


# =========================================================
# 5. RESUMEN POR SPLIT
# =========================================================

summary = (
    df
    .groupBy("split")
    .agg(

        F.count("*")
        .alias("observations"),

        F.countDistinct(
            "customer_id"
        )
        .alias("customers"),

        F.sum(
            "target_drop_30"
        )
        .alias("positives"),

        F.avg(
            "target_drop_30"
        )
        .alias("positive_rate"),

        F.min(
            "month_start"
        )
        .alias("min_month"),

        F.max(
            "month_start"
        )
        .alias("max_month"),
    )
)


print("\nRESUMEN SPLIT V2:")

(
    summary
    .orderBy(
        F.when(
            F.col("split") == "TRAIN",
            1
        )
        .when(
            F.col("split") == "VALIDATION",
            2
        )
        .when(
            F.col("split") == "TEST",
            3
        )
        .when(
            F.col("split") == "EMBARGO",
            4
        )
        .otherwise(5)
    )
    .show(
        truncate=False
    )
)


# =========================================================
# 6. EXTRAER DATASETS
# =========================================================

train = (
    df
    .filter(
        F.col("split") == "TRAIN"
    )
    .drop("split")
)

validation = (
    df
    .filter(
        F.col("split") == "VALIDATION"
    )
    .drop("split")
)

test = (
    df
    .filter(
        F.col("split") == "TEST"
    )
    .drop("split")
)

embargo = (
    df
    .filter(
        F.col("split") == "EMBARGO"
    )
    .drop("split")
)


# =========================================================
# 7. VALIDACIÓN CONTRA SPLIT V1
#
# Como V2 mantiene exactamente las mismas observaciones
# que V1, estos counts DEBEN coincidir.
# =========================================================

EXPECTED_COUNTS = {
    "TRAIN": 2986,
    "VALIDATION": 867,
    "TEST": 322,
    "EMBARGO": 534,
}


actual_counts = {
    "TRAIN": train.count(),
    "VALIDATION": validation.count(),
    "TEST": test.count(),
    "EMBARGO": embargo.count(),
}


print("\nVALIDACIÓN CONTRA V1:")

for split_name in [
    "TRAIN",
    "VALIDATION",
    "TEST",
    "EMBARGO",
]:

    actual = actual_counts[
        split_name
    ]

    expected = EXPECTED_COUNTS[
        split_name
    ]

    status = (
        "OK"
        if actual == expected
        else "ERROR"
    )

    print(
        f"{split_name:<12}"
        f"actual={actual:<6}"
        f"esperado={expected:<6}"
        f"{status}"
    )


for split_name in EXPECTED_COUNTS:

    if (
        actual_counts[split_name]
        !=
        EXPECTED_COUNTS[split_name]
    ):

        raise ValueError(
            f"El split V2 no coincide con V1 "
            f"para {split_name}."
        )


# =========================================================
# 8. COMPROBAR DISTRIBUCIÓN TARGET
# =========================================================

print("\nTARGET POR SPLIT:")

for split_name, split_df in [

    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
    ("EMBARGO", embargo),

]:

    stats = (
        split_df
        .agg(

            F.count("*")
            .alias("n"),

            F.sum(
                "target_drop_30"
            )
            .alias("positive"),

            F.avg(
                "target_drop_30"
            )
            .alias("rate"),

        )
        .first()
    )

    print(
        f"{split_name:<12}"
        f"obs={stats['n']:<6}"
        f"positivos={int(stats['positive']):<6}"
        f"rate={stats['rate'] * 100:.3f}%"
    )


# =========================================================
# 9. GUARDAR SPLIT COMPLETO
# =========================================================

(
    df
    .write
    .mode("overwrite")
    .parquet(
        OUTPUT_SPLIT_PATH
    )
)


# =========================================================
# 10. GUARDAR TRAIN
# =========================================================

(
    train
    .write
    .mode("overwrite")
    .parquet(
        OUTPUT_TRAIN_PATH
    )
)


# =========================================================
# 11. GUARDAR VALIDATION
# =========================================================

(
    validation
    .write
    .mode("overwrite")
    .parquet(
        OUTPUT_VALIDATION_PATH
    )
)


# =========================================================
# 12. GUARDAR TEST
#
# Lo guardamos, pero todavía NO lo utilizamos
# para entrenamiento, tuning ni selección del modelo.
# =========================================================

(
    test
    .write
    .mode("overwrite")
    .parquet(
        OUTPUT_TEST_PATH
    )
)


# =========================================================
# 13. RESULTADO
# =========================================================

print("\n" + "=" * 70)
print("SPLIT V2 COMPLETADO")
print("=" * 70)

print(
    f"\nSplit completo: "
    f"{OUTPUT_SPLIT_PATH}"
)

print(
    f"TRAIN:          "
    f"{OUTPUT_TRAIN_PATH}"
)

print(
    f"VALIDATION:     "
    f"{OUTPUT_VALIDATION_PATH}"
)

print(
    f"TEST:           "
    f"{OUTPUT_TEST_PATH}"
)

print(
    "\nTEST sigue reservado y no se utilizará "
    "hasta congelar el modelo champion."
)


# =========================================================
# 14. FIN
# =========================================================

spark.stop()