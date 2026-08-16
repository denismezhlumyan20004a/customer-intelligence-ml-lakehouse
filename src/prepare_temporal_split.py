from datetime import date

from pyspark.sql import SparkSession, functions as F


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-prepare-temporal-split")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

INPUT_PATH = "data/gold/ml_features"

OUTPUT_ALL = "data/gold/ml_features_split"
OUTPUT_TRAIN = "data/gold/ml_train"
OUTPUT_VALIDATION = "data/gold/ml_validation"
OUTPUT_TEST = "data/gold/ml_test"


df = spark.read.parquet(INPUT_PATH)


# ---------------------------------------------------------
# 1. Fechas del split temporal
# ---------------------------------------------------------

TRAIN_END = date(2024, 9, 1)

VALIDATION_START = date(2025, 1, 1)
VALIDATION_END = date(2025, 9, 1)

TEST_START = date(2026, 1, 1)
TEST_END = date(2026, 4, 1)


# ---------------------------------------------------------
# 2. Asignar split
#
# Los meses que quedan entre conjuntos son EMBARGO.
# Esto evita contaminación por el horizonte futuro
# de 3 meses usado para construir el target.
# ---------------------------------------------------------

split_df = (
    df
    .withColumn(
        "split",
        F.when(
            F.col("month_start") <= F.lit(TRAIN_END),
            F.lit("TRAIN")
        )
        .when(
            (
                F.col("month_start") >=
                F.lit(VALIDATION_START)
            ) &
            (
                F.col("month_start") <=
                F.lit(VALIDATION_END)
            ),
            F.lit("VALIDATION")
        )
        .when(
            (
                F.col("month_start") >=
                F.lit(TEST_START)
            ) &
            (
                F.col("month_start") <=
                F.lit(TEST_END)
            ),
            F.lit("TEST")
        )
        .otherwise(F.lit("EMBARGO"))
    )
)


# ---------------------------------------------------------
# 3. Crear datasets
# ---------------------------------------------------------

train = split_df.filter(
    F.col("split") == "TRAIN"
)

validation = split_df.filter(
    F.col("split") == "VALIDATION"
)

test = split_df.filter(
    F.col("split") == "TEST"
)


# ---------------------------------------------------------
# 4. Guardar
# ---------------------------------------------------------

split_df.write \
    .mode("overwrite") \
    .parquet(OUTPUT_ALL)

train.write \
    .mode("overwrite") \
    .parquet(OUTPUT_TRAIN)

validation.write \
    .mode("overwrite") \
    .parquet(OUTPUT_VALIDATION)

test.write \
    .mode("overwrite") \
    .parquet(OUTPUT_TEST)


# ---------------------------------------------------------
# 5. Validación general
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TEMPORAL SPLIT CREADO")
print("=" * 70)

print("\nDISTRIBUCIÓN:")

split_df.groupBy("split") \
    .agg(
        F.count("*").alias("observaciones"),
        F.countDistinct("customer_id").alias("clientes"),
        F.sum("target_drop_30").alias("positivos"),
        (
            F.avg("target_drop_30") * 100
        ).alias("positive_rate_pct"),
        F.min("month_start").alias("fecha_min"),
        F.max("month_start").alias("fecha_max")
    ) \
    .orderBy("fecha_min") \
    .show(truncate=False)


# ---------------------------------------------------------
# 6. Comprobar cada conjunto
# ---------------------------------------------------------

for name, dataset in [
    ("TRAIN", train),
    ("VALIDATION", validation),
    ("TEST", test),
]:

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    dataset.select(
        F.count("*").alias("observaciones"),
        F.countDistinct("customer_id").alias("clientes"),
        F.sum("target_drop_30").alias("positivos"),
        (
            F.avg("target_drop_30") * 100
        ).alias("positive_rate_pct"),
        F.min("month_start").alias("fecha_min"),
        F.max("month_start").alias("fecha_max")
    ).show(truncate=False)


# ---------------------------------------------------------
# 7. Verificación de separación temporal
# ---------------------------------------------------------

print("\nVERIFICACIÓN TEMPORAL:")

print(
    "TRAIN max:",
    train.agg(F.max("month_start")).first()[0]
)

print(
    "VALIDATION min:",
    validation.agg(F.min("month_start")).first()[0]
)

print(
    "VALIDATION max:",
    validation.agg(F.max("month_start")).first()[0]
)

print(
    "TEST min:",
    test.agg(F.min("month_start")).first()[0]
)

spark.stop()