from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-build-silver")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

RAW_DIR = Path("data/raw/aqua")
OUTPUT_PATH = "data/silver/transactions"

CSV_SCHEMA = """
    Fecha STRING,
    Operacion STRING,
    Documento STRING,
    Cliente STRING,
    Producto STRING,
    Unidades STRING,
    Venta STRING
"""

columns = [
    "Fecha",
    "Operacion",
    "Documento",
    "Cliente",
    "Producto",
    "Unidades",
    "Venta",
]

frames = []

# ------------------------------------------------------------------
# 1. Leer cada CSV conservando el número original de cada fila
# ------------------------------------------------------------------

for file_path in sorted(RAW_DIR.rglob("*.csv")):

    lines_rdd = (
        spark.sparkContext
        .textFile(file_path.as_posix())
        .zipWithIndex()
        .map(lambda x: (int(x[1]), x[0]))
    )

    lines = (
        spark.createDataFrame(
            lines_rdd,
            ["source_row_number", "raw_line"]
        )
        .filter(F.col("source_row_number") > 0)  # quitar cabecera
    )

    parsed = (
        lines
        .select(
            "source_row_number",
            F.from_csv(
                F.col("raw_line"),
                CSV_SCHEMA,
                {"sep": ";"}
            ).alias("csv")
        )
        .select(
            "source_row_number",
            "csv.*"
        )
        .withColumn("source_file", F.lit(file_path.name))
        .withColumn("source_year", F.lit(file_path.parent.name))
    )

    frames.append(parsed)


if not frames:
    raise RuntimeError("No se encontraron CSV en data/raw/aqua/2022")


raw = frames[0]

for frame in frames[1:]:
    raw = raw.unionByName(frame)


# ------------------------------------------------------------------
# 2. Convertir strings vacíos en NULL
# ------------------------------------------------------------------

for column in columns:
    raw = raw.withColumn(
        column,
        F.when(
            F.trim(F.col(column)) == "",
            F.lit(None)
        ).otherwise(F.trim(F.col(column)))
    )


# ------------------------------------------------------------------
# 3. Eliminar filas de subtotales de Aqua
# ------------------------------------------------------------------

is_total = F.lit(False)

for column in ["Fecha", "Operacion", "Documento", "Cliente", "Producto"]:
    is_total = is_total | (
        F.upper(F.coalesce(F.col(column), F.lit(""))) == "TOTALES"
    )

transactions = raw.filter(
    (~is_total) &
    F.col("Producto").isNotNull()
)


# ------------------------------------------------------------------
# 4. Reconstruir la estructura jerárquica de Aqua
# ------------------------------------------------------------------

window = (
    Window
    .partitionBy("source_file")
    .orderBy("source_row_number")
    .rowsBetween(Window.unboundedPreceding, 0)
)

for column in ["Fecha", "Operacion", "Documento", "Cliente"]:
    transactions = transactions.withColumn(
        column,
        F.last(
            F.col(column),
            ignorenulls=True
        ).over(window)
    )


# ------------------------------------------------------------------
# 5. Crear esquema Silver estandarizado
# ------------------------------------------------------------------

silver = transactions.select(

    F.to_date(
        F.col("Fecha"),
        "dd/MM/yyyy"
    ).alias("transaction_date"),

    F.upper(
        F.trim(F.col("Operacion"))
    ).alias("operation_type"),

    F.trim(
        F.col("Documento")
    ).alias("document_id"),

    F.trim(
        F.col("Cliente")
    ).alias("customer_id"),

    F.trim(
        F.col("Producto")
    ).alias("product_id"),

    F.regexp_replace(
        F.regexp_replace(F.col("Unidades"), r"\.", ""),
        ",",
        "."
    ).cast("decimal(18,2)").alias("units"),

    F.regexp_replace(
        F.regexp_replace(F.col("Venta"), r"\.", ""),
        ",",
        "."
    ).cast("decimal(18,2)").alias("sales_amount"),

    F.col("source_year"),

    F.col("source_file"),

    F.col("source_row_number"),
)


# ------------------------------------------------------------------
# 6. Guardar Silver
# ------------------------------------------------------------------

silver.write \
    .mode("overwrite") \
    .parquet(OUTPUT_PATH)


print("\n" + "=" * 70)
print("SILVER CREADA")
print("=" * 70)

print(f"\nFilas Silver: {silver.count():,}")

print("\nOperaciones:")
silver.groupBy("operation_type") \
    .count() \
    .orderBy("operation_type") \
    .show()

print("\nPrimeras 20 transacciones:")
silver.orderBy("source_file", "source_row_number") \
    .show(20, truncate=False)

spark.stop()
