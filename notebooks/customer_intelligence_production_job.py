# Databricks notebook source
# ============================================================
# retailco CUSTOMER INTELLIGENCE
# PRODUCTION PIPELINE
#
# CELL 1 - CONFIGURATION
#
# Aqua -> S3 -> Bronze -> Silver -> Gold
#      -> Current Features
#      -> Registered RF Model
#      -> Current Scoring
#      -> Top25 / Top30
# ============================================================

import json
import os
import uuid
from datetime import date

import mlflow
import mlflow.spark

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.functions import vector_to_array


# ============================================================
# 1. PROJECT
# ============================================================

PROJECT_NAME = "retailco_customer_intelligence"

CATALOG = "workspace"
SCHEMA = "default"


# ============================================================
# 2. S3 ROOT
# ============================================================

S3_BUCKET = (
    "YOUR_S3_BUCKET_NAME"
)

S3_ROOT = f"s3://{S3_BUCKET}"


# ============================================================
# 3. RAW / BRONZE / SILVER
# ============================================================

S3_RAW_AQUA = (
    f"{S3_ROOT}/raw/aqua/"
)

S3_BRONZE_AQUA = (
    f"{S3_ROOT}/bronze/aqua_files/"
)

S3_SILVER_TRANSACTIONS = (
    f"{S3_ROOT}/silver/transactions/"
)


# ============================================================
# 4. GOLD
# ============================================================

S3_GOLD_FACT_SALES = (
    f"{S3_ROOT}/gold/fact_sales/"
)

S3_GOLD_CUSTOMER_MONTHLY = (
    f"{S3_ROOT}/gold/customer_monthly/"
)

S3_CURRENT_SCORING_FEATURES = (
    f"{S3_ROOT}/gold/current_scoring_features/"
)

# ============================================================
# 5. UNITY CATALOG TABLES
# ============================================================

TABLE_CURRENT_CUSTOMER_SCORES = (
    f"{CATALOG}.{SCHEMA}.current_customer_scores"
)

TABLE_TOP25_CHURN_RISK = (
    f"{CATALOG}.{SCHEMA}.top_25_churn_risk"
)

TABLE_TOP30_CHURN_RISK = (
    f"{CATALOG}.{SCHEMA}.top_30_churn_risk"
)


# ============================================================
# 6. REGISTERED PRODUCTION MODEL
# ============================================================

REGISTERED_MODEL_NAME = (
    f"{CATALOG}.{SCHEMA}."
    "retailco_churn_random_forest"
)

MLFLOW_DFS_TMP = (
    "/Volumes/workspace/default/mlflow_tmp/sparkml"
)

mlflow.set_registry_uri(
    "databricks-uc"
)


# ============================================================
# 6.1 JOB PARAMETERS
#
# Databricks Jobs can override these values with base parameters.
# Empty snapshot_override means: score the previous complete month.
# ============================================================

def get_or_create_widget(name, default_value, label):
    try:
        return dbutils.widgets.get(name)
    except Exception:
        dbutils.widgets.text(name, default_value, label)
        return dbutils.widgets.get(name)


SNAPSHOT_OVERRIDE = get_or_create_widget(
    "snapshot_override",
    "",
    "Snapshot override (YYYY-MM-01)",
).strip()

PUBLISH_OUTPUTS_RAW = get_or_create_widget(
    "publish_outputs",
    "true",
    "Publish production outputs",
).strip().lower()

REGISTERED_MODEL_ALIAS = get_or_create_widget(
    "model_alias",
    "champion",
    "Unity Catalog model alias",
).strip()

PIPELINE_RUN_ID = (
    get_or_create_widget(
        "pipeline_run_id",
        "",
        "External orchestration run id",
    ).strip()
    or uuid.uuid4().hex
)

if PUBLISH_OUTPUTS_RAW not in {"true", "false"}:
    raise ValueError(
        "publish_outputs must be either 'true' or 'false'."
    )

if not REGISTERED_MODEL_ALIAS:
    raise ValueError("model_alias cannot be empty.")

PUBLISH_OUTPUTS = PUBLISH_OUTPUTS_RAW == "true"

REGISTERED_MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}"
    f"@{REGISTERED_MODEL_ALIAS}"
)


# ============================================================
# 7. BUSINESS POLICY
# ============================================================

GENERIC_CUSTOMER_ID = "000000"

CHURN_DROP_THRESHOLD = 0.30

TOP25_SIZE = 25
TOP30_SIZE = 30


# ============================================================
# 8. MODEL FEATURES
#
# IMPORTANT:
# Order must remain identical to model training.
# ============================================================

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


# ============================================================
# 9. CONFIG VALIDATION
# ============================================================

if len(FEATURE_COLUMNS) != 43:
    raise ValueError(
        f"Expected 43 model features, "
        f"found {len(FEATURE_COLUMNS)}."
    )


if len(FEATURE_COLUMNS) != len(set(FEATURE_COLUMNS)):
    raise ValueError(
        "Duplicate feature names found."
    )


# ============================================================
# 10. STARTUP SUMMARY
# ============================================================

print("=" * 90)
print("retailco CUSTOMER INTELLIGENCE - PRODUCTION PIPELINE")
print("=" * 90)

print(f"Project:            {PROJECT_NAME}")
print(f"S3 bucket:          {S3_BUCKET}")
print(f"Model:              {REGISTERED_MODEL_NAME}")
print(f"Model alias:        {REGISTERED_MODEL_ALIAS}")
print(f"Model features:     {len(FEATURE_COLUMNS)}")
print(f"Primary policy:     Top {TOP25_SIZE}")
print(f"Capacity expansion: Top {TOP30_SIZE}")
print(f"Snapshot override:  {SNAPSHOT_OVERRIDE or 'automatic'}")
print(f"Publish outputs:    {PUBLISH_OUTPUTS}")
print(f"Pipeline run id:    {PIPELINE_RUN_ID}")

print()
print("Configuration loaded successfully.")

# COMMAND ----------

# ============================================================
# CELL 2 - RAW / BRONZE PRODUCTION CHECK
#
# PURPOSE:
# - Discover all Aqua CSV files currently available in S3 RAW.
# - Inspect the existing Bronze Delta contract.
# - Compare RAW files against files already ingested.
#
# IMPORTANT:
# This cell does NOT modify any data.
# ============================================================

from pyspark.sql import functions as F


# ============================================================
# 1. RECURSIVE FILE DISCOVERY
# ============================================================

def list_files_recursive(path):
    """
    Recursively list files under an S3 path.
    """

    discovered_files = []

    for item in dbutils.fs.ls(path):

        if item.path.endswith("/"):
            discovered_files.extend(
                list_files_recursive(item.path)
            )
        else:
            discovered_files.append(item)

    return discovered_files


# ============================================================
# 2. DISCOVER RAW AQUA CSV FILES
# ============================================================

raw_files = [
    f
    for f in list_files_recursive(S3_RAW_AQUA)
    if f.path.lower().endswith(".csv")
]


raw_file_rows = [
    (
        f.path,
        f.name,
        int(f.size),
        int(f.modificationTime),
    )
    for f in raw_files
]


raw_files_df = spark.createDataFrame(
    raw_file_rows,
    [
        "raw_path",
        "raw_file",
        "raw_size_bytes",
        "raw_modification_time_ms",
    ]
)


# ============================================================
# 3. RAW VALIDATION
# ============================================================

raw_count = raw_files_df.count()


print("=" * 90)
print("RAW AQUA CHECK")
print("=" * 90)

print(
    f"RAW CSV files found: "
    f"{raw_count:,}"
)


if raw_count == 0:
    raise ValueError(
        "No Aqua CSV files found in RAW."
    )


# ============================================================
# 4. LOAD EXISTING BRONZE DELTA
# ============================================================

bronze_aqua = (
    spark.read
    .format("delta")
    .load(S3_BRONZE_AQUA)
)


bronze_count = bronze_aqua.count()


# ============================================================
# 5. BRONZE CONTRACT
# ============================================================

print()
print("=" * 90)
print("EXISTING BRONZE CHECK")
print("=" * 90)

print(
    f"Bronze rows: "
    f"{bronze_count:,}"
)

print(
    f"Bronze columns: "
    f"{len(bronze_aqua.columns)}"
)

print()
print("Bronze columns:")

for column_name in bronze_aqua.columns:
    print(
        f" - {column_name}"
    )


print()
print("Bronze schema:")

bronze_aqua.printSchema()


# ============================================================
# 6. SAMPLE
# ============================================================

print()
print("=" * 90)
print("BRONZE SAMPLE")
print("=" * 90)

display(
    bronze_aqua.limit(10)
)


# ============================================================
# 7. IDENTIFY FILE TRACKING COLUMN
# ============================================================

candidate_path_columns = [
    "source_path",
    "file_path",
    "path",
]

candidate_file_columns = [
    "source_file",
    "file_name",
    "name",
]


bronze_path_column = next(
    (
        c
        for c in candidate_path_columns
        if c in bronze_aqua.columns
    ),
    None,
)


bronze_file_column = next(
    (
        c
        for c in candidate_file_columns
        if c in bronze_aqua.columns
    ),
    None,
)


print()
print("=" * 90)
print("BRONZE FILE TRACKING")
print("=" * 90)

print(
    f"Path column detected: "
    f"{bronze_path_column}"
)

print(
    f"File column detected: "
    f"{bronze_file_column}"
)


# ============================================================
# 8. RAW VS BRONZE COMPARISON
# ============================================================

if bronze_path_column is not None:

    already_ingested = (
        bronze_aqua
        .select(
            F.col(
                bronze_path_column
            ).alias("raw_path")
        )
        .distinct()
    )

    new_raw_files = (
        raw_files_df
        .join(
            already_ingested,
            on="raw_path",
            how="left_anti",
        )
    )


elif bronze_file_column is not None:

    already_ingested = (
        bronze_aqua
        .select(
            F.col(
                bronze_file_column
            ).alias("raw_file")
        )
        .distinct()
    )

    new_raw_files = (
        raw_files_df
        .join(
            already_ingested,
            on="raw_file",
            how="left_anti",
        )
    )


else:

    new_raw_files = None


# ============================================================
# 9. RESULT
# ============================================================

print()
print("=" * 90)
print("INCREMENTAL BRONZE STATUS")
print("=" * 90)


if new_raw_files is None:

    print(
        "Could not automatically identify "
        "the Bronze file-tracking column."
    )

    print(
        "No data has been modified."
    )

else:

    new_raw_count = (
        new_raw_files.count()
    )

    print(
        f"RAW files:               "
        f"{raw_count:,}"
    )

    print(
        f"New RAW files to ingest: "
        f"{new_raw_count:,}"
    )

    if new_raw_count > 0:

        print()
        print("New files detected:")

        display(
            new_raw_files
            .orderBy(
                "raw_path"
            )
        )

    else:

        print()
        print(
            "Bronze is already up to date."
        )


print()
print(
    "READ-ONLY CHECK COMPLETED."
)

# COMMAND ----------

# ============================================================
# FIND ACTUAL BRONZE PATH
# READ ONLY - DOES NOT MODIFY DATA
# ============================================================

print("S3 ROOT:")
display(
    dbutils.fs.ls(S3_ROOT)
)

print("BRONZE ROOT:")
display(
    dbutils.fs.ls(f"{S3_ROOT}/bronze/")
)

# COMMAND ----------

# ============================================================
# CHECK ACTUAL BRONZE STORAGE
# READ ONLY
# ============================================================

print("BRONZE PATH:")
print(S3_BRONZE_AQUA)

print()
print("=" * 90)
print("BRONZE CONTENTS")
print("=" * 90)

display(
    dbutils.fs.ls(S3_BRONZE_AQUA)
)

print()
print("=" * 90)
print("DELTA LOG CHECK")
print("=" * 90)

try:
    delta_log_files = dbutils.fs.ls(
        f"{S3_BRONZE_AQUA}_delta_log/"
    )

    print("Delta log found: YES")

    display(
        spark.createDataFrame(
            [
                (
                    f.path,
                    f.name,
                    f.size
                )
                for f in delta_log_files
            ],
            [
                "path",
                "name",
                "size"
            ]
        )
    )

except Exception as exc:

    print("Delta log found: NO")
    print()
    print("Actual error:")
    print(str(exc))

# COMMAND ----------

# ============================================================
# CELL 3 - INCREMENTAL RAW -> BRONZE INGESTION
#
# PURPOSE:
# - Append ONLY previously unseen Aqua CSV files.
# - Preserve one Bronze row per source file.
# - Never overwrite existing Bronze data.
#
# INPUT:
#   new_raw_files
#   bronze_aqua
#
# OUTPUT:
#   Delta Bronze at S3_BRONZE_AQUA
# ============================================================

from functools import reduce

from pyspark.sql import functions as F


# ============================================================
# 1. CURRENT STATE
# ============================================================

bronze_before_count = (
    bronze_aqua.count()
)

new_raw_count = (
    new_raw_files.count()
)


print("=" * 90)
print("INCREMENTAL RAW -> BRONZE")
print("=" * 90)

print(
    f"Bronze rows before:       "
    f"{bronze_before_count:,}"
)

print(
    f"New RAW files detected:   "
    f"{new_raw_count:,}"
)


# ============================================================
# 2. EXISTING BRONZE TYPES
#
# We reuse the existing Delta schema instead of guessing types.
# ============================================================

bronze_types = {
    field.name: field.dataType.simpleString()
    for field in bronze_aqua.schema.fields
}


required_bronze_columns = [
    "source_path",
    "source_year",
    "source_file",
    "source_modification_time",
    "source_size_bytes",
    "raw_content",
    "ingested_at",
]


missing_bronze_columns = [
    column_name
    for column_name in required_bronze_columns
    if column_name not in bronze_types
]


if missing_bronze_columns:
    raise ValueError(
        "Bronze contract is missing columns: "
        f"{missing_bronze_columns}"
    )


# ============================================================
# 3. INGEST ONLY NEW FILES
# ============================================================

if new_raw_count == 0:

    print()
    print(
        "No new RAW files found."
    )

    print(
        "Bronze will not be modified."
    )


else:

    # --------------------------------------------------------
    # File metadata is tiny.
    # Collect ONLY metadata rows to the driver.
    # File contents are still read by Spark.
    # --------------------------------------------------------

    new_file_metadata = (
        new_raw_files
        .orderBy("raw_path")
        .collect()
    )


    bronze_file_dfs = []


    for file_metadata in new_file_metadata:

        source_path = (
            file_metadata["raw_path"]
        )

        source_file = (
            file_metadata["raw_file"]
        )

        source_size_bytes = int(
            file_metadata["raw_size_bytes"]
        )

        source_modification_time_ms = int(
            file_metadata[
                "raw_modification_time_ms"
            ]
        )


        # ----------------------------------------------------
        # Extract year from:
        # .../raw/aqua/2026/file.csv
        # ----------------------------------------------------

        source_year = int(
            source_path
            .split("/raw/aqua/")[1]
            .split("/")[0]
        )


        # ----------------------------------------------------
        # Read the ENTIRE CSV as one string / one Spark row.
        # ----------------------------------------------------

        file_content_df = (
            spark.read
            .text(
                source_path,
                wholetext=True,
            )
        )


        # Safety: exactly one row per input file.
        file_content_count = (
            file_content_df.count()
        )


        if file_content_count != 1:
            raise ValueError(
                f"Expected exactly 1 whole-text row for "
                f"{source_path}, found "
                f"{file_content_count}."
            )


        bronze_file_df = (
            file_content_df

            .select(
                F.lit(
                    source_path
                )
                .cast(
                    bronze_types[
                        "source_path"
                    ]
                )
                .alias(
                    "source_path"
                ),

                F.lit(
                    source_year
                )
                .cast(
                    bronze_types[
                        "source_year"
                    ]
                )
                .alias(
                    "source_year"
                ),

                F.lit(
                    source_file
                )
                .cast(
                    bronze_types[
                        "source_file"
                    ]
                )
                .alias(
                    "source_file"
                ),

                F.timestamp_millis(
                    F.lit(
                        source_modification_time_ms
                    )
                )
                .cast(
                    bronze_types[
                        "source_modification_time"
                    ]
                )
                .alias(
                    "source_modification_time"
                ),

                F.lit(
                    source_size_bytes
                )
                .cast(
                    bronze_types[
                        "source_size_bytes"
                    ]
                )
                .alias(
                    "source_size_bytes"
                ),

                F.col(
                    "value"
                )
                .cast(
                    bronze_types[
                        "raw_content"
                    ]
                )
                .alias(
                    "raw_content"
                ),

                F.current_timestamp()
                .cast(
                    bronze_types[
                        "ingested_at"
                    ]
                )
                .alias(
                    "ingested_at"
                ),
            )
        )


        bronze_file_dfs.append(
            bronze_file_df
        )


    # ========================================================
    # 4. UNION NEW BRONZE ROWS
    # ========================================================

    new_bronze_rows = reduce(
        lambda left, right:
            left.unionByName(right),

        bronze_file_dfs,
    )


    rows_to_append = (
        new_bronze_rows.count()
    )


    if rows_to_append != new_raw_count:
        raise ValueError(
            f"Expected {new_raw_count} Bronze rows "
            f"to append, found {rows_to_append}."
        )


    # ========================================================
    # 5. APPEND - NEVER OVERWRITE
    # ========================================================

    (
        new_bronze_rows.write
        .format("delta")
        .mode("append")
        .save(
            S3_BRONZE_AQUA
        )
    )


    print()
    print(
        f"Bronze rows appended:     "
        f"{rows_to_append:,}"
    )


# ============================================================
# 6. RELOAD BRONZE
# ============================================================

bronze_aqua = (
    spark.read
    .format("delta")
    .load(
        S3_BRONZE_AQUA
    )
)


bronze_after_count = (
    bronze_aqua.count()
)


# ============================================================
# 7. QUALITY CHECKS
# ============================================================

duplicate_source_paths = (
    bronze_aqua

    .groupBy(
        "source_path"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


expected_after_count = (
    bronze_before_count
    +
    new_raw_count
)


print()
print("=" * 90)
print("BRONZE VALIDATION")
print("=" * 90)

print(
    f"Bronze rows after:        "
    f"{bronze_after_count:,}"
)

print(
    f"Expected rows after:      "
    f"{expected_after_count:,}"
)

print(
    f"Duplicate source paths:   "
    f"{duplicate_source_paths:,}"
)


if bronze_after_count != expected_after_count:
    raise ValueError(
        "Unexpected Bronze row count after ingestion."
    )


if duplicate_source_paths != 0:
    raise ValueError(
        "Duplicate source_path values found in Bronze."
    )


print()
print(
    "INCREMENTAL BRONZE INGESTION COMPLETED SUCCESSFULLY."
)

# COMMAND ----------

# ============================================================
# CELL 4 - EXISTING SILVER CONTRACT CHECK
#
# READ ONLY
#
# PURPOSE:
# - Load the existing Silver transactions Delta.
# - Inspect schema and provenance columns.
# - Confirm how many Bronze files are represented in Silver.
#
# This cell DOES NOT modify data.
# ============================================================

from pyspark.sql import functions as F


# ============================================================
# 1. LOAD SILVER
# ============================================================

silver_transactions = (
    spark.read
    .format("delta")
    .load(S3_SILVER_TRANSACTIONS)
)


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

silver_rows = (
    silver_transactions.count()
)

silver_columns = (
    len(silver_transactions.columns)
)


print("=" * 90)
print("EXISTING SILVER CHECK")
print("=" * 90)

print(
    f"Silver rows:     "
    f"{silver_rows:,}"
)

print(
    f"Silver columns:  "
    f"{silver_columns}"
)


# ============================================================
# 3. COLUMN LIST
# ============================================================

print()
print("Silver columns:")

for column_name in silver_transactions.columns:
    print(
        f" - {column_name}"
    )


# ============================================================
# 4. SCHEMA
# ============================================================

print()
print("=" * 90)
print("SILVER SCHEMA")
print("=" * 90)

silver_transactions.printSchema()


# ============================================================
# 5. PROVENANCE / FILE TRACKING
# ============================================================

candidate_source_columns = [
    "source_path",
    "source_file",
    "source_year",
]


print()
print("=" * 90)
print("SILVER PROVENANCE")
print("=" * 90)


for column_name in candidate_source_columns:

    print(
        f"{column_name}: "
        f"{column_name in silver_transactions.columns}"
    )


if "source_path" in silver_transactions.columns:

    silver_source_files = (
        silver_transactions
        .select(
            "source_path"
        )
        .distinct()
        .count()
    )

elif "source_file" in silver_transactions.columns:

    silver_source_files = (
        silver_transactions
        .select(
            "source_file"
        )
        .distinct()
        .count()
    )

else:

    silver_source_files = None


print()

if silver_source_files is not None:

    print(
        f"Distinct Bronze source files represented in Silver: "
        f"{silver_source_files:,}"
    )

else:

    print(
        "No source-file tracking column detected."
    )


# ============================================================
# 6. BUSINESS CHECKS
# ============================================================

print()
print("=" * 90)
print("SILVER BUSINESS CHECK")
print("=" * 90)


if "document_type" in silver_transactions.columns:

    display(
        silver_transactions
        .groupBy(
            "document_type"
        )
        .agg(
            F.count("*").alias(
                "rows"
            )
        )
        .orderBy(
            "document_type"
        )
    )


# ============================================================
# 7. SAMPLE
# ============================================================

print()
print("=" * 90)
print("SILVER SAMPLE")
print("=" * 90)

display(
    silver_transactions.limit(10)
)


print()
print(
    "READ-ONLY SILVER CHECK COMPLETED."
)

# COMMAND ----------

# ============================================================
# CELL 5 - RAW -> SILVER PARSER CONTRACT INSPECTION
#
# READ ONLY
#
# PURPOSE:
# Inspect one real Aqua source file and compare:
#   Bronze raw lines
#       vs
#   Silver parsed rows
#
# This cell DOES NOT modify data.
# ============================================================

from pyspark.sql import functions as F


# ============================================================
# 1. SELECT ONE SOURCE FILE
# ============================================================

inspection_source = (
    bronze_aqua
    .select(
        "source_path",
        "source_file",
        "source_year",
        "raw_content",
    )
    .orderBy(
        "source_year",
        "source_file",
    )
    .first()
)


inspection_path = (
    inspection_source["source_path"]
)

inspection_file = (
    inspection_source["source_file"]
)

inspection_year = (
    inspection_source["source_year"]
)

raw_content = (
    inspection_source["raw_content"]
)


print("=" * 90)
print("RAW -> SILVER PARSER INSPECTION")
print("=" * 90)

print(
    f"Source year: {inspection_year}"
)

print(
    f"Source file: {inspection_file}"
)

print(
    f"Source path: {inspection_path}"
)


# ============================================================
# 2. DECODE BRONZE RAW CONTENT
#
# Bronze stores the original file content as binary.
# We decode it only for inspection.
# ============================================================

if isinstance(
    raw_content,
    (bytes, bytearray),
):

    try:

        raw_text = (
            raw_content
            .decode("utf-8-sig")
        )

        detected_encoding = (
            "utf-8-sig"
        )

    except UnicodeDecodeError:

        raw_text = (
            raw_content
            .decode("cp1252")
        )

        detected_encoding = (
            "cp1252"
        )

else:

    raw_text = str(
        raw_content
    )

    detected_encoding = (
        "already-text"
    )


print()
print(
    f"Inspection encoding: "
    f"{detected_encoding}"
)


# ============================================================
# 3. NORMALIZE LINE ENDINGS
# ============================================================

raw_lines = (
    raw_text
    .replace("\r\n", "\n")
    .replace("\r", "\n")
    .split("\n")
)


print()
print(
    f"RAW lines detected: "
    f"{len(raw_lines):,}"
)


# ============================================================
# 4. PRINT FIRST RAW LINES WITH LINE NUMBERS
# ============================================================

print()
print("=" * 90)
print("FIRST 60 RAW LINES")
print("=" * 90)


for line_number, line in enumerate(
    raw_lines[:60],
    start=1,
):

    print(
        f"{line_number:04d} | {line}"
    )


# ============================================================
# 5. SILVER ROWS GENERATED FROM SAME FILE
# ============================================================

silver_inspection = (
    silver_transactions

    .filter(
        F.col("source_path")
        ==
        inspection_path
    )

    .select(
        "source_row_number",
        "transaction_date",
        "operation_type",
        "document_id",
        "customer_id",
        "product_id",
        "units",
        "amount",
    )

    .orderBy(
        "source_row_number"
    )
)


silver_inspection_count = (
    silver_inspection.count()
)


print()
print("=" * 90)
print("SILVER OUTPUT FROM SAME FILE")
print("=" * 90)

print(
    f"Parsed Silver rows: "
    f"{silver_inspection_count:,}"
)


display(
    silver_inspection.limit(40)
)


# ============================================================
# 6. OPERATION TYPE COUNTS
# ============================================================

print()
print("=" * 90)
print("OPERATION TYPES IN SOURCE FILE")
print("=" * 90)


display(
    silver_transactions

    .filter(
        F.col("source_path")
        ==
        inspection_path
    )

    .groupBy(
        "operation_type"
    )

    .agg(
        F.count("*").alias(
            "rows"
        ),

        F.countDistinct(
            "document_id"
        ).alias(
            "documents"
        )
    )

    .orderBy(
        "operation_type"
    )
)


print()
print(
    "READ-ONLY PARSER INSPECTION COMPLETED."
)

# COMMAND ----------

# ============================================================
# CELL 6 - VALIDATE PRODUCTION AQUA PARSER
#
# READ ONLY
#
# PURPOSE:
# - Reconstruct the hierarchical Aqua parser.
# - Parse one historical Bronze file.
# - Compare the result row-by-row against existing Silver.
#
# NOTHING IS WRITTEN.
# ============================================================

import csv
import io

from datetime import datetime

from pyspark.sql import functions as F


# ============================================================
# 1. HELPERS
# ============================================================

def clean_text(value):
    """
    Normalize Aqua text fields.
    """

    if value is None:
        return ""

    return (
        str(value)
        .replace("\u00a0", " ")
        .strip()
    )


def parse_spanish_number(value):
    """
    Convert Spanish-formatted numbers:

        1,00       -> 1.0
        174,90     -> 174.9
        1.234,56   -> 1234.56
        -25,50     -> -25.5

    Blank values become 0.0.
    """

    value = clean_text(value)

    if value == "":
        return 0.0

    value = (
        value
        .replace("€", "")
        .replace(" ", "")
        .replace(".", "")
        .replace(",", ".")
    )

    return float(value)


def decode_aqua_content(raw_content):
    """
    Decode the binary content stored in Bronze.
    """

    if isinstance(
        raw_content,
        (bytes, bytearray),
    ):

        try:
            return raw_content.decode(
                "utf-8-sig"
            )

        except UnicodeDecodeError:
            return raw_content.decode(
                "cp1252"
            )

    return str(raw_content)


# ============================================================
# 2. AQUA HIERARCHICAL PARSER
# ============================================================

def parse_aqua_file(raw_content):
    """
    Parse one Aqua historical-sales export.

    Aqua structure:
      Fecha;Operacion;Documento;Cliente;Producto;Unidades;Venta

    Some product rows omit:
      Fecha
      Operacion
      Documento
      Cliente

    These fields belong to the preceding document and must
    therefore be forward-filled.

    Totales rows are ignored.
    """

    raw_text = decode_aqua_content(
        raw_content
    )

    raw_text = (
        raw_text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    reader = csv.reader(
        io.StringIO(raw_text),
        delimiter=";",
        quotechar='"',
    )


    parsed_rows = []


    current_date = None
    current_operation = None
    current_document = None
    current_customer = None


    for source_row_number, row in enumerate(
        reader,
        start=1,
    ):

        # ----------------------------------------------------
        # Ensure at least seven Aqua fields
        # ----------------------------------------------------

        if len(row) < 7:

            row = (
                row
                +
                [""] * (7 - len(row))
            )


        fecha = clean_text(
            row[0]
        )

        operacion = clean_text(
            row[1]
        ).upper()

        documento = clean_text(
            row[2]
        )

        cliente = clean_text(
            row[3]
        )

        producto = clean_text(
            row[4]
        )

        unidades = clean_text(
            row[5]
        )

        venta = clean_text(
            row[6]
        )


        # ----------------------------------------------------
        # Skip header
        # ----------------------------------------------------

        if (
            source_row_number == 1
            and
            fecha.lower() == "fecha"
        ):
            continue


        # ----------------------------------------------------
        # Update hierarchical document context
        # ----------------------------------------------------

        if fecha:

            try:

                current_date = (
                    datetime.strptime(
                        fecha,
                        "%d/%m/%Y",
                    )
                    .date()
                )

            except ValueError:

                # Not a valid transaction date.
                continue


        if operacion:
            current_operation = (
                operacion
            )


        if documento:
            current_document = (
                documento
            )


        if cliente:
            current_customer = (
                cliente
            )


        # ----------------------------------------------------
        # Skip empty lines
        # ----------------------------------------------------

        if not any(
            [
                fecha,
                operacion,
                documento,
                cliente,
                producto,
                unidades,
                venta,
            ]
        ):
            continue


        # ----------------------------------------------------
        # Skip totals / aggregation rows
        # ----------------------------------------------------

        if (
            producto
            .casefold()
            .startswith("totales")
        ):
            continue


        # ----------------------------------------------------
        # Only transactional document types
        # ----------------------------------------------------

        if current_operation not in {
            "ALBARAN",
            "FACTURA",
        }:
            continue


        # ----------------------------------------------------
        # Need an actual product line
        # ----------------------------------------------------

        if producto == "":
            continue


        # ----------------------------------------------------
        # Require complete inherited document context
        # ----------------------------------------------------

        if (
            current_date is None
            or current_document is None
            or current_customer is None
        ):
            continue


        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        try:

            parsed_units = (
                parse_spanish_number(
                    unidades
                )
            )

            parsed_amount = (
                parse_spanish_number(
                    venta
                )
            )

        except ValueError:

            continue


        # ----------------------------------------------------
        # Output Silver business fields
        # ----------------------------------------------------

        parsed_rows.append(
            (
                int(
                    source_row_number
                ),

                current_date,

                current_operation,

                current_document,

                current_customer,

                producto,

                float(
                    parsed_units
                ),

                float(
                    parsed_amount
                ),
            )
        )


    return parsed_rows


# ============================================================
# 3. PARSE HISTORICAL INSPECTION FILE
# ============================================================

parsed_test_rows = (
    parse_aqua_file(
        inspection_source[
            "raw_content"
        ]
    )
)


print("=" * 90)
print("PRODUCTION PARSER TEST")
print("=" * 90)

print(
    f"Source file:              "
    f"{inspection_file}"
)

print(
    f"Parser rows generated:    "
    f"{len(parsed_test_rows):,}"
)

print(
    f"Existing Silver rows:     "
    f"{silver_inspection_count:,}"
)


# ============================================================
# 4. CREATE TEMPORARY PARSED DATAFRAME
#
# Still READ ONLY.
# ============================================================

parsed_test_df = spark.createDataFrame(
    parsed_test_rows,
    [
        "source_row_number",
        "transaction_date",
        "operation_type",
        "document_id",
        "customer_id",
        "product_id",
        "units",
        "amount",
    ],
)


# ============================================================
# 5. COMPARE ROW NUMBERS
# ============================================================

existing_test_df = (
    silver_transactions

    .filter(
        F.col("source_path")
        ==
        inspection_path
    )

    .select(
        "source_row_number",
        "transaction_date",
        "operation_type",
        "document_id",
        "customer_id",
        "product_id",
        "units",
        "amount",
    )
)


missing_in_parser = (
    existing_test_df
    .select(
        "source_row_number"
    )
    .join(
        parsed_test_df.select(
            "source_row_number"
        ),
        on="source_row_number",
        how="left_anti",
    )
)


extra_in_parser = (
    parsed_test_df
    .select(
        "source_row_number"
    )
    .join(
        existing_test_df.select(
            "source_row_number"
        ),
        on="source_row_number",
        how="left_anti",
    )
)


missing_count = (
    missing_in_parser.count()
)

extra_count = (
    extra_in_parser.count()
)


# ============================================================
# 6. FULL BUSINESS-FIELD COMPARISON
# ============================================================

p = parsed_test_df.alias("p")
s = existing_test_df.alias("s")


comparison = (
    p.join(
        s,
        on=(
            F.col("p.source_row_number")
            ==
            F.col("s.source_row_number")
        ),
        how="inner",
    )
)


mismatches = (
    comparison

    .filter(

        (
            F.col("p.transaction_date")
            !=
            F.col("s.transaction_date")
        )

        |

        (
            F.col("p.operation_type")
            !=
            F.col("s.operation_type")
        )

        |

        (
            F.col("p.document_id")
            !=
            F.col("s.document_id")
        )

        |

        (
            F.col("p.customer_id")
            !=
            F.col("s.customer_id")
        )

        |

        (
            F.col("p.product_id")
            !=
            F.col("s.product_id")
        )

        |

        (
            F.abs(
                F.col("p.units")
                -
                F.col("s.units")
            )
            >
            F.lit(0.000001)
        )

        |

        (
            F.abs(
                F.col("p.amount")
                -
                F.col("s.amount")
            )
            >
            F.lit(0.000001)
        )
    )
)


mismatch_count = (
    mismatches.count()
)


# ============================================================
# 7. VALIDATION SUMMARY
# ============================================================

print()
print("=" * 90)
print("PARSER VALIDATION")
print("=" * 90)

print(
    f"Rows expected:            "
    f"{silver_inspection_count:,}"
)

print(
    f"Rows parsed:              "
    f"{len(parsed_test_rows):,}"
)

print(
    f"Missing source rows:      "
    f"{missing_count:,}"
)

print(
    f"Extra source rows:        "
    f"{extra_count:,}"
)

print(
    f"Business-field mismatches:"
    f" {mismatch_count:,}"
)


# ============================================================
# 8. SHOW DIFFERENCES ONLY IF NEEDED
# ============================================================

if missing_count > 0:

    print()
    print("MISSING ROWS:")

    display(
        missing_in_parser
        .orderBy(
            "source_row_number"
        )
        .limit(20)
    )


if extra_count > 0:

    print()
    print("EXTRA ROWS:")

    display(
        extra_in_parser
        .orderBy(
            "source_row_number"
        )
        .limit(20)
    )


if mismatch_count > 0:

    print()
    print("FIELD MISMATCHES:")

    display(
        mismatches.limit(20)
    )


# ============================================================
# 9. FINAL RESULT
# ============================================================

if (
    len(parsed_test_rows)
    ==
    silver_inspection_count
    and
    missing_count == 0
    and
    extra_count == 0
    and
    mismatch_count == 0
):

    print()
    print(
        "AQUA PARSER VALIDATED EXACTLY AGAINST EXISTING SILVER."
    )

else:

    print()
    print(
        "PARSER DOES NOT YET MATCH EXISTING SILVER."
    )

    print(
        "DO NOT USE IT FOR PRODUCTION WRITES YET."
    )


print()
print(
    "Existing Silver columns:"
)

print(
    silver_transactions.columns
)

# COMMAND ----------

# ============================================================
# CELL 7 - INCREMENTAL BRONZE -> SILVER
#
# PURPOSE:
# - Detect Bronze files not yet represented in Silver.
# - Parse ONLY those files with the validated Aqua parser.
# - Append them to Silver.
# - Never overwrite existing Silver history.
#
# DEPENDS ON:
# - bronze_aqua
# - silver_transactions
# - parse_aqua_file()
# - decode_aqua_content()
# ============================================================

from datetime import datetime

from pyspark.sql import functions as F


# ============================================================
# 1. CURRENT SILVER STATE
# ============================================================

silver_before_count = (
    silver_transactions.count()
)


bronze_file_count = (
    bronze_aqua
    .select("source_path")
    .distinct()
    .count()
)


silver_existing_files = (
    silver_transactions
    .select("source_path")
    .distinct()
)


# ============================================================
# 2. FIND BRONZE FILES NOT YET IN SILVER
# ============================================================

new_bronze_files = (
    bronze_aqua

    .join(
        silver_existing_files,
        on="source_path",
        how="left_anti",
    )

    .select(
        "source_path",
        "source_year",
        "source_file",
        "source_modification_time",
        "source_size_bytes",
        "raw_content",
    )

    .orderBy(
        "source_year",
        "source_file",
    )
)


new_bronze_file_count = (
    new_bronze_files.count()
)


print("=" * 90)
print("INCREMENTAL BRONZE -> SILVER")
print("=" * 90)

print(
    f"Bronze files available:      "
    f"{bronze_file_count:,}"
)

print(
    f"Silver rows before:          "
    f"{silver_before_count:,}"
)

print(
    f"New Bronze files to process: "
    f"{new_bronze_file_count:,}"
)


# ============================================================
# 3. SILVER CONTRACT
# ============================================================

SILVER_REQUIRED_COLUMNS = [
    "transaction_date",
    "operation_type",
    "document_id",
    "customer_id",
    "product_id",
    "units",
    "amount",
    "source_year",
    "source_file",
    "source_path",
    "source_row_number",
    "source_size_bytes",
    "source_modification_time",
    "ingested_at",
    "raw_line",
]


missing_silver_columns = [
    c
    for c in SILVER_REQUIRED_COLUMNS
    if c not in silver_transactions.columns
]


if missing_silver_columns:
    raise ValueError(
        "Existing Silver contract is missing columns: "
        f"{missing_silver_columns}"
    )


silver_types = {
    field.name: field.dataType.simpleString()
    for field in silver_transactions.schema.fields
}


# ============================================================
# 4. PROCESS ONLY NEW BRONZE FILES
# ============================================================

rows_to_append = 0


if new_bronze_file_count == 0:

    print()
    print(
        "No new Bronze files found."
    )

    print(
        "Silver will not be modified."
    )


else:

    # --------------------------------------------------------
    # Bronze contains one row per source file.
    # New-file volume is small, so collecting the new files
    # for the hierarchical parser is intentional here.
    # --------------------------------------------------------

    new_files = (
        new_bronze_files.collect()
    )


    silver_records = []


    for source in new_files:

        source_path = (
            source["source_path"]
        )

        source_year = int(
            source["source_year"]
        )

        source_file = (
            source["source_file"]
        )

        source_size_bytes = int(
            source["source_size_bytes"]
        )

        source_modification_time = (
            source[
                "source_modification_time"
            ]
        )

        raw_content = (
            source["raw_content"]
        )


        print()
        print(
            f"Parsing: {source_file}"
        )


        # ----------------------------------------------------
        # Parse using the parser already validated exactly
        # against historical Silver.
        # ----------------------------------------------------

        parsed_rows = (
            parse_aqua_file(
                raw_content
            )
        )


        if len(parsed_rows) == 0:
            raise ValueError(
                f"No transactional rows parsed from "
                f"{source_path}"
            )


        # ----------------------------------------------------
        # Recover original RAW lines for provenance.
        # ----------------------------------------------------

        raw_text = (
            decode_aqua_content(
                raw_content
            )
        )

        raw_lines = (
            raw_text
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .split("\n")
        )


        # ----------------------------------------------------
        # Enrich parsed business rows with Bronze metadata.
        # ----------------------------------------------------

        for parsed_row in parsed_rows:

            (
                source_row_number,
                transaction_date,
                operation_type,
                document_id,
                customer_id,
                product_id,
                units,
                amount,
            ) = parsed_row


            if (
                source_row_number >= 1
                and
                source_row_number <= len(raw_lines)
            ):

                raw_line = (
                    raw_lines[
                        source_row_number - 1
                    ]
                )

            else:

                raw_line = None


            silver_records.append(
                (
                    transaction_date,
                    operation_type,
                    document_id,
                    customer_id,
                    product_id,
                    float(units),
                    float(amount),
                    source_year,
                    source_file,
                    source_path,
                    int(source_row_number),
                    source_size_bytes,
                    source_modification_time,
                    raw_line,
                )
            )


        print(
            f"Parsed transactional rows: "
            f"{len(parsed_rows):,}"
        )


    # ========================================================
    # 5. BUILD NEW SILVER DATAFRAME
    # ========================================================

    new_silver = spark.createDataFrame(
        silver_records,
        [
            "transaction_date",
            "operation_type",
            "document_id",
            "customer_id",
            "product_id",
            "units",
            "amount",
            "source_year",
            "source_file",
            "source_path",
            "source_row_number",
            "source_size_bytes",
            "source_modification_time",
            "raw_line",
        ],
    )


    # --------------------------------------------------------
    # Add Silver processing timestamp.
    # --------------------------------------------------------

    new_silver = (
        new_silver
        .withColumn(
            "ingested_at",
            F.current_timestamp(),
        )
    )


    # ========================================================
    # 6. CAST EXACTLY TO EXISTING SILVER CONTRACT
    # ========================================================

    new_silver = (
        new_silver
        .select(
            *[
                F.col(column_name)
                .cast(
                    silver_types[
                        column_name
                    ]
                )
                .alias(
                    column_name
                )
                for column_name
                in silver_transactions.columns
            ]
        )
    )


    rows_to_append = (
        new_silver.count()
    )


    if rows_to_append == 0:
        raise ValueError(
            "New Bronze files were detected but "
            "zero Silver rows were produced."
        )


    # ========================================================
    # 7. PRE-WRITE QUALITY CHECKS
    # ========================================================

    duplicate_new_rows = (
        new_silver

        .groupBy(
            "source_path",
            "source_row_number",
        )

        .count()

        .filter(
            F.col("count") > 1
        )

        .count()
    )


    critical_null_condition = (
        F.col("transaction_date").isNull()
        |
        F.col("operation_type").isNull()
        |
        F.col("document_id").isNull()
        |
        F.col("customer_id").isNull()
        |
        F.col("product_id").isNull()
        |
        F.col("units").isNull()
        |
        F.col("amount").isNull()
    )


    critical_nulls_new = (
        new_silver
        .filter(
            critical_null_condition
        )
        .count()
    )


    print()
    print("=" * 90)
    print("PRE-WRITE SILVER VALIDATION")
    print("=" * 90)

    print(
        f"Rows to append:           "
        f"{rows_to_append:,}"
    )

    print(
        f"Duplicate provenance:     "
        f"{duplicate_new_rows:,}"
    )

    print(
        f"Critical null rows:       "
        f"{critical_nulls_new:,}"
    )


    if duplicate_new_rows != 0:
        raise ValueError(
            "Duplicate source_path/source_row_number "
            "found in new Silver rows."
        )


    if critical_nulls_new != 0:
        raise ValueError(
            "Critical nulls found in new Silver rows."
        )


    # ========================================================
    # 8. APPEND TO DELTA SILVER
    #
    # NEVER OVERWRITE HISTORY.
    # ========================================================

    (
        new_silver.write
        .format("delta")
        .mode("append")
        .save(
            S3_SILVER_TRANSACTIONS
        )
    )


    print()
    print(
        f"Silver rows appended:     "
        f"{rows_to_append:,}"
    )


# ============================================================
# 9. RELOAD SILVER
# ============================================================

silver_transactions = (
    spark.read
    .format("delta")
    .load(
        S3_SILVER_TRANSACTIONS
    )
)


silver_after_count = (
    silver_transactions.count()
)


expected_silver_after = (
    silver_before_count
    +
    rows_to_append
)


# ============================================================
# 10. GLOBAL PROVENANCE VALIDATION
# ============================================================

duplicate_provenance = (
    silver_transactions

    .groupBy(
        "source_path",
        "source_row_number",
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


silver_file_count = (
    silver_transactions
    .select(
        "source_path"
    )
    .distinct()
    .count()
)


# ============================================================
# 11. FINAL VALIDATION
# ============================================================

print()
print("=" * 90)
print("SILVER VALIDATION")
print("=" * 90)

print(
    f"Silver rows after:        "
    f"{silver_after_count:,}"
)

print(
    f"Expected rows after:      "
    f"{expected_silver_after:,}"
)

print(
    f"Bronze source files:      "
    f"{bronze_file_count:,}"
)

print(
    f"Silver source files:      "
    f"{silver_file_count:,}"
)

print(
    f"Duplicate provenance:     "
    f"{duplicate_provenance:,}"
)


if (
    silver_after_count
    !=
    expected_silver_after
):
    raise ValueError(
        "Unexpected Silver row count."
    )


if duplicate_provenance != 0:
    raise ValueError(
        "Duplicate Silver provenance detected."
    )


if (
    silver_file_count
    !=
    bronze_file_count
):
    raise ValueError(
        "Not all Bronze source files are represented "
        "in Silver."
    )


# ============================================================
# 12. OPERATION SUMMARY
# ============================================================

print()
print("=" * 90)
print("SILVER OPERATION SUMMARY")
print("=" * 90)


display(
    silver_transactions

    .groupBy(
        "operation_type"
    )

    .agg(
        F.count("*").alias(
            "rows"
        ),

        F.countDistinct(
            "document_id"
        ).alias(
            "documents"
        )
    )

    .orderBy(
        "operation_type"
    )
)


print()
print(
    "INCREMENTAL BRONZE -> SILVER COMPLETED SUCCESSFULLY."
)

# COMMAND ----------

# ============================================================
# CELL 8 - EXISTING GOLD FACT_SALES CHECK
#
# READ ONLY
#
# PURPOSE:
# - Load the validated Gold invoice-level sales table.
# - Inspect its exact schema.
# - Validate its business totals.
#
# This cell DOES NOT modify data.
# ============================================================

from pyspark.sql import functions as F


# ============================================================
# 1. LOAD EXISTING GOLD FACT_SALES
# ============================================================

fact_sales = (
    spark.read
    .format("delta")
    .load(S3_GOLD_FACT_SALES)
)


# ============================================================
# 2. BASIC COUNTS
# ============================================================

fact_sales_rows = (
    fact_sales.count()
)

fact_sales_customers = (
    fact_sales
    .select("customer_id")
    .distinct()
    .count()
)


print("=" * 90)
print("EXISTING GOLD FACT_SALES CHECK")
print("=" * 90)

print(
    f"Invoices:          "
    f"{fact_sales_rows:,}"
)

print(
    f"Customers:         "
    f"{fact_sales_customers:,}"
)

print(
    f"Columns:           "
    f"{len(fact_sales.columns)}"
)


# ============================================================
# 3. COLUMN CONTRACT
# ============================================================

print()
print("=" * 90)
print("FACT_SALES COLUMNS")
print("=" * 90)

for column_name in fact_sales.columns:
    print(
        f" - {column_name}"
    )


print()
print("=" * 90)
print("FACT_SALES SCHEMA")
print("=" * 90)

fact_sales.printSchema()


# ============================================================
# 4. REQUIRED GOLD CONTRACT CHECK
# ============================================================

required_fact_sales_columns = [
    "invoice_id",
    "invoice_date",
    "customer_id",
    "net_revenue",
]


missing_fact_sales_columns = [
    column_name
    for column_name in required_fact_sales_columns
    if column_name not in fact_sales.columns
]


if missing_fact_sales_columns:

    raise ValueError(
        "fact_sales is missing required columns: "
        f"{missing_fact_sales_columns}"
    )


# ============================================================
# 5. BUSINESS TOTALS
# ============================================================

summary = (
    fact_sales
    .agg(
        F.count("*").alias(
            "invoices"
        ),

        F.countDistinct(
            "customer_id"
        ).alias(
            "customers"
        ),

        F.round(
            F.sum("net_revenue"),
            2,
        ).alias(
            "net_revenue"
        ),

        F.min(
            "invoice_date"
        ).alias(
            "min_invoice_date"
        ),

        F.max(
            "invoice_date"
        ).alias(
            "max_invoice_date"
        ),
    )
)


print()
print("=" * 90)
print("FACT_SALES BUSINESS SUMMARY")
print("=" * 90)

display(
    summary
)


# ============================================================
# 6. INVOICE TYPE SUMMARY
# ============================================================

if "invoice_type" in fact_sales.columns:

    print()
    print("=" * 90)
    print("INVOICE TYPE SUMMARY")
    print("=" * 90)

    display(
        fact_sales
        .groupBy(
            "invoice_type"
        )
        .agg(
            F.count("*").alias(
                "invoices"
            ),

            F.round(
                F.sum("net_revenue"),
                2,
            ).alias(
                "net_revenue"
            ),
        )
        .orderBy(
            "invoice_type"
        )
    )


# ============================================================
# 7. GENERIC CUSTOMER 000000 AUDIT
#
# IMPORTANT:
# Here we work from SILVER, where the identifier
# is still called document_id.
# ============================================================

generic_customer_audit = (
    silver_transactions

    .filter(
        (F.col("operation_type") == "FACTURA")
        &
        (F.col("customer_id") == GENERIC_CUSTOMER_ID)
    )

    .groupBy(
        "document_id"
    )

    .agg(
        F.sum(
            "amount"
        ).alias(
            "invoice_revenue"
        )
    )

    .agg(
        F.count("*").alias(
            "generic_invoices"
        ),

        F.round(
            F.sum(
                "invoice_revenue"
            ),
            2,
        ).alias(
            "generic_net_revenue"
        )
    )
)


print()
print("=" * 90)
print("GENERIC CUSTOMER 000000 AUDIT")
print("=" * 90)

display(
    generic_customer_audit
)


# ============================================================
# 8. FACT_SALES SAMPLE
#
# IMPORTANT:
# Gold uses invoice_id, not document_id.
# ============================================================

print()
print("=" * 90)
print("FACT_SALES SAMPLE")
print("=" * 90)

display(
    fact_sales
    .orderBy(
        "invoice_date",
        "invoice_id",
    )
    .limit(10)
)


print()
print(
    "READ-ONLY GOLD FACT_SALES CHECK COMPLETED."
)

# COMMAND ----------

# ============================================================
# CELL 9 - EXACT FACT_SALES CONTRACT
# READ ONLY
# ============================================================

print("FACT_SALES COLUMNS:")
print(fact_sales.columns)

print()
print("FACT_SALES SCHEMA:")
fact_sales.printSchema()

# COMMAND ----------

# ============================================================
# CELL 10 - VALIDATE SILVER -> GOLD FACT_SALES REBUILD
#
# READ ONLY
#
# PURPOSE:
# - Rebuild fact_sales entirely from validated Silver.
# - Compare it against the existing Gold fact_sales.
# - Verify invoice-level business logic exactly.
#
# NOTHING IS WRITTEN.
# ============================================================

from pyspark.sql import functions as F


# ============================================================
# 1. START FROM COMMERCIAL INVOICES ONLY
#
# BUSINESS RULES:
# - Revenue comes ONLY from FACTURA.
# - ALBARAN is never added to revenue.
# - Customer 000000 is generic cash/particulars and excluded.
# - Negative invoice totals are valid credit notes.
# ============================================================

factura_lines = (
    silver_transactions

    .filter(
        (F.col("operation_type") == "FACTURA")
        &
        (F.col("customer_id") != GENERIC_CUSTOMER_ID)
    )
)


print("=" * 90)
print("SILVER -> GOLD FACT_SALES REBUILD TEST")
print("=" * 90)

print(
    f"FACTURA Silver lines used: "
    f"{factura_lines.count():,}"
)


# ============================================================
# 2. CHECK INVOICE CONSISTENCY
#
# One invoice should have:
# - one date
# - one customer
# ============================================================

invoice_consistency = (
    factura_lines

    .groupBy(
        "document_id"
    )

    .agg(
        F.countDistinct(
            "transaction_date"
        ).alias(
            "date_count"
        ),

        F.countDistinct(
            "customer_id"
        ).alias(
            "customer_count"
        ),

        F.countDistinct(
            "source_year"
        ).alias(
            "source_year_count"
        ),
    )
)


inconsistent_invoices = (
    invoice_consistency

    .filter(
        (F.col("date_count") != 1)
        |
        (F.col("customer_count") != 1)
        |
        (F.col("source_year_count") != 1)
    )
)


inconsistent_invoice_count = (
    inconsistent_invoices.count()
)


print(
    f"Inconsistent invoices:    "
    f"{inconsistent_invoice_count:,}"
)


if inconsistent_invoice_count != 0:

    print()
    print("INCONSISTENT INVOICES:")

    display(
        inconsistent_invoices.limit(20)
    )

    raise ValueError(
        "Invoice consistency check failed."
    )


# ============================================================
# 3. REBUILD INVOICE-LEVEL GOLD
# ============================================================

rebuilt_fact_sales = (
    factura_lines

    .groupBy(
        "document_id",
        "transaction_date",
        "customer_id",
    )

    .agg(
        F.sum(
            "amount"
        ).alias(
            "net_revenue"
        ),

        F.sum(
            "units"
        ).alias(
            "total_units"
        ),

        F.count(
            "*"
        ).alias(
            "line_count"
        ),

        F.countDistinct(
            "product_id"
        ).alias(
            "distinct_products"
        ),

        F.first(
            "source_year"
        ).alias(
            "source_year"
        ),

        F.array_sort(
            F.collect_set(
                "source_file"
            )
        ).alias(
            "source_files"
        ),
    )

    .withColumnRenamed(
        "document_id",
        "invoice_id",
    )

    .withColumnRenamed(
        "transaction_date",
        "invoice_date",
    )
)


# ============================================================
# 4. INVOICE TYPE
#
# Positive -> PURCHASE
# Negative -> CREDIT_NOTE
# Zero     -> ZERO
# ============================================================

rebuilt_fact_sales = (
    rebuilt_fact_sales

    .withColumn(
        "invoice_type",

        F.when(
            F.col("net_revenue") > 0,
            F.lit("PURCHASE"),
        )

        .when(
            F.col("net_revenue") < 0,
            F.lit("CREDIT_NOTE"),
        )

        .otherwise(
            F.lit("ZERO")
        )
    )
)


# ============================================================
# 5. DATE FEATURES
# ============================================================

rebuilt_fact_sales = (
    rebuilt_fact_sales

    .withColumn(
        "invoice_year",
        F.year(
            "invoice_date"
        ),
    )

    .withColumn(
        "invoice_month",
        F.month(
            "invoice_date"
        ),
    )

    .withColumn(
        "year_month",
        F.date_format(
            "invoice_date",
            "yyyy-MM",
        ),
    )
)


# ============================================================
# 6. EXACT COLUMN ORDER
#
# gold_created_at is intentionally excluded from the
# comparison because it is a processing timestamp.
# ============================================================

COMPARISON_COLUMNS = [
    "invoice_date",
    "invoice_id",
    "customer_id",
    "net_revenue",
    "invoice_type",
    "total_units",
    "line_count",
    "distinct_products",
    "invoice_year",
    "invoice_month",
    "year_month",
    "source_year",
    "source_files",
]


rebuilt_fact_sales = (
    rebuilt_fact_sales
    .select(
        *COMPARISON_COLUMNS
    )
)


existing_fact_sales_comparison = (
    fact_sales
    .select(
        *COMPARISON_COLUMNS
    )
)


# ============================================================
# 7. BASIC COUNTS
# ============================================================

rebuilt_invoice_count = (
    rebuilt_fact_sales.count()
)

existing_invoice_count = (
    existing_fact_sales_comparison.count()
)


print()
print("=" * 90)
print("REBUILD SUMMARY")
print("=" * 90)

print(
    f"Existing Gold invoices:   "
    f"{existing_invoice_count:,}"
)

print(
    f"Rebuilt Gold invoices:    "
    f"{rebuilt_invoice_count:,}"
)


# ============================================================
# 8. BUSINESS TOTALS
# ============================================================

rebuilt_summary = (
    rebuilt_fact_sales

    .agg(
        F.count("*").alias(
            "invoices"
        ),

        F.countDistinct(
            "customer_id"
        ).alias(
            "customers"
        ),

        F.round(
            F.sum(
                "net_revenue"
            ),
            2,
        ).alias(
            "net_revenue"
        ),

        F.min(
            "invoice_date"
        ).alias(
            "min_invoice_date"
        ),

        F.max(
            "invoice_date"
        ).alias(
            "max_invoice_date"
        ),
    )
)


print()
print("=" * 90)
print("REBUILT BUSINESS SUMMARY")
print("=" * 90)

display(
    rebuilt_summary
)


# ============================================================
# 9. COMPARE INVOICE IDS
# ============================================================

missing_in_rebuild = (
    existing_fact_sales_comparison
    .select(
        "invoice_id"
    )
    .join(
        rebuilt_fact_sales.select(
            "invoice_id"
        ),
        on="invoice_id",
        how="left_anti",
    )
)


extra_in_rebuild = (
    rebuilt_fact_sales
    .select(
        "invoice_id"
    )
    .join(
        existing_fact_sales_comparison.select(
            "invoice_id"
        ),
        on="invoice_id",
        how="left_anti",
    )
)


missing_invoice_count = (
    missing_in_rebuild.count()
)

extra_invoice_count = (
    extra_in_rebuild.count()
)


# ============================================================
# 10. COMPARE BUSINESS FIELDS
# ============================================================

existing = (
    existing_fact_sales_comparison
    .alias("e")
)

rebuilt = (
    rebuilt_fact_sales
    .alias("r")
)


joined_comparison = (
    existing

    .join(
        rebuilt,
        on=(
            F.col("e.invoice_id")
            ==
            F.col("r.invoice_id")
        ),
        how="inner",
    )
)


field_mismatches = (
    joined_comparison

    .filter(

        (
            F.col("e.invoice_date")
            !=
            F.col("r.invoice_date")
        )

        |

        (
            F.col("e.customer_id")
            !=
            F.col("r.customer_id")
        )

        |

        (
            F.abs(
                F.col("e.net_revenue")
                -
                F.col("r.net_revenue")
            )
            >
            F.lit(0.000001)
        )

        |

        (
            F.col("e.invoice_type")
            !=
            F.col("r.invoice_type")
        )

        |

        (
            F.abs(
                F.col("e.total_units")
                -
                F.col("r.total_units")
            )
            >
            F.lit(0.000001)
        )

        |

        (
            F.col("e.line_count")
            !=
            F.col("r.line_count")
        )

        |

        (
            F.col("e.distinct_products")
            !=
            F.col("r.distinct_products")
        )

        |

        (
            F.col("e.invoice_year")
            !=
            F.col("r.invoice_year")
        )

        |

        (
            F.col("e.invoice_month")
            !=
            F.col("r.invoice_month")
        )

        |

        (
            F.col("e.year_month")
            !=
            F.col("r.year_month")
        )

        |

        (
            F.col("e.source_year")
            !=
            F.col("r.source_year")
        )

        |

        (
            F.col("e.source_files")
            !=
            F.col("r.source_files")
        )
    )
)


field_mismatch_count = (
    field_mismatches.count()
)


# ============================================================
# 11. VALIDATION SUMMARY
# ============================================================

print()
print("=" * 90)
print("FACT_SALES REBUILD VALIDATION")
print("=" * 90)

print(
    f"Expected invoices:         "
    f"{existing_invoice_count:,}"
)

print(
    f"Rebuilt invoices:          "
    f"{rebuilt_invoice_count:,}"
)

print(
    f"Missing invoices:          "
    f"{missing_invoice_count:,}"
)

print(
    f"Extra invoices:            "
    f"{extra_invoice_count:,}"
)

print(
    f"Business-field mismatches: "
    f"{field_mismatch_count:,}"
)


# ============================================================
# 12. SHOW DIFFERENCES IF ANY
# ============================================================

if missing_invoice_count > 0:

    print()
    print("MISSING IN REBUILD:")

    display(
        missing_in_rebuild.limit(20)
    )


if extra_invoice_count > 0:

    print()
    print("EXTRA IN REBUILD:")

    display(
        extra_in_rebuild.limit(20)
    )


if field_mismatch_count > 0:

    print()
    print("FIELD MISMATCHES:")

    display(
        field_mismatches.limit(20)
    )


# ============================================================
# 13. FINAL RESULT
# ============================================================

if (
    rebuilt_invoice_count
    ==
    existing_invoice_count
    and
    missing_invoice_count == 0
    and
    extra_invoice_count == 0
    and
    field_mismatch_count == 0
):

    print()
    print(
        "FACT_SALES REBUILD VALIDATED EXACTLY "
        "AGAINST EXISTING GOLD."
    )

else:

    print()
    print(
        "FACT_SALES REBUILD DOES NOT YET MATCH GOLD."
    )

    print(
        "DO NOT WRITE GOLD YET."
    )

# COMMAND ----------

# ============================================================
# CELL 11 - PRODUCTION SILVER -> GOLD FACT_SALES
#
# PURPOSE:
# - Identify invoices affected by newly processed Bronze files.
# - Rebuild those invoices from the COMPLETE Silver history.
# - MERGE them safely into Gold fact_sales.
# - Preserve an idempotent invoice-level Gold table.
#
# CURRENT RUN:
# If there are no new Bronze files, Gold is not modified.
# ============================================================

from delta.tables import DeltaTable
from pyspark.sql import functions as F


# ============================================================
# 1. CURRENT GOLD STATE
# ============================================================

fact_sales_before = (
    spark.read
    .format("delta")
    .load(S3_GOLD_FACT_SALES)
)

gold_rows_before = (
    fact_sales_before.count()
)

gold_revenue_before = (
    fact_sales_before
    .agg(
        F.round(
            F.sum("net_revenue"),
            2,
        ).alias("net_revenue")
    )
    .first()["net_revenue"]
)


print("=" * 90)
print("PRODUCTION SILVER -> GOLD FACT_SALES")
print("=" * 90)

print(
    f"Gold invoices before:       "
    f"{gold_rows_before:,}"
)

print(
    f"Gold net revenue before:    "
    f"{gold_revenue_before:,.2f}"
)


# ============================================================
# 2. FILES PROCESSED IN THIS PIPELINE RUN
#
# new_bronze_files was created in CELL 7 before Silver append.
# It represents exactly the Bronze files that were missing
# from Silver at the start of this run.
# ============================================================

processed_source_paths = (
    new_bronze_files
    .select("source_path")
    .distinct()
)

processed_file_count = (
    processed_source_paths.count()
)


print(
    f"New source files this run:  "
    f"{processed_file_count:,}"
)


# ============================================================
# 3. IDENTIFY AFFECTED INVOICES
#
# We identify invoice IDs appearing in the newly processed
# source files.
#
# Then we rebuild those invoice IDs using ALL Silver lines,
# not only the new file lines.
#
# This makes the MERGE safe even if an invoice appears across
# more than one source file.
# ============================================================

affected_invoice_ids = (
    silver_transactions

    .join(
        processed_source_paths,
        on="source_path",
        how="inner",
    )

    .filter(
        (F.col("operation_type") == "FACTURA")
        &
        (F.col("customer_id") != GENERIC_CUSTOMER_ID)
    )

    .select(
        F.col("document_id").alias("invoice_id")
    )

    .distinct()
)


affected_invoice_count = (
    affected_invoice_ids.count()
)


print(
    f"Affected invoices:          "
    f"{affected_invoice_count:,}"
)


# ============================================================
# 4. NO-OP IF NOTHING CHANGED
# ============================================================

if affected_invoice_count == 0:

    print()
    print(
        "No commercial invoices require Gold updates."
    )

    print(
        "fact_sales will not be modified."
    )


# ============================================================
# 5. REBUILD AFFECTED INVOICES
# ============================================================

else:

    affected_factura_lines = (
        silver_transactions

        .filter(
            (F.col("operation_type") == "FACTURA")
            &
            (F.col("customer_id") != GENERIC_CUSTOMER_ID)
        )

        .join(
            affected_invoice_ids
            .withColumnRenamed(
                "invoice_id",
                "document_id",
            ),
            on="document_id",
            how="inner",
        )
    )


    affected_fact_sales = (
        affected_factura_lines

        .groupBy(
            "document_id",
            "transaction_date",
            "customer_id",
        )

        .agg(
            F.sum(
                "amount"
            ).alias(
                "net_revenue"
            ),

            F.sum(
                "units"
            ).alias(
                "total_units"
            ),

            F.count(
                "*"
            ).alias(
                "line_count"
            ),

            F.countDistinct(
                "product_id"
            ).alias(
                "distinct_products"
            ),

            F.first(
                "source_year"
            ).alias(
                "source_year"
            ),

            F.array_sort(
                F.collect_set(
                    "source_file"
                )
            ).alias(
                "source_files"
            ),
        )

        .withColumnRenamed(
            "document_id",
            "invoice_id",
        )

        .withColumnRenamed(
            "transaction_date",
            "invoice_date",
        )
    )


    # ========================================================
    # 6. INVOICE TYPE
    # ========================================================

    affected_fact_sales = (
        affected_fact_sales

        .withColumn(
            "invoice_type",

            F.when(
                F.col("net_revenue") > 0,
                F.lit("PURCHASE"),
            )

            .when(
                F.col("net_revenue") < 0,
                F.lit("CREDIT_NOTE"),
            )

            .otherwise(
                F.lit("ZERO")
            )
        )
    )


    # ========================================================
    # 7. DATE FEATURES + PROCESSING TIMESTAMP
    # ========================================================

    affected_fact_sales = (
        affected_fact_sales

        .withColumn(
            "invoice_year",
            F.year(
                "invoice_date"
            ),
        )

        .withColumn(
            "invoice_month",
            F.month(
                "invoice_date"
            ),
        )

        .withColumn(
            "year_month",
            F.date_format(
                "invoice_date",
                "yyyy-MM",
            ),
        )

        .withColumn(
            "gold_created_at",
            F.current_timestamp(),
        )
    )


    # ========================================================
    # 8. EXACT EXISTING GOLD CONTRACT
    # ========================================================

    affected_fact_sales = (
        affected_fact_sales

        .select(
            *fact_sales_before.columns
        )
    )


    affected_rows = (
        affected_fact_sales.count()
    )


    duplicate_invoice_ids = (
        affected_fact_sales

        .groupBy(
            "invoice_id"
        )

        .count()

        .filter(
            F.col("count") > 1
        )

        .count()
    )


    print()
    print("=" * 90)
    print("PRE-MERGE GOLD VALIDATION")
    print("=" * 90)

    print(
        f"Invoices to merge:          "
        f"{affected_rows:,}"
    )

    print(
        f"Duplicate invoice IDs:      "
        f"{duplicate_invoice_ids:,}"
    )


    if duplicate_invoice_ids != 0:

        raise ValueError(
            "Duplicate invoice_id values detected "
            "before Gold MERGE."
        )


    # ========================================================
    # 9. DELTA MERGE
    #
    # Existing invoice -> UPDATE
    # New invoice      -> INSERT
    # ========================================================

    gold_delta = (
        DeltaTable.forPath(
            spark,
            S3_GOLD_FACT_SALES,
        )
    )


    (
        gold_delta
        .alias("target")

        .merge(
            affected_fact_sales.alias("source"),

            """
            target.invoice_id = source.invoice_id
            """
        )

        .whenMatchedUpdateAll()

        .whenNotMatchedInsertAll()

        .execute()
    )


    print()
    print(
        "Gold MERGE completed."
    )


# ============================================================
# 10. RELOAD GOLD
# ============================================================

fact_sales = (
    spark.read
    .format("delta")
    .load(
        S3_GOLD_FACT_SALES
    )
)


gold_rows_after = (
    fact_sales.count()
)


gold_summary_after = (
    fact_sales

    .agg(
        F.count("*").alias(
            "invoices"
        ),

        F.countDistinct(
            "customer_id"
        ).alias(
            "customers"
        ),

        F.round(
            F.sum(
                "net_revenue"
            ),
            2,
        ).alias(
            "net_revenue"
        ),

        F.min(
            "invoice_date"
        ).alias(
            "min_invoice_date"
        ),

        F.max(
            "invoice_date"
        ).alias(
            "max_invoice_date"
        ),
    )
)


# ============================================================
# 11. GLOBAL QUALITY CHECKS
# ============================================================

duplicate_gold_invoices = (
    fact_sales

    .groupBy(
        "invoice_id"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


generic_gold_customers = (
    fact_sales

    .filter(
        F.col("customer_id")
        ==
        GENERIC_CUSTOMER_ID
    )

    .count()
)


print()
print("=" * 90)
print("GOLD FACT_SALES VALIDATION")
print("=" * 90)

print(
    f"Gold invoices after:        "
    f"{gold_rows_after:,}"
)

print(
    f"Duplicate invoice IDs:      "
    f"{duplicate_gold_invoices:,}"
)

print(
    f"Generic 000000 invoices:    "
    f"{generic_gold_customers:,}"
)


if duplicate_gold_invoices != 0:

    raise ValueError(
        "Duplicate invoice IDs detected in Gold."
    )


if generic_gold_customers != 0:

    raise ValueError(
        "Generic customer 000000 found in commercial Gold."
    )


display(
    gold_summary_after
)


# ============================================================
# 12. INVOICE TYPES
# ============================================================

print()
print("=" * 90)
print("GOLD INVOICE TYPES")
print("=" * 90)


display(
    fact_sales

    .groupBy(
        "invoice_type"
    )

    .agg(
        F.count("*").alias(
            "invoices"
        ),

        F.round(
            F.sum(
                "net_revenue"
            ),
            2,
        ).alias(
            "net_revenue"
        ),
    )

    .orderBy(
        "invoice_type"
    )
)


print()
print(
    "PRODUCTION SILVER -> GOLD FACT_SALES "
    "COMPLETED SUCCESSFULLY."
)

# COMMAND ----------

# ============================================================
# CELL 12 - EXISTING GOLD CUSTOMER_MONTHLY CHECK
#
# READ ONLY
#
# PURPOSE:
# - Load the validated customer_monthly Gold table.
# - Inspect its exact schema.
# - Validate totals and date range.
#
# This cell DOES NOT modify data.
# ============================================================

from pyspark.sql import functions as F


# ============================================================
# 1. LOAD EXISTING CUSTOMER_MONTHLY
# ============================================================

customer_monthly = (
    spark.read
    .format("delta")
    .load(S3_GOLD_CUSTOMER_MONTHLY)
)


# ============================================================
# 2. BASIC COUNTS
# ============================================================

monthly_rows = (
    customer_monthly.count()
)

monthly_customers = (
    customer_monthly
    .select("customer_id")
    .distinct()
    .count()
)


print("=" * 90)
print("EXISTING GOLD CUSTOMER_MONTHLY CHECK")
print("=" * 90)

print(
    f"Rows:           "
    f"{monthly_rows:,}"
)

print(
    f"Customers:      "
    f"{monthly_customers:,}"
)

print(
    f"Columns:        "
    f"{len(customer_monthly.columns)}"
)


# ============================================================
# 3. COLUMN CONTRACT
# ============================================================

print()
print("=" * 90)
print("CUSTOMER_MONTHLY COLUMNS")
print("=" * 90)

for column_name in customer_monthly.columns:
    print(
        f" - {column_name}"
    )


print()
print("=" * 90)
print("CUSTOMER_MONTHLY SCHEMA")
print("=" * 90)

customer_monthly.printSchema()


# ============================================================
# 4. BUSINESS SUMMARY
# ============================================================

monthly_summary = (
    customer_monthly

    .agg(
        F.count("*").alias(
            "rows"
        ),

        F.countDistinct(
            "customer_id"
        ).alias(
            "customers"
        ),

        F.min(
            "month_start"
        ).alias(
            "min_month"
        ),

        F.max(
            "month_start"
        ).alias(
            "max_month"
        ),

        F.round(
            F.sum(
                "net_revenue"
            ),
            2,
        ).alias(
            "net_revenue"
        ),
    )
)


print()
print("=" * 90)
print("CUSTOMER_MONTHLY BUSINESS SUMMARY")
print("=" * 90)

display(
    monthly_summary
)


# ============================================================
# 5. ACTIVE / INACTIVE MONTHS
# ============================================================

if "is_active_month" in customer_monthly.columns:

    print()
    print("=" * 90)
    print("ACTIVE / INACTIVE MONTHS")
    print("=" * 90)

    display(
        customer_monthly

        .groupBy(
            "is_active_month"
        )

        .agg(
            F.count("*").alias(
                "rows"
            )
        )

        .orderBy(
            "is_active_month"
        )
    )


# ============================================================
# 6. NULL CHECK
# ============================================================

critical_columns = [
    "customer_id",
    "month_start",
    "net_revenue",
]


critical_null_condition = None


for column_name in critical_columns:

    condition = (
        F.col(column_name).isNull()
    )

    if critical_null_condition is None:
        critical_null_condition = condition

    else:
        critical_null_condition = (
            critical_null_condition
            |
            condition
        )


critical_nulls = (
    customer_monthly

    .filter(
        critical_null_condition
    )

    .count()
)


print()
print("=" * 90)
print("CUSTOMER_MONTHLY QUALITY")
print("=" * 90)

print(
    f"Critical null rows: "
    f"{critical_nulls:,}"
)


# ============================================================
# 7. DUPLICATE CUSTOMER-MONTH CHECK
# ============================================================

duplicate_customer_months = (
    customer_monthly

    .groupBy(
        "customer_id",
        "month_start",
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


print(
    f"Duplicate customer-months: "
    f"{duplicate_customer_months:,}"
)


# ============================================================
# 8. SAMPLE
# ============================================================

print()
print("=" * 90)
print("CUSTOMER_MONTHLY SAMPLE")
print("=" * 90)

display(
    customer_monthly

    .orderBy(
        "customer_id",
        "month_start",
    )

    .limit(20)
)


print()
print(
    "READ-ONLY CUSTOMER_MONTHLY CHECK COMPLETED."
)

# COMMAND ----------

# ============================================================
# CELL 13 - EXACT CUSTOMER_MONTHLY CONTRACT
# READ ONLY
# ============================================================

print("CUSTOMER_MONTHLY COLUMNS:")
print(customer_monthly.columns)

print()
print("CUSTOMER_MONTHLY SCHEMA:")
customer_monthly.printSchema()

# COMMAND ----------

# ============================================================
# CELL 14 - VALIDATE FACT_SALES -> CUSTOMER_MONTHLY REBUILD
#
# READ ONLY
#
# PURPOSE:
# - Rebuild customer_monthly from validated fact_sales.
# - Recreate zero-activity months for every customer.
# - Recreate recency correctly, including partial current month.
# - Compare the result row-by-row against existing Gold.
#
# NOTHING IS WRITTEN.
# ============================================================

from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ============================================================
# 1. GLOBAL OBSERVATION DATE
#
# Latest commercial invoice currently available.
# ============================================================

latest_invoice_date = (
    fact_sales
    .agg(
        F.max("invoice_date")
        .alias("latest_invoice_date")
    )
    .first()["latest_invoice_date"]
)


latest_month = (
    fact_sales
    .select(
        F.trunc(
            F.lit(latest_invoice_date),
            "month",
        ).alias("latest_month")
    )
    .first()["latest_month"]
)


print("=" * 90)
print("FACT_SALES -> CUSTOMER_MONTHLY REBUILD TEST")
print("=" * 90)

print(
    f"Latest invoice date: "
    f"{latest_invoice_date}"
)

print(
    f"Latest month:        "
    f"{latest_month}"
)


# ============================================================
# 2. CUSTOMER UNIVERSE
#
# Customers enter the monthly panel when they have their
# first real PURCHASE.
#
# This avoids creating panels for IDs that only contain
# credits / zero-value documents.
# ============================================================

customer_start = (
    fact_sales

    .filter(
        F.col("invoice_type") == "PURCHASE"
    )

    .groupBy(
        "customer_id"
    )

    .agg(
        F.min(
            F.trunc(
                "invoice_date",
                "month",
            )
        ).alias(
            "first_month"
        )
    )
)


customer_universe_count = (
    customer_start.count()
)


print(
    f"Customer universe:   "
    f"{customer_universe_count:,}"
)


# ============================================================
# 3. BUILD COMPLETE CUSTOMER-MONTH PANEL
#
# One row per customer per month:
# first purchase month -> latest observed month.
# ============================================================

customer_calendar = (
    customer_start

    .withColumn(
        "month_start",

        F.explode(
            F.sequence(
                F.col("first_month"),
                F.lit(latest_month),
                F.expr("INTERVAL 1 MONTH"),
            )
        )
    )

    .select(
        "customer_id",
        "month_start",
    )
)


calendar_rows = (
    customer_calendar.count()
)


print(
    f"Calendar rows:       "
    f"{calendar_rows:,}"
)


# ============================================================
# 4. MONTHLY INVOICE AGGREGATION
# ============================================================

invoice_monthly = (
    fact_sales

    .withColumn(
        "month_start",
        F.trunc(
            "invoice_date",
            "month",
        )
    )

    .groupBy(
        "customer_id",
        "month_start",
    )

    .agg(
        F.sum(
            "net_revenue"
        ).alias(
            "net_revenue"
        ),

        F.sum(
            F.when(
                F.col("invoice_type") == "PURCHASE",
                F.col("net_revenue"),
            )
            .otherwise(
                F.lit(0.0)
            )
        ).alias(
            "purchase_revenue"
        ),

        F.sum(
            F.when(
                F.col("invoice_type") == "PURCHASE",
                F.lit(1),
            )
            .otherwise(
                F.lit(0)
            )
        ).cast("long").alias(
            "purchase_count"
        ),

        F.sum(
            F.when(
                F.col("invoice_type") == "CREDIT_NOTE",
                F.lit(1),
            )
            .otherwise(
                F.lit(0)
            )
        ).cast("long").alias(
            "credit_note_count"
        ),

        F.count(
            "*"
        ).cast("long").alias(
            "invoice_count"
        ),

        F.max(
            F.when(
                F.col("invoice_type") == "PURCHASE",
                F.col("invoice_date"),
            )
        ).alias(
            "last_purchase_in_month"
        ),
    )
)


# ============================================================
# 5. JOIN ACTIVITY INTO COMPLETE PANEL
# ============================================================

rebuilt_customer_monthly = (
    customer_calendar

    .join(
        invoice_monthly,
        on=[
            "customer_id",
            "month_start",
        ],
        how="left",
    )

    .fillna(
        {
            "net_revenue": 0.0,
            "purchase_revenue": 0.0,
            "purchase_count": 0,
            "credit_note_count": 0,
            "invoice_count": 0,
        }
    )
)


# ============================================================
# 6. CUMULATIVE LAST PURCHASE DATE
# ============================================================

customer_month_window = (
    Window
    .partitionBy(
        "customer_id"
    )
    .orderBy(
        "month_start"
    )
    .rowsBetween(
        Window.unboundedPreceding,
        Window.currentRow,
    )
)


rebuilt_customer_monthly = (
    rebuilt_customer_monthly

    .withColumn(
        "last_purchase_date",

        F.max(
            "last_purchase_in_month"
        ).over(
            customer_month_window
        )
    )
)


# ============================================================
# 7. OBSERVATION DATE
#
# Historical months:
#   last calendar day of month
#
# Current partial month:
#   latest invoice date actually observed
#
# least(last_day(month), latest_invoice_date)
# reproduces both cases.
# ============================================================

rebuilt_customer_monthly = (
    rebuilt_customer_monthly

    .withColumn(
        "observation_date",

        F.least(
            F.last_day(
                "month_start"
            ),
            F.lit(
                latest_invoice_date
            ),
        )
    )
)


# ============================================================
# 8. RECENCY
# ============================================================

rebuilt_customer_monthly = (
    rebuilt_customer_monthly

    .withColumn(
        "days_since_last_purchase",

        F.datediff(
            F.col(
                "observation_date"
            ),
            F.col(
                "last_purchase_date"
            ),
        )
    )
)


# ============================================================
# 9. CALENDAR FEATURES
# ============================================================

rebuilt_customer_monthly = (
    rebuilt_customer_monthly

    .withColumn(
        "year",
        F.year(
            "month_start"
        ),
    )

    .withColumn(
        "month",
        F.month(
            "month_start"
        ),
    )

    .withColumn(
        "year_month",
        F.date_format(
            "month_start",
            "yyyy-MM",
        ),
    )

    .withColumn(
        "is_active_month",

        F.when(
            F.col("purchase_count") > 0,
            F.lit(1),
        )
        .otherwise(
            F.lit(0)
        )
        .cast("int")
    )
)


# ============================================================
# 10. EXACT EXISTING CONTRACT
# ============================================================

rebuilt_customer_monthly = (
    rebuilt_customer_monthly

    .select(
        *customer_monthly.columns
    )
)


# ============================================================
# 11. BASIC COUNTS
# ============================================================

existing_rows = (
    customer_monthly.count()
)

rebuilt_rows = (
    rebuilt_customer_monthly.count()
)


existing_customers = (
    customer_monthly
    .select("customer_id")
    .distinct()
    .count()
)

rebuilt_customers = (
    rebuilt_customer_monthly
    .select("customer_id")
    .distinct()
    .count()
)


print()
print("=" * 90)
print("CUSTOMER_MONTHLY REBUILD SUMMARY")
print("=" * 90)

print(
    f"Existing rows:       "
    f"{existing_rows:,}"
)

print(
    f"Rebuilt rows:        "
    f"{rebuilt_rows:,}"
)

print(
    f"Existing customers:  "
    f"{existing_customers:,}"
)

print(
    f"Rebuilt customers:   "
    f"{rebuilt_customers:,}"
)


# ============================================================
# 12. BUSINESS SUMMARY
# ============================================================

print()
print("=" * 90)
print("REBUILT BUSINESS SUMMARY")
print("=" * 90)


display(
    rebuilt_customer_monthly

    .agg(
        F.count("*").alias(
            "rows"
        ),

        F.countDistinct(
            "customer_id"
        ).alias(
            "customers"
        ),

        F.min(
            "month_start"
        ).alias(
            "min_month"
        ),

        F.max(
            "month_start"
        ).alias(
            "max_month"
        ),

        F.round(
            F.sum(
                "net_revenue"
            ),
            2,
        ).alias(
            "net_revenue"
        ),
    )
)


# ============================================================
# 13. CUSTOMER-MONTH KEY COMPARISON
# ============================================================

existing_keys = (
    customer_monthly
    .select(
        "customer_id",
        "month_start",
    )
)


rebuilt_keys = (
    rebuilt_customer_monthly
    .select(
        "customer_id",
        "month_start",
    )
)


missing_customer_months = (
    existing_keys
    .join(
        rebuilt_keys,
        on=[
            "customer_id",
            "month_start",
        ],
        how="left_anti",
    )
)


extra_customer_months = (
    rebuilt_keys
    .join(
        existing_keys,
        on=[
            "customer_id",
            "month_start",
        ],
        how="left_anti",
    )
)


missing_count = (
    missing_customer_months.count()
)

extra_count = (
    extra_customer_months.count()
)


# ============================================================
# 14. FULL FIELD COMPARISON
#
# eqNullSafe is important because some monthly fields
# legitimately contain NULL values.
# ============================================================

e = (
    customer_monthly
    .alias("e")
)

r = (
    rebuilt_customer_monthly
    .alias("r")
)


joined = (
    e.join(
        r,
        on=(
            (F.col("e.customer_id") == F.col("r.customer_id"))
            &
            (F.col("e.month_start") == F.col("r.month_start"))
        ),
        how="inner",
    )
)


mismatch_condition = (

    ~F.col("e.net_revenue")
    .eqNullSafe(
        F.col("r.net_revenue")
    )

    |

    ~F.col("e.purchase_revenue")
    .eqNullSafe(
        F.col("r.purchase_revenue")
    )

    |

    ~F.col("e.purchase_count")
    .eqNullSafe(
        F.col("r.purchase_count")
    )

    |

    ~F.col("e.credit_note_count")
    .eqNullSafe(
        F.col("r.credit_note_count")
    )

    |

    ~F.col("e.invoice_count")
    .eqNullSafe(
        F.col("r.invoice_count")
    )

    |

    ~F.col("e.last_purchase_in_month")
    .eqNullSafe(
        F.col("r.last_purchase_in_month")
    )

    |

    ~F.col("e.last_purchase_date")
    .eqNullSafe(
        F.col("r.last_purchase_date")
    )

    |

    ~F.col("e.observation_date")
    .eqNullSafe(
        F.col("r.observation_date")
    )

    |

    ~F.col("e.days_since_last_purchase")
    .eqNullSafe(
        F.col("r.days_since_last_purchase")
    )

    |

    ~F.col("e.year")
    .eqNullSafe(
        F.col("r.year")
    )

    |

    ~F.col("e.month")
    .eqNullSafe(
        F.col("r.month")
    )

    |

    ~F.col("e.year_month")
    .eqNullSafe(
        F.col("r.year_month")
    )

    |

    ~F.col("e.is_active_month")
    .eqNullSafe(
        F.col("r.is_active_month")
    )
)


field_mismatches = (
    joined
    .filter(
        mismatch_condition
    )
)


field_mismatch_count = (
    field_mismatches.count()
)


# ============================================================
# 15. ACTIVE / INACTIVE VALIDATION
# ============================================================

print()
print("=" * 90)
print("REBUILT ACTIVE / INACTIVE MONTHS")
print("=" * 90)


display(
    rebuilt_customer_monthly

    .groupBy(
        "is_active_month"
    )

    .agg(
        F.count("*").alias(
            "rows"
        )
    )

    .orderBy(
        "is_active_month"
    )
)


# ============================================================
# 16. FINAL VALIDATION
# ============================================================

print()
print("=" * 90)
print("CUSTOMER_MONTHLY REBUILD VALIDATION")
print("=" * 90)

print(
    f"Expected rows:             "
    f"{existing_rows:,}"
)

print(
    f"Rebuilt rows:              "
    f"{rebuilt_rows:,}"
)

print(
    f"Missing customer-months:   "
    f"{missing_count:,}"
)

print(
    f"Extra customer-months:     "
    f"{extra_count:,}"
)

print(
    f"Field mismatches:          "
    f"{field_mismatch_count:,}"
)


# ============================================================
# 17. SHOW DIFFERENCES ONLY IF NEEDED
# ============================================================

if missing_count > 0:

    print()
    print("MISSING CUSTOMER-MONTHS:")

    display(
        missing_customer_months
        .limit(20)
    )


if extra_count > 0:

    print()
    print("EXTRA CUSTOMER-MONTHS:")

    display(
        extra_customer_months
        .limit(20)
    )


if field_mismatch_count > 0:

    print()
    print("FIELD MISMATCHES:")

    display(
        field_mismatches
        .limit(20)
    )


# ============================================================
# 18. FINAL RESULT
# ============================================================

if (
    rebuilt_rows == existing_rows
    and
    rebuilt_customers == existing_customers
    and
    missing_count == 0
    and
    extra_count == 0
    and
    field_mismatch_count == 0
):

    print()
    print(
        "CUSTOMER_MONTHLY REBUILD VALIDATED EXACTLY "
        "AGAINST EXISTING GOLD."
    )

else:

    print()
    print(
        "CUSTOMER_MONTHLY REBUILD DOES NOT YET "
        "MATCH EXISTING GOLD."
    )

    print(
        "DO NOT WRITE CUSTOMER_MONTHLY YET."
    )

# COMMAND ----------

# ============================================================
# CELL 15 - DIAGNOSE CUSTOMER_MONTHLY MISMATCHES
#
# READ ONLY
#
# PURPOSE:
# Determine whether the 627 mismatches are only floating-point
# precision differences or real business-logic differences.
#
# NOTHING IS WRITTEN.
# ============================================================

from pyspark.sql import functions as F


# ============================================================
# 1. JOIN EXISTING VS REBUILT
# ============================================================

cm_compare = (
    customer_monthly.alias("e")

    .join(
        rebuilt_customer_monthly.alias("r"),
        on=(
            (F.col("e.customer_id") == F.col("r.customer_id"))
            &
            (F.col("e.month_start") == F.col("r.month_start"))
        ),
        how="inner",
    )
)


# ============================================================
# 2. NUMERIC TOLERANCE
#
# We do NOT require bit-for-bit equality for doubles.
# ============================================================

TOLERANCE = 0.000001


def double_mismatch(existing_col, rebuilt_col):

    e_col = F.col(existing_col)
    r_col = F.col(rebuilt_col)

    return (
        F.when(
            e_col.isNull() & r_col.isNull(),
            F.lit(False),
        )
        .when(
            e_col.isNull() | r_col.isNull(),
            F.lit(True),
        )
        .otherwise(
            F.abs(e_col - r_col)
            >
            F.lit(TOLERANCE)
        )
    )


# ============================================================
# 3. FIELD-BY-FIELD MISMATCH COUNTS
# ============================================================

diagnostics = (
    cm_compare

    .agg(

        # ----------------------------------------------------
        # DOUBLE FIELDS
        # ----------------------------------------------------

        F.sum(
            F.when(
                double_mismatch(
                    "e.net_revenue",
                    "r.net_revenue",
                ),
                1,
            ).otherwise(0)
        ).alias(
            "net_revenue_mismatches"
        ),

        F.sum(
            F.when(
                double_mismatch(
                    "e.purchase_revenue",
                    "r.purchase_revenue",
                ),
                1,
            ).otherwise(0)
        ).alias(
            "purchase_revenue_mismatches"
        ),

        # ----------------------------------------------------
        # INTEGER / DATE / STRING FIELDS
        # ----------------------------------------------------

        F.sum(
            F.when(
                ~F.col("e.purchase_count")
                .eqNullSafe(
                    F.col("r.purchase_count")
                ),
                1,
            ).otherwise(0)
        ).alias(
            "purchase_count_mismatches"
        ),

        F.sum(
            F.when(
                ~F.col("e.credit_note_count")
                .eqNullSafe(
                    F.col("r.credit_note_count")
                ),
                1,
            ).otherwise(0)
        ).alias(
            "credit_note_count_mismatches"
        ),

        F.sum(
            F.when(
                ~F.col("e.invoice_count")
                .eqNullSafe(
                    F.col("r.invoice_count")
                ),
                1,
            ).otherwise(0)
        ).alias(
            "invoice_count_mismatches"
        ),

        F.sum(
            F.when(
                ~F.col("e.last_purchase_in_month")
                .eqNullSafe(
                    F.col("r.last_purchase_in_month")
                ),
                1,
            ).otherwise(0)
        ).alias(
            "last_purchase_in_month_mismatches"
        ),

        F.sum(
            F.when(
                ~F.col("e.last_purchase_date")
                .eqNullSafe(
                    F.col("r.last_purchase_date")
                ),
                1,
            ).otherwise(0)
        ).alias(
            "last_purchase_date_mismatches"
        ),

        F.sum(
            F.when(
                ~F.col("e.observation_date")
                .eqNullSafe(
                    F.col("r.observation_date")
                ),
                1,
            ).otherwise(0)
        ).alias(
            "observation_date_mismatches"
        ),

        F.sum(
            F.when(
                ~F.col("e.days_since_last_purchase")
                .eqNullSafe(
                    F.col("r.days_since_last_purchase")
                ),
                1,
            ).otherwise(0)
        ).alias(
            "days_since_last_purchase_mismatches"
        ),

        F.sum(
            F.when(
                ~F.col("e.year")
                .eqNullSafe(
                    F.col("r.year")
                ),
                1,
            ).otherwise(0)
        ).alias(
            "year_mismatches"
        ),

        F.sum(
            F.when(
                ~F.col("e.month")
                .eqNullSafe(
                    F.col("r.month")
                ),
                1,
            ).otherwise(0)
        ).alias(
            "month_mismatches"
        ),

        F.sum(
            F.when(
                ~F.col("e.year_month")
                .eqNullSafe(
                    F.col("r.year_month")
                ),
                1,
            ).otherwise(0)
        ).alias(
            "year_month_mismatches"
        ),

        F.sum(
            F.when(
                ~F.col("e.is_active_month")
                .eqNullSafe(
                    F.col("r.is_active_month")
                ),
                1,
            ).otherwise(0)
        ).alias(
            "is_active_month_mismatches"
        ),

        # ----------------------------------------------------
        # MAXIMUM FLOAT DIFFERENCES
        # ----------------------------------------------------

        F.max(
            F.abs(
                F.col("e.net_revenue")
                -
                F.col("r.net_revenue")
            )
        ).alias(
            "max_net_revenue_abs_diff"
        ),

        F.max(
            F.abs(
                F.col("e.purchase_revenue")
                -
                F.col("r.purchase_revenue")
            )
        ).alias(
            "max_purchase_revenue_abs_diff"
        ),
    )
)


print("=" * 90)
print("CUSTOMER_MONTHLY MISMATCH DIAGNOSTICS")
print("=" * 90)

print(
    f"Double tolerance: {TOLERANCE}"
)

display(
    diagnostics
)


# ============================================================
# 4. VALIDATE AGAIN USING CORRECT DOUBLE TOLERANCE
# ============================================================

correct_mismatch_condition = (

    double_mismatch(
        "e.net_revenue",
        "r.net_revenue",
    )

    |

    double_mismatch(
        "e.purchase_revenue",
        "r.purchase_revenue",
    )

    |

    ~F.col("e.purchase_count")
    .eqNullSafe(
        F.col("r.purchase_count")
    )

    |

    ~F.col("e.credit_note_count")
    .eqNullSafe(
        F.col("r.credit_note_count")
    )

    |

    ~F.col("e.invoice_count")
    .eqNullSafe(
        F.col("r.invoice_count")
    )

    |

    ~F.col("e.last_purchase_in_month")
    .eqNullSafe(
        F.col("r.last_purchase_in_month")
    )

    |

    ~F.col("e.last_purchase_date")
    .eqNullSafe(
        F.col("r.last_purchase_date")
    )

    |

    ~F.col("e.observation_date")
    .eqNullSafe(
        F.col("r.observation_date")
    )

    |

    ~F.col("e.days_since_last_purchase")
    .eqNullSafe(
        F.col("r.days_since_last_purchase")
    )

    |

    ~F.col("e.year")
    .eqNullSafe(
        F.col("r.year")
    )

    |

    ~F.col("e.month")
    .eqNullSafe(
        F.col("r.month")
    )

    |

    ~F.col("e.year_month")
    .eqNullSafe(
        F.col("r.year_month")
    )

    |

    ~F.col("e.is_active_month")
    .eqNullSafe(
        F.col("r.is_active_month")
    )
)


true_mismatches = (
    cm_compare
    .filter(
        correct_mismatch_condition
    )
)


true_mismatch_count = (
    true_mismatches.count()
)


print()
print("=" * 90)
print("TOLERANCE-AWARE VALIDATION")
print("=" * 90)

print(
    f"Rows compared:       "
    f"{cm_compare.count():,}"
)

print(
    f"True mismatches:     "
    f"{true_mismatch_count:,}"
)


if true_mismatch_count == 0:

    print()
    print(
        "CUSTOMER_MONTHLY REBUILD MATCHES EXISTING GOLD."
    )

    print(
        "The previous 627 differences were only "
        "floating-point representation differences."
    )

else:

    print()
    print(
        "REAL BUSINESS-LOGIC DIFFERENCES STILL EXIST."
    )

    print()
    print("EXAMPLES:")

    display(
        true_mismatches.limit(20)
    )

# COMMAND ----------

# ============================================================
# CELL 16 - PRODUCTION FACT_SALES -> CUSTOMER_MONTHLY
#
# PURPOSE:
# - Persist the validated customer_monthly reconstruction.
# - Rebuild the complete customer-month panel only when
#   commercial invoices changed during this pipeline run.
# - Perform quality checks BEFORE and AFTER the write.
#
# CURRENT RUN:
# affected_invoice_count == 0
# -> customer_monthly is NOT modified.
# ============================================================

from pyspark.sql import functions as F


# ============================================================
# 1. CURRENT STATE
# ============================================================

customer_monthly_before = (
    spark.read
    .format("delta")
    .load(S3_GOLD_CUSTOMER_MONTHLY)
)


monthly_rows_before = (
    customer_monthly_before.count()
)


monthly_revenue_before = (
    customer_monthly_before

    .agg(
        F.round(
            F.sum("net_revenue"),
            2,
        ).alias(
            "net_revenue"
        )
    )

    .first()["net_revenue"]
)


print("=" * 90)
print("PRODUCTION FACT_SALES -> CUSTOMER_MONTHLY")
print("=" * 90)

print(
    f"Rows before:              "
    f"{monthly_rows_before:,}"
)

print(
    f"Net revenue before:       "
    f"{monthly_revenue_before:,.2f}"
)

print(
    f"Affected invoices:        "
    f"{affected_invoice_count:,}"
)


# ============================================================
# 2. PREPARE VALIDATED REBUILD
#
# rebuilt_customer_monthly was created in CELL 14 and
# validated in CELL 15.
# ============================================================

customer_monthly_candidate = (
    rebuilt_customer_monthly

    .select(
        *customer_monthly_before.columns
    )
)


candidate_rows = (
    customer_monthly_candidate.count()
)


candidate_customers = (
    customer_monthly_candidate

    .select(
        "customer_id"
    )

    .distinct()

    .count()
)


candidate_revenue = (
    customer_monthly_candidate

    .agg(
        F.round(
            F.sum("net_revenue"),
            2,
        ).alias(
            "net_revenue"
        )
    )

    .first()["net_revenue"]
)


fact_sales_revenue = (
    fact_sales

    .agg(
        F.round(
            F.sum("net_revenue"),
            2,
        ).alias(
            "net_revenue"
        )
    )

    .first()["net_revenue"]
)


# ============================================================
# 3. PRE-WRITE QUALITY CHECKS
# ============================================================

duplicate_customer_months_candidate = (
    customer_monthly_candidate

    .groupBy(
        "customer_id",
        "month_start",
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


critical_null_condition = (
    F.col("customer_id").isNull()
    |
    F.col("month_start").isNull()
    |
    F.col("net_revenue").isNull()
    |
    F.col("purchase_count").isNull()
    |
    F.col("invoice_count").isNull()
    |
    F.col("observation_date").isNull()
)


critical_null_rows_candidate = (
    customer_monthly_candidate

    .filter(
        critical_null_condition
    )

    .count()
)


candidate_date_summary = (
    customer_monthly_candidate

    .agg(
        F.min(
            "month_start"
        ).alias(
            "min_month"
        ),

        F.max(
            "month_start"
        ).alias(
            "max_month"
        ),

        F.max(
            "observation_date"
        ).alias(
            "max_observation_date"
        ),
    )

    .first()
)


print()
print("=" * 90)
print("PRE-WRITE CUSTOMER_MONTHLY VALIDATION")
print("=" * 90)

print(
    f"Candidate rows:           "
    f"{candidate_rows:,}"
)

print(
    f"Candidate customers:      "
    f"{candidate_customers:,}"
)

print(
    f"Candidate net revenue:    "
    f"{candidate_revenue:,.2f}"
)

print(
    f"fact_sales net revenue:   "
    f"{fact_sales_revenue:,.2f}"
)

print(
    f"Duplicate customer-months:"
    f" {duplicate_customer_months_candidate:,}"
)

print(
    f"Critical null rows:       "
    f"{critical_null_rows_candidate:,}"
)

print(
    f"Month range:              "
    f"{candidate_date_summary['min_month']} "
    f"-> "
    f"{candidate_date_summary['max_month']}"
)

print(
    f"Observation date:         "
    f"{candidate_date_summary['max_observation_date']}"
)


# ============================================================
# 4. HARD VALIDATION GATES
# ============================================================

if duplicate_customer_months_candidate != 0:

    raise ValueError(
        "Duplicate customer-month rows detected "
        "before customer_monthly write."
    )


if critical_null_rows_candidate != 0:

    raise ValueError(
        "Critical null rows detected "
        "before customer_monthly write."
    )


if abs(
    float(candidate_revenue)
    -
    float(fact_sales_revenue)
) > 0.01:

    raise ValueError(
        "Revenue is not preserved between "
        "fact_sales and customer_monthly."
    )


if candidate_rows <= 0:

    raise ValueError(
        "customer_monthly candidate is empty."
    )


# ============================================================
# 5. WRITE ONLY IF COMMERCIAL DATA CHANGED
#
# Full snapshot overwrite is intentional:
#
# If the observation horizon advances, inactive months may
# need to be added for many existing customers even when
# those individual customers made no purchase.
#
# Therefore a full validated snapshot is safer than a
# customer-level incremental MERGE here.
# ============================================================

if affected_invoice_count == 0:

    print()
    print(
        "No commercial invoice changes detected."
    )

    print(
        "customer_monthly will not be modified."
    )


else:

    print()
    print(
        "Commercial invoice changes detected."
    )

    print(
        "Writing validated customer_monthly snapshot..."
    )


    (
        customer_monthly_candidate.write

        .format("delta")

        .mode("overwrite")

        .option(
            "overwriteSchema",
            "false",
        )

        .save(
            S3_GOLD_CUSTOMER_MONTHLY
        )
    )


    print(
        "customer_monthly snapshot written successfully."
    )


# ============================================================
# 6. RELOAD PRODUCTION TABLE
# ============================================================

customer_monthly = (
    spark.read
    .format("delta")
    .load(
        S3_GOLD_CUSTOMER_MONTHLY
    )
)


# ============================================================
# 7. POST-WRITE VALIDATION
# ============================================================

monthly_rows_after = (
    customer_monthly.count()
)


monthly_customers_after = (
    customer_monthly

    .select(
        "customer_id"
    )

    .distinct()

    .count()
)


duplicate_customer_months_after = (
    customer_monthly

    .groupBy(
        "customer_id",
        "month_start",
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


critical_null_rows_after = (
    customer_monthly

    .filter(
        critical_null_condition
    )

    .count()
)


monthly_summary_after = (
    customer_monthly

    .agg(
        F.count("*").alias(
            "rows"
        ),

        F.countDistinct(
            "customer_id"
        ).alias(
            "customers"
        ),

        F.min(
            "month_start"
        ).alias(
            "min_month"
        ),

        F.max(
            "month_start"
        ).alias(
            "max_month"
        ),

        F.max(
            "observation_date"
        ).alias(
            "observation_date"
        ),

        F.round(
            F.sum(
                "net_revenue"
            ),
            2,
        ).alias(
            "net_revenue"
        ),
    )
)


print()
print("=" * 90)
print("CUSTOMER_MONTHLY PRODUCTION VALIDATION")
print("=" * 90)

print(
    f"Rows after:               "
    f"{monthly_rows_after:,}"
)

print(
    f"Customers after:          "
    f"{monthly_customers_after:,}"
)

print(
    f"Duplicate customer-months:"
    f" {duplicate_customer_months_after:,}"
)

print(
    f"Critical null rows:       "
    f"{critical_null_rows_after:,}"
)


if duplicate_customer_months_after != 0:

    raise ValueError(
        "Duplicate customer-month rows detected "
        "after production write."
    )


if critical_null_rows_after != 0:

    raise ValueError(
        "Critical null rows detected "
        "after production write."
    )


display(
    monthly_summary_after
)


# ============================================================
# 8. ACTIVE / INACTIVE MONTHS
# ============================================================

print()
print("=" * 90)
print("CUSTOMER_MONTHLY ACTIVITY")
print("=" * 90)


display(
    customer_monthly

    .groupBy(
        "is_active_month"
    )

    .agg(
        F.count("*").alias(
            "rows"
        )
    )

    .orderBy(
        "is_active_month"
    )
)


print()
print(
    "PRODUCTION FACT_SALES -> CUSTOMER_MONTHLY "
    "COMPLETED SUCCESSFULLY."
)

# COMMAND ----------

# ============================================================
# CELL 17 - EXISTING CURRENT SCORING FEATURES CHECK
#
# READ ONLY
#
# PURPOSE:
# - Load the validated current scoring feature table.
# - Inspect its exact schema.
# - Confirm snapshot, eligibility and 43-feature contract.
#
# NOTHING IS WRITTEN.
# ============================================================

from pyspark.sql import functions as F


# ============================================================
# 1. LOAD EXISTING CURRENT SCORING FEATURES
# ============================================================

current_scoring_features = (
    spark.read
    .format("delta")
    .load(
        S3_CURRENT_SCORING_FEATURES
    )
)


# ============================================================
# 2. BASIC CONTRACT
# ============================================================

current_rows = (
    current_scoring_features.count()
)

current_customers = (
    current_scoring_features
    .select("customer_id")
    .distinct()
    .count()
)


print("=" * 90)
print("EXISTING CURRENT SCORING FEATURES CHECK")
print("=" * 90)

print(
    f"Rows:              "
    f"{current_rows:,}"
)

print(
    f"Customers:         "
    f"{current_customers:,}"
)

print(
    f"Columns:           "
    f"{len(current_scoring_features.columns)}"
)

print(
    f"Model features:    "
    f"{len(FEATURE_COLUMNS)}"
)


# ============================================================
# 3. EXACT COLUMN CONTRACT
# ============================================================

print()
print("=" * 90)
print("CURRENT SCORING COLUMNS")
print("=" * 90)

print(
    current_scoring_features.columns
)


print()
print("=" * 90)
print("CURRENT SCORING SCHEMA")
print("=" * 90)

current_scoring_features.printSchema()


# ============================================================
# 4. FEATURE CONTRACT VALIDATION
# ============================================================

missing_features = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in current_scoring_features.columns
]


unexpected_feature_duplicates = (
    len(FEATURE_COLUMNS)
    !=
    len(set(FEATURE_COLUMNS))
)


print()
print("=" * 90)
print("FEATURE CONTRACT")
print("=" * 90)

print(
    f"Missing model features: "
    f"{missing_features}"
)

print(
    f"Duplicate configured features: "
    f"{unexpected_feature_duplicates}"
)


if missing_features:

    raise ValueError(
        f"Current scoring table is missing "
        f"model features: {missing_features}"
    )


# ============================================================
# 5. SNAPSHOT
# ============================================================

snapshot_summary = (
    current_scoring_features

    .agg(
        F.min(
            "month_start"
        ).alias(
            "min_snapshot"
        ),

        F.max(
            "month_start"
        ).alias(
            "max_snapshot"
        ),

        F.countDistinct(
            "month_start"
        ).alias(
            "snapshot_months"
        ),
    )
)


print()
print("=" * 90)
print("CURRENT SNAPSHOT")
print("=" * 90)

display(
    snapshot_summary
)


# ============================================================
# 6. CURRENT DROP DISTRIBUTION
# ============================================================

print()
print("=" * 90)
print("CURRENT DROP DISTRIBUTION")
print("=" * 90)

display(
    current_scoring_features

    .agg(
        F.min(
            "current_drop_pct"
        ).alias(
            "min_current_drop"
        ),

        F.avg(
            "current_drop_pct"
        ).alias(
            "avg_current_drop"
        ),

        F.max(
            "current_drop_pct"
        ).alias(
            "max_current_drop"
        ),
    )
)


# ============================================================
# 7. NULL CHECK ACROSS 43 MODEL FEATURES
# ============================================================

feature_null_expressions = [
    F.sum(
        F.when(
            F.col(feature).isNull(),
            1,
        ).otherwise(0)
    ).alias(feature)
    for feature in FEATURE_COLUMNS
]


feature_null_row = (
    current_scoring_features
    .agg(
        *feature_null_expressions
    )
    .first()
)


total_feature_nulls = sum(
    int(feature_null_row[feature] or 0)
    for feature in FEATURE_COLUMNS
)


print()
print("=" * 90)
print("FEATURE QUALITY")
print("=" * 90)

print(
    f"Nulls in 43 model features: "
    f"{total_feature_nulls:,}"
)


# ============================================================
# 8. DUPLICATE CUSTOMER CHECK
# ============================================================

duplicate_customers = (
    current_scoring_features

    .groupBy(
        "customer_id"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


print(
    f"Duplicate customers:          "
    f"{duplicate_customers:,}"
)


# ============================================================
# 9. SAMPLE
# ============================================================

print()
print("=" * 90)
print("CURRENT SCORING SAMPLE")
print("=" * 90)

display(
    current_scoring_features

    .select(
        "customer_id",
        "month_start",
        "current_drop_pct",
        "revenue_12m",
        "revenue_3m",
        "purchases_12m",
        "active_months_12m",
        "recency_days",
        "recency_ratio_12m",
        "revenue_change_3m",
    )

    .orderBy(
        "customer_id"
    )

    .limit(20)
)


print()
print(
    "READ-ONLY CURRENT SCORING FEATURES CHECK COMPLETED."
)

# COMMAND ----------

# ============================================================
# CELL 18 - VALIDATE CURRENT SCORING ELIGIBILITY + CORE FEATURES
#
# READ ONLY
#
# PURPOSE:
# - Reconstruct the operational scoring snapshot from
#   customer_monthly.
# - Reproduce the early-warning eligibility policy.
# - Compare customer membership and core features against
#   the existing validated current_scoring_features table.
#
# NOTHING IS WRITTEN.
# ============================================================

from pyspark.sql import functions as F


# ============================================================
# 1. DETERMINE OPERATIONAL SNAPSHOT
#
# Default: previous COMPLETE month.
# Optional job parameter: snapshot_override=YYYY-MM-01.
# ============================================================

latest_invoice_date_for_scoring = (
    fact_sales
    .agg(
        F.max("invoice_date").alias("latest_invoice_date")
    )
    .first()["latest_invoice_date"]
)


if latest_invoice_date_for_scoring is None:
    raise ValueError(
        "Cannot determine a scoring snapshot because fact_sales is empty."
    )

if SNAPSHOT_OVERRIDE:
    try:
        snapshot_month = date.fromisoformat(SNAPSHOT_OVERRIDE)
    except ValueError as exc:
        raise ValueError(
            "snapshot_override must use ISO format YYYY-MM-01."
        ) from exc

    if snapshot_month.day != 1:
        raise ValueError(
            "snapshot_override must be the first day of a month."
        )

    snapshot_source = "job parameter"
else:
    snapshot_month = (
        spark
        .range(1)
        .select(
            F.add_months(
                F.trunc(
                    F.lit(latest_invoice_date_for_scoring),
                    "month",
                ),
                -1,
            ).alias("snapshot_month")
        )
        .first()["snapshot_month"]
    )
    snapshot_source = "previous complete month"


print("=" * 90)
print("CURRENT SCORING ELIGIBILITY REBUILD")
print("=" * 90)

print(
    f"Latest invoice date: "
    f"{latest_invoice_date_for_scoring}"
)

print(
    f"Operational snapshot: "
    f"{snapshot_month}"
)

print(
    f"Snapshot source:      "
    f"{snapshot_source}"
)


# ============================================================
# 2. CUSTOMERS PRESENT AT SNAPSHOT
# ============================================================

snapshot_rows = (
    customer_monthly

    .filter(
        F.col("month_start")
        ==
        F.lit(snapshot_month)
    )

    .select(
        "customer_id",
        "days_since_last_purchase",
    )
)


customers_at_snapshot = (
    snapshot_rows.count()
)


print(
    f"Customers with snapshot row: "
    f"{customers_at_snapshot:,}"
)


# ============================================================
# 3. CUSTOMER FIRST MONTH
# ============================================================

customer_first_month = (
    customer_monthly

    .groupBy(
        "customer_id"
    )

    .agg(
        F.min(
            "month_start"
        ).alias(
            "first_month"
        )
    )
)


# ============================================================
# 4. LAST 12 MONTHS RELATIVE TO SNAPSHOT
#
# month_offset:
#   0 = July 2026
#   1 = June 2026
#   ...
#   11 = August 2025
# ============================================================

scoring_history = (
    customer_monthly

    .filter(
        (F.col("month_start") <= F.lit(snapshot_month))
        &
        (
            F.col("month_start")
            >=
            F.add_months(
                F.lit(snapshot_month),
                -11,
            )
        )
    )

    .withColumn(
        "month_offset",

        F.round(
            F.months_between(
                F.lit(snapshot_month),
                F.col("month_start"),
            )
        ).cast("int")
    )
)


# ============================================================
# 5. CORE ROLLING FEATURES
# ============================================================

core_features = (
    scoring_history

    .groupBy(
        "customer_id"
    )

    .agg(

        # ----------------------------------------------------
        # REVENUE
        # ----------------------------------------------------

        F.sum(
            F.when(
                F.col("month_offset") == 0,
                F.col("net_revenue"),
            ).otherwise(0.0)
        ).alias(
            "revenue_1m"
        ),

        F.sum(
            F.when(
                F.col("month_offset").between(0, 2),
                F.col("net_revenue"),
            ).otherwise(0.0)
        ).alias(
            "revenue_3m"
        ),

        F.sum(
            F.when(
                F.col("month_offset").between(0, 5),
                F.col("net_revenue"),
            ).otherwise(0.0)
        ).alias(
            "revenue_6m"
        ),

        F.sum(
            F.when(
                F.col("month_offset").between(0, 11),
                F.col("net_revenue"),
            ).otherwise(0.0)
        ).alias(
            "revenue_12m"
        ),


        # ----------------------------------------------------
        # PURCHASE COUNTS
        # ----------------------------------------------------

        F.sum(
            F.when(
                F.col("month_offset").between(0, 2),
                F.col("purchase_count"),
            ).otherwise(0)
        ).alias(
            "purchases_3m"
        ),

        F.sum(
            F.when(
                F.col("month_offset").between(0, 5),
                F.col("purchase_count"),
            ).otherwise(0)
        ).alias(
            "purchases_6m"
        ),

        F.sum(
            F.when(
                F.col("month_offset").between(0, 11),
                F.col("purchase_count"),
            ).otherwise(0)
        ).alias(
            "purchases_12m"
        ),


        # ----------------------------------------------------
        # ACTIVE MONTHS
        # ----------------------------------------------------

        F.sum(
            F.when(
                F.col("month_offset").between(0, 2),
                F.col("is_active_month"),
            ).otherwise(0)
        ).alias(
            "active_months_3m"
        ),

        F.sum(
            F.when(
                F.col("month_offset").between(0, 5),
                F.col("is_active_month"),
            ).otherwise(0)
        ).alias(
            "active_months_6m"
        ),

        F.sum(
            F.when(
                F.col("month_offset").between(0, 11),
                F.col("is_active_month"),
            ).otherwise(0)
        ).alias(
            "active_months_12m"
        ),


        # ----------------------------------------------------
        # EARLY-WARNING WINDOWS
        #
        # previous:
        #   t-5, t-4, t-3
        #
        # recent:
        #   t-2, t-1, t
        # ----------------------------------------------------

        F.sum(
            F.when(
                F.col("month_offset").between(3, 5),
                F.col("net_revenue"),
            ).otherwise(0.0)
        ).alias(
            "previous_3m_revenue"
        ),

        F.sum(
            F.when(
                F.col("month_offset").between(0, 2),
                F.col("net_revenue"),
            ).otherwise(0.0)
        ).alias(
            "recent_3m_revenue_feature"
        ),
    )
)


# ============================================================
# 6. ADD RECENCY + CUSTOMER AGE
# ============================================================

core_features = (
    core_features

    .join(
        snapshot_rows,
        on="customer_id",
        how="inner",
    )

    .join(
        customer_first_month,
        on="customer_id",
        how="left",
    )

    .withColumn(
        "month_start",
        F.lit(snapshot_month),
    )

    .withColumn(
        "recency_days",
        F.col(
            "days_since_last_purchase"
        ).cast("double"),
    )

    .withColumn(
        "customer_age_months",

        (
            F.floor(
                F.months_between(
                    F.lit(snapshot_month),
                    F.col("first_month"),
                )
            )
            +
            F.lit(1)
        ).cast("double")
    )

    .drop(
        "days_since_last_purchase",
        "first_month",
    )
)


# ============================================================
# 7. CURRENT DROP / REVENUE CHANGE
# ============================================================

core_features = (
    core_features

    .withColumn(
        "current_drop_pct",

        F.when(
            F.col("previous_3m_revenue") > 0,

            F.lit(1.0)
            -
            (
                F.col("recent_3m_revenue_feature")
                /
                F.col("previous_3m_revenue")
            )
        )
    )

    .withColumn(
        "revenue_change_3m",

        F.when(
            F.col("previous_3m_revenue") > 0,

            (
                F.col("recent_3m_revenue_feature")
                -
                F.col("previous_3m_revenue")
            )
            /
            F.col("previous_3m_revenue")
        )
    )
)


# ============================================================
# 8. EARLY-WARNING ELIGIBILITY
#
# Customer must:
# - have at least 6 months of history
# - have activity in >= 3 of last 6 months
# - have positive previous and recent revenue
# - NOT already have a >=30% visible drop
# ============================================================

rebuilt_eligible_core = (
    core_features

    .filter(
        (F.col("customer_age_months") >= 6)
        &
        (F.col("active_months_6m") >= 3)
        &
        (F.col("previous_3m_revenue") > 0)
        &
        (F.col("recent_3m_revenue_feature") > 0)
        &
        (F.col("current_drop_pct") < CHURN_DROP_THRESHOLD)
    )
)


rebuilt_eligible_count = (
    rebuilt_eligible_core.count()
)


print()
print("=" * 90)
print("ELIGIBILITY RESULT")
print("=" * 90)

print(
    f"Customers before eligibility: "
    f"{core_features.count():,}"
)

print(
    f"Customers eligible:           "
    f"{rebuilt_eligible_count:,}"
)

print(
    f"Validated existing customers: "
    f"{current_scoring_features.count():,}"
)


# ============================================================
# 9. MEMBERSHIP COMPARISON
# ============================================================

existing_ids = (
    current_scoring_features
    .select("customer_id")
)


rebuilt_ids = (
    rebuilt_eligible_core
    .select("customer_id")
)


missing_customers = (
    existing_ids
    .join(
        rebuilt_ids,
        on="customer_id",
        how="left_anti",
    )
)


extra_customers = (
    rebuilt_ids
    .join(
        existing_ids,
        on="customer_id",
        how="left_anti",
    )
)


missing_customer_count = (
    missing_customers.count()
)

extra_customer_count = (
    extra_customers.count()
)


# ============================================================
# 10. CORE FEATURE COMPARISON
# ============================================================

CORE_COMPARE_COLUMNS = [
    "current_drop_pct",
    "revenue_1m",
    "revenue_3m",
    "revenue_6m",
    "revenue_12m",
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
    "customer_age_months",
]


core_comparison = (
    current_scoring_features
    .select(
        "customer_id",
        *CORE_COMPARE_COLUMNS,
    )
    .alias("e")

    .join(
        rebuilt_eligible_core
        .select(
            "customer_id",
            *CORE_COMPARE_COLUMNS,
        )
        .alias("r"),

        on=(
            F.col("e.customer_id")
            ==
            F.col("r.customer_id")
        ),

        how="inner",
    )
)


TOLERANCE = 1e-6


mismatch_expressions = []


for feature in CORE_COMPARE_COLUMNS:

    mismatch_expressions.append(

        F.sum(
            F.when(
                F.abs(
                    F.col(f"e.{feature}")
                    -
                    F.col(f"r.{feature}")
                )
                >
                F.lit(TOLERANCE),

                1,
            )
            .otherwise(0)
        ).alias(feature)
    )


core_mismatch_summary = (
    core_comparison
    .agg(
        *mismatch_expressions
    )
)


# ============================================================
# 11. FINAL VALIDATION
# ============================================================

print()
print("=" * 90)
print("CURRENT SCORING MEMBERSHIP VALIDATION")
print("=" * 90)

print(
    f"Missing customers: "
    f"{missing_customer_count:,}"
)

print(
    f"Extra customers:   "
    f"{extra_customer_count:,}"
)


print()
print("=" * 90)
print("CORE FEATURE MISMATCH COUNTS")
print("=" * 90)

display(
    core_mismatch_summary
)


# ============================================================
# 12. CURRENT DROP CHECK
# ============================================================

print()
print("=" * 90)
print("REBUILT CURRENT DROP DISTRIBUTION")
print("=" * 90)

display(
    rebuilt_eligible_core

    .agg(
        F.min(
            "current_drop_pct"
        ).alias(
            "min"
        ),

        F.avg(
            "current_drop_pct"
        ).alias(
            "avg"
        ),

        F.max(
            "current_drop_pct"
        ).alias(
            "max"
        ),
    )
)


# ============================================================
# 13. SHOW MEMBERSHIP DIFFERENCES IF ANY
# ============================================================

if missing_customer_count > 0:

    print()
    print("MISSING CUSTOMERS:")

    display(
        missing_customers
        .limit(20)
    )


if extra_customer_count > 0:

    print()
    print("EXTRA CUSTOMERS:")

    display(
        extra_customers
        .limit(20)
    )


print()
print(
    "READ-ONLY CURRENT SCORING CORE VALIDATION COMPLETED."
)

# COMMAND ----------

# ============================================================
# CELL 19 - REBUILD + VALIDATE ALL 43 CURRENT MODEL FEATURES
#
# READ ONLY
#
# PURPOSE:
# - Reconstruct all 43 model features for the 99 eligible
#   customers.
# - Compare every feature against the already validated
#   current_scoring_features Delta table.
#
# NOTHING IS WRITTEN.
# ============================================================

from pyspark.sql import functions as F


# ============================================================
# 1. HISTORY FOR THE 99 ELIGIBLE CUSTOMERS
# ============================================================

eligible_ids = (
    rebuilt_eligible_core
    .select("customer_id")
    .distinct()
)


feature_history = (
    scoring_history

    .join(
        eligible_ids,
        on="customer_id",
        how="inner",
    )

    .select(
        "customer_id",
        "month_start",
        "month_offset",
        F.col("net_revenue").cast("double").alias("net_revenue"),
        F.col("purchase_count").cast("double").alias("purchase_count"),
        F.col("credit_note_count").cast("double").alias("credit_note_count"),
        F.col("invoice_count").cast("double").alias("invoice_count"),
        F.col("is_active_month").cast("double").alias("is_active_month"),
    )
)


# ============================================================
# 2. WINDOW STATISTICS
# ============================================================

feature_stats = (
    feature_history

    .groupBy("customer_id")

    .agg(

        # ----------------------------------------------------
        # MONTHLY REVENUE AVERAGES
        # ----------------------------------------------------

        F.avg(
            F.when(
                F.col("month_offset").between(0, 2),
                F.col("net_revenue"),
            )
        ).alias("avg_monthly_revenue_3m"),

        F.avg(
            F.when(
                F.col("month_offset").between(0, 5),
                F.col("net_revenue"),
            )
        ).alias("avg_monthly_revenue_6m"),

        F.avg(
            F.when(
                F.col("month_offset").between(0, 11),
                F.col("net_revenue"),
            )
        ).alias("avg_monthly_revenue_12m"),


        # ----------------------------------------------------
        # REVENUE VOLATILITY
        # ----------------------------------------------------

        F.stddev_samp(
            F.when(
                F.col("month_offset").between(0, 5),
                F.col("net_revenue"),
            )
        ).alias("revenue_std_6m"),

        F.stddev_samp(
            F.when(
                F.col("month_offset").between(0, 11),
                F.col("net_revenue"),
            )
        ).alias("revenue_std_12m"),


        # ----------------------------------------------------
        # CREDIT NOTES
        # ----------------------------------------------------

        F.sum(
            F.when(
                F.col("month_offset").between(0, 2),
                F.col("credit_note_count"),
            ).otherwise(0.0)
        ).alias("credit_notes_3m"),

        F.sum(
            F.when(
                F.col("month_offset").between(0, 5),
                F.col("credit_note_count"),
            ).otherwise(0.0)
        ).alias("credit_notes_6m"),


        # Used only for credit-note rate.
        F.sum(
            F.when(
                F.col("month_offset").between(0, 5),
                F.col("invoice_count"),
            ).otherwise(0.0)
        ).alias("invoice_count_6m"),


        # ----------------------------------------------------
        # TREND CALCULATION COMPONENTS
        #
        # For 6 months:
        # oldest month -> x=0
        # newest month -> x=5
        # ----------------------------------------------------

        F.sum(
            F.when(
                F.col("month_offset").between(0, 5),

                (
                    F.lit(5.0)
                    -
                    F.col("month_offset").cast("double")
                )
                *
                F.col("net_revenue")
            )
            .otherwise(0.0)
        ).alias("revenue_xy_6m"),

        F.sum(
            F.when(
                F.col("month_offset").between(0, 5),
                F.col("net_revenue"),
            )
            .otherwise(0.0)
        ).alias("revenue_y_6m"),

        F.sum(
            F.when(
                F.col("month_offset").between(0, 5),

                (
                    F.lit(5.0)
                    -
                    F.col("month_offset").cast("double")
                )
                *
                F.col("purchase_count")
            )
            .otherwise(0.0)
        ).alias("purchase_xy_6m"),

        F.sum(
            F.when(
                F.col("month_offset").between(0, 5),
                F.col("purchase_count"),
            )
            .otherwise(0.0)
        ).alias("purchase_y_6m"),


        # ----------------------------------------------------
        # CURRENT INACTIVITY STREAK
        #
        # Most recent active month:
        # offset 0 -> streak 0
        # offset 1 -> streak 1
        # offset 2 -> streak 2
        # ----------------------------------------------------

        F.min(
            F.when(
                (
                    F.col("month_offset").between(0, 11)
                    &
                    (F.col("is_active_month") > 0)
                ),
                F.col("month_offset").cast("double"),
            )
        ).alias("inactivity_streak_months"),
    )
)


# ============================================================
# 3. JOIN CORE FEATURES
# ============================================================

rebuilt_43 = (
    rebuilt_eligible_core

    .join(
        feature_stats,
        on="customer_id",
        how="left",
    )
)


# ============================================================
# 4. V1 DERIVED FEATURES
# ============================================================

rebuilt_43 = (

    rebuilt_43

    # --------------------------------------------------------
    # Revenue CV
    # --------------------------------------------------------

    .withColumn(
        "revenue_cv_6m",

        F.when(
            F.abs(F.col("avg_monthly_revenue_6m")) > 1e-12,

            F.col("revenue_std_6m")
            /
            F.abs(F.col("avg_monthly_revenue_6m"))
        )
        .otherwise(0.0)
    )


    # --------------------------------------------------------
    # Average ticket
    # --------------------------------------------------------

    .withColumn(
        "avg_ticket_3m",

        F.when(
            F.col("purchases_3m") > 0,

            F.col("revenue_3m")
            /
            F.col("purchases_3m")
        )
        .otherwise(0.0)
    )

    .withColumn(
        "avg_ticket_6m",

        F.when(
            F.col("purchases_6m") > 0,

            F.col("revenue_6m")
            /
            F.col("purchases_6m")
        )
        .otherwise(0.0)
    )
)


# ============================================================
# 5. EXPECTED PURCHASE CADENCE
#
# IMPORTANT:
# Cadence is based on ACTIVE MONTHS, not invoice count.
# ============================================================

rebuilt_43 = (

    rebuilt_43

    .withColumn(
        "expected_days_between_purchases_12m",

        F.when(
            F.col("active_months_12m") > 0,

            F.lit(365.25)
            /
            F.col("active_months_12m")
        )
        .otherwise(365.25)
    )

    .withColumn(
        "_expected_days_between_purchases_6m",

        F.when(
            F.col("active_months_6m") > 0,

            F.lit(182.625)
            /
            F.col("active_months_6m")
        )
        .otherwise(182.625)
    )
)


# ============================================================
# 6. RECENCY RATIOS
# ============================================================

rebuilt_43 = (

    rebuilt_43

    .withColumn(
        "recency_ratio_12m",

        F.col("recency_days")
        /
        F.col("expected_days_between_purchases_12m")
    )

    .withColumn(
        "recency_ratio_6m",

        F.col("recency_days")
        /
        F.col("_expected_days_between_purchases_6m")
    )
)


# ============================================================
# 7. ACTIVE RATES
# ============================================================

rebuilt_43 = (

    rebuilt_43

    .withColumn(
        "active_rate_3m",
        F.col("active_months_3m") / F.lit(3.0),
    )

    .withColumn(
        "active_rate_6m",
        F.col("active_months_6m") / F.lit(6.0),
    )

    .withColumn(
        "active_rate_12m",
        F.col("active_months_12m") / F.lit(12.0),
    )
)


# ============================================================
# 8. ACTIVITY-ADJUSTED VALUE / PURCHASE FREQUENCY
# ============================================================

rebuilt_43 = (

    rebuilt_43

    .withColumn(
        "purchases_per_active_month_12m",

        F.when(
            F.col("active_months_12m") > 0,

            F.col("purchases_12m")
            /
            F.col("active_months_12m")
        )
        .otherwise(0.0)
    )

    .withColumn(
        "revenue_per_active_month_12m",

        F.when(
            F.col("active_months_12m") > 0,

            F.col("revenue_12m")
            /
            F.col("active_months_12m")
        )
        .otherwise(0.0)
    )
)


# ============================================================
# 9. REVENUE MOMENTUM
# ============================================================

rebuilt_43 = (

    rebuilt_43

    .withColumn(
        "revenue_momentum_1m_vs_6m",

        F.when(
            F.abs(F.col("avg_monthly_revenue_6m")) > 1e-12,

            (
                F.col("revenue_1m")
                /
                F.col("avg_monthly_revenue_6m")
            )
            -
            F.lit(1.0)
        )
        .otherwise(0.0)
    )

    .withColumn(
        "revenue_momentum_3m_vs_12m",

        F.when(
            F.abs(F.col("avg_monthly_revenue_12m")) > 1e-12,

            (
                F.col("avg_monthly_revenue_3m")
                /
                F.col("avg_monthly_revenue_12m")
            )
            -
            F.lit(1.0)
        )
        .otherwise(0.0)
    )
)


# ============================================================
# 10. PURCHASE MOMENTUM
# ============================================================

rebuilt_43 = (
    rebuilt_43

    .withColumn(
        "purchase_momentum_3m_vs_12m",

        F.when(
            F.col("purchases_12m") > 0,

            (
                (
                    F.col("purchases_3m")
                    /
                    F.lit(3.0)
                )
                /
                (
                    F.col("purchases_12m")
                    /
                    F.lit(12.0)
                )
            )
            -
            F.lit(1.0)
        )
        .otherwise(0.0)
    )
)


# ============================================================
# 11. AVG TICKET CHANGE
# ============================================================

rebuilt_43 = (
    rebuilt_43

    .withColumn(
        "avg_ticket_change_3m_vs_6m",

        F.when(
            F.abs(F.col("avg_ticket_6m")) > 1e-12,

            (
                F.col("avg_ticket_3m")
                /
                F.col("avg_ticket_6m")
            )
            -
            F.lit(1.0)
        )
        .otherwise(0.0)
    )
)


# ============================================================
# 12. CREDIT NOTE RATE
# ============================================================

rebuilt_43 = (
    rebuilt_43

    .withColumn(
        "credit_note_rate_6m",

        F.when(
            F.col("invoice_count_6m") > 0,

            F.col("credit_notes_6m")
            /
            F.col("invoice_count_6m")
        )
        .otherwise(0.0)
    )
)


# ============================================================
# 13. SIX-MONTH LINEAR TRENDS
#
# x = [0, 1, 2, 3, 4, 5]
#
# slope =
#   (n*sum(xy) - sum(x)*sum(y))
#   /
#   (n*sum(x²) - sum(x)²)
#
# n = 6
# sum(x) = 15
# sum(x²) = 55
# denominator = 105
# ============================================================

rebuilt_43 = (

    rebuilt_43

    .withColumn(
        "revenue_slope_6m",

        (
            (
                F.lit(6.0)
                *
                F.col("revenue_xy_6m")
            )
            -
            (
                F.lit(15.0)
                *
                F.col("revenue_y_6m")
            )
        )
        /
        F.lit(105.0)
    )

    .withColumn(
        "purchase_slope_6m",

        (
            (
                F.lit(6.0)
                *
                F.col("purchase_xy_6m")
            )
            -
            (
                F.lit(15.0)
                *
                F.col("purchase_y_6m")
            )
        )
        /
        F.lit(105.0)
    )
)


# ============================================================
# 14. NORMALIZED TRENDS
# ============================================================

rebuilt_43 = (

    rebuilt_43

    .withColumn(
        "revenue_trend_6m_normalized",

        F.when(
            F.abs(F.col("avg_monthly_revenue_6m")) > 1e-12,

            F.col("revenue_slope_6m")
            /
            F.abs(F.col("avg_monthly_revenue_6m"))
        )
        .otherwise(0.0)
    )

    .withColumn(
        "purchase_trend_6m_normalized",

        F.when(
            F.col("purchases_6m") > 0,

            F.col("purchase_slope_6m")
            /
            (
                F.col("purchases_6m")
                /
                F.lit(6.0)
            )
        )
        .otherwise(0.0)
    )
)


# ============================================================
# 15. CLEAN TEMPORARY VALUES
# ============================================================

rebuilt_43 = (
    rebuilt_43

    .fillna(
        0.0,
        subset=FEATURE_COLUMNS,
    )

    .select(
        "customer_id",
        "month_start",
        "current_drop_pct",
        *[
            F.col(feature)
            .cast("double")
            .alias(feature)
            for feature in FEATURE_COLUMNS
        ],
    )
)


# ============================================================
# 16. BASIC VALIDATION
# ============================================================

print("=" * 90)
print("43-FEATURE REBUILD")
print("=" * 90)

print(
    f"Rows rebuilt:       "
    f"{rebuilt_43.count():,}"
)

print(
    f"Model features:     "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    f"Total columns:      "
    f"{len(rebuilt_43.columns)}"
)


# ============================================================
# 17. NULL CHECK
# ============================================================

null_summary = (
    rebuilt_43
    .agg(
        *[
            F.sum(
                F.when(
                    F.col(feature).isNull(),
                    1,
                )
                .otherwise(0)
            ).alias(feature)
            for feature in FEATURE_COLUMNS
        ]
    )
    .first()
)


total_nulls = sum(
    int(null_summary[feature] or 0)
    for feature in FEATURE_COLUMNS
)


print(
    f"Nulls in features:  "
    f"{total_nulls:,}"
)


# ============================================================
# 18. COMPARE ALL 43 FEATURES AGAINST VALIDATED DATASET
# ============================================================

existing_43 = (
    current_scoring_features
    .select(
        "customer_id",
        *[
            F.col(feature)
            .cast("double")
            .alias(feature)
            for feature in FEATURE_COLUMNS
        ],
    )
    .alias("e")
)


candidate_43 = (
    rebuilt_43
    .select(
        "customer_id",
        *FEATURE_COLUMNS,
    )
    .alias("r")
)


feature_comparison_43 = (
    existing_43

    .join(
        candidate_43,

        on=(
            F.col("e.customer_id")
            ==
            F.col("r.customer_id")
        ),

        how="inner",
    )
)


TOLERANCE_43 = 1e-6


mismatch_aggregations = []


for feature in FEATURE_COLUMNS:

    e_col = F.col(f"e.{feature}")
    r_col = F.col(f"r.{feature}")

    mismatch = (
        F.when(
            e_col.isNull() & r_col.isNull(),
            F.lit(False),
        )
        .when(
            e_col.isNull() | r_col.isNull(),
            F.lit(True),
        )
        .otherwise(
            F.abs(e_col - r_col)
            >
            F.lit(TOLERANCE_43)
        )
    )

    mismatch_aggregations.append(
        F.sum(
            F.when(
                mismatch,
                1,
            )
            .otherwise(0)
        ).alias(feature)
    )


mismatch_row_43 = (
    feature_comparison_43
    .agg(
        *mismatch_aggregations
    )
    .first()
)


mismatch_dict_43 = {
    feature: int(
        mismatch_row_43[feature] or 0
    )
    for feature in FEATURE_COLUMNS
}


nonzero_feature_mismatches = {
    feature: count
    for feature, count
    in mismatch_dict_43.items()
    if count != 0
}


total_feature_mismatches = sum(
    mismatch_dict_43.values()
)


# ============================================================
# 19. FINAL RESULT
# ============================================================

print()
print("=" * 90)
print("43-FEATURE VALIDATION")
print("=" * 90)

print(
    f"Customers compared:       "
    f"{feature_comparison_43.count():,}"
)

print(
    f"Total feature mismatches: "
    f"{total_feature_mismatches:,}"
)

print(
    f"Features with mismatches: "
    f"{len(nonzero_feature_mismatches):,}"
)


if nonzero_feature_mismatches:

    print()
    print("NON-ZERO MISMATCHES:")

    for feature, count in nonzero_feature_mismatches.items():
        print(
            f" - {feature}: {count:,}"
        )

else:

    print()
    print(
        "ALL 43 FEATURES MATCH THE VALIDATED "
        "CURRENT SCORING DATASET."
    )


# ============================================================
# 20. KNOWN-CUSTOMER SANITY CHECK
# ============================================================

print()
print("=" * 90)
print("KNOWN CUSTOMER CHECK - 000647")
print("=" * 90)


display(
    rebuilt_43

    .filter(
        F.col("customer_id") == "000647"
    )

    .select(
        "customer_id",
        "current_drop_pct",
        "revenue_12m",
        "revenue_3m",
        "purchases_12m",
        "active_months_12m",
        "recency_days",
        "recency_ratio_12m",
        "revenue_change_3m",
        "revenue_trend_6m_normalized",
    )
)


print()
print(
    "READ-ONLY 43-FEATURE VALIDATION COMPLETED."
)

# COMMAND ----------

# ============================================================
# CELL 20 - CURRENT FEATURE CONTRACT
#
# Fail fast before loading the model or modifying production.
# ============================================================

current_feature_snapshot = (
    rebuilt_43
    .select(
        "customer_id",
        "month_start",
        "current_drop_pct",
        *FEATURE_COLUMNS,
    )
)

feature_row_count = current_feature_snapshot.count()
feature_customer_count = (
    current_feature_snapshot
    .select("customer_id")
    .distinct()
    .count()
)

feature_snapshot_profile = (
    current_feature_snapshot
    .agg(
        F.countDistinct("month_start").alias("snapshot_count"),
        F.min("month_start").alias("minimum_snapshot"),
        F.max("month_start").alias("maximum_snapshot"),
    )
    .first()
)

feature_null_count = (
    current_feature_snapshot
    .agg(
        *[
            F.sum(
                F.when(F.col(column_name).isNull(), 1).otherwise(0)
            ).alias(column_name)
            for column_name in FEATURE_COLUMNS
        ]
    )
    .first()
)

total_feature_nulls = sum(
    int(feature_null_count[column_name] or 0)
    for column_name in FEATURE_COLUMNS
)

if feature_row_count == 0:
    raise ValueError("The current feature snapshot is empty.")

if feature_customer_count != feature_row_count:
    raise ValueError(
        "The current feature snapshot contains duplicate customers."
    )

if feature_snapshot_profile["snapshot_count"] != 1:
    raise ValueError(
        "The current feature snapshot contains more than one month."
    )

if feature_snapshot_profile["minimum_snapshot"] != snapshot_month:
    raise ValueError(
        "The reconstructed snapshot does not match the requested snapshot."
    )

if total_feature_nulls != 0:
    raise ValueError(
        f"The 43 model features contain {total_feature_nulls} null values."
    )

print()
print("=" * 90)
print("CURRENT FEATURE CONTRACT")
print("=" * 90)
print(f"Rows:               {feature_row_count:,}")
print(f"Distinct customers: {feature_customer_count:,}")
print(f"Snapshot:           {snapshot_month}")
print(f"Model features:     {len(FEATURE_COLUMNS)}")
print(f"Nulls in features:  {total_feature_nulls:,}")
print("CURRENT FEATURE CONTRACT: PASSED")

# COMMAND ----------

# ============================================================
# CELL 21 - LOAD THE CHAMPION MODEL FROM UNITY CATALOG
#
# Serverless-safe SparkML loading through a UC Volume.
# ============================================================

from mlflow import MlflowClient
from pyspark.ml.feature import VectorAssembler

os.environ["MLFLOW_DFS_TMP"] = MLFLOW_DFS_TMP
dbutils.fs.mkdirs(MLFLOW_DFS_TMP)
mlflow.set_registry_uri("databricks-uc")

registry_client = MlflowClient()
resolved_model_version = registry_client.get_model_version_by_alias(
    name=REGISTERED_MODEL_NAME,
    alias=REGISTERED_MODEL_ALIAS,
)

RESOLVED_MODEL_VERSION = str(resolved_model_version.version)
RESOLVED_MODEL_RUN_ID = str(resolved_model_version.run_id)

registered_spark_model = mlflow.spark.load_model(
    REGISTERED_MODEL_URI,
    dfs_tmpdir=MLFLOW_DFS_TMP,
)

model_stages = list(registered_spark_model.stages)
assembler_stages = [
    stage
    for stage in model_stages
    if isinstance(stage, VectorAssembler)
]

if len(assembler_stages) != 1:
    raise ValueError(
        "The registered pipeline must contain exactly one VectorAssembler."
    )

registered_assembler = assembler_stages[0]
assembler_inputs = list(registered_assembler.getInputCols())

classifier_stages = [
    stage
    for stage in model_stages
    if stage.__class__.__name__ == "RandomForestClassificationModel"
]

if len(classifier_stages) != 1:
    raise ValueError(
        "The registered pipeline must contain exactly one "
        "RandomForestClassificationModel."
    )

registered_classifier = classifier_stages[0]

if assembler_inputs != FEATURE_COLUMNS:
    raise ValueError(
        "The registered model feature order does not match FEATURE_COLUMNS."
    )

if int(registered_classifier.numFeatures) != len(FEATURE_COLUMNS):
    raise ValueError(
        "The registered classifier was not trained with 43 features."
    )

if int(registered_classifier.numClasses) != 2:
    raise ValueError(
        "The registered classifier is not a binary classifier."
    )

print()
print("=" * 90)
print("REGISTERED MODEL CONTRACT")
print("=" * 90)
print(f"Registry URI:       {mlflow.get_registry_uri()}")
print(f"Model name:         {REGISTERED_MODEL_NAME}")
print(f"Model alias:        {REGISTERED_MODEL_ALIAS}")
print(f"Resolved version:   {RESOLVED_MODEL_VERSION}")
print(f"Resolved run id:    {RESOLVED_MODEL_RUN_ID}")
print(f"Pipeline stages:    {len(model_stages)}")
print(f"Assembler inputs:   {len(assembler_inputs)}")
print(f"Model numFeatures:  {registered_classifier.numFeatures}")
print(f"Model numClasses:   {registered_classifier.numClasses}")
print("REGISTERED MODEL CONTRACT: PASSED")

# COMMAND ----------

# ============================================================
# CELL 22 - CURRENT CUSTOMER SCORING
# ============================================================

from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import ArrayType

raw_current_predictions = registered_spark_model.transform(
    current_feature_snapshot
)

probability_data_type = (
    raw_current_predictions.schema["probability"].dataType
)

if isinstance(probability_data_type, VectorUDT):
    churn_probability_expression = vector_to_array(
        F.col("probability")
    )[1]
elif isinstance(probability_data_type, ArrayType):
    churn_probability_expression = F.col("probability")[1]
else:
    raise TypeError(
        "Unsupported probability column type: "
        f"{probability_data_type.simpleString()}"
    )

risk_window = Window.orderBy(
    F.col("churn_probability").desc(),
    F.col("customer_id").asc(),
)

current_customer_scores = (
    raw_current_predictions
    .withColumn(
        "churn_probability",
        churn_probability_expression.cast("double"),
    )
    .withColumn(
        "churn_prediction",
        F.col("prediction").cast("int"),
    )
    .withColumn(
        "scoring_snapshot",
        F.lit(snapshot_month).cast("date"),
    )
    .withColumn(
        "risk_rank",
        F.row_number().over(risk_window),
    )
    .withColumn(
        "risk_percentile",
        (F.lit(1.0) - F.percent_rank().over(risk_window)).cast("double"),
    )
    .withColumn("model_name", F.lit(REGISTERED_MODEL_NAME))
    .withColumn("model_version", F.lit(RESOLVED_MODEL_VERSION))
    .withColumn("model_run_id", F.lit(RESOLVED_MODEL_RUN_ID))
    .withColumn("scored_at", F.current_timestamp())
    .select(
        "customer_id",
        "scoring_snapshot",
        "churn_prediction",
        "churn_probability",
        "current_drop_pct",
        "revenue_12m",
        "revenue_6m",
        "revenue_3m",
        "purchases_12m",
        "active_months_12m",
        "recency_days",
        "risk_rank",
        "risk_percentile",
        "model_name",
        "model_version",
        "model_run_id",
        "scored_at",
    )
)

score_validation = (
    current_customer_scores
    .agg(
        F.count("*").alias("rows"),
        F.countDistinct("customer_id").alias("customers"),
        F.countDistinct("scoring_snapshot").alias("snapshots"),
        F.sum(
            F.when(
                F.col("churn_probability").isNull()
                | (F.col("churn_probability") < 0.0)
                | (F.col("churn_probability") > 1.0),
                1,
            ).otherwise(0)
        ).alias("invalid_probabilities"),
        F.sum(
            F.when(
                F.col("churn_prediction").isNull()
                | ~F.col("churn_prediction").isin(0, 1),
                1,
            ).otherwise(0)
        ).alias("invalid_predictions"),
        F.min("risk_rank").alias("minimum_rank"),
        F.max("risk_rank").alias("maximum_rank"),
    )
    .first()
)

score_row_count = int(score_validation["rows"])

if int(score_validation["customers"]) != score_row_count:
    raise ValueError("Scoring produced duplicate customer rows.")

if int(score_validation["snapshots"]) != 1:
    raise ValueError("Scoring produced more than one snapshot.")

if int(score_validation["invalid_probabilities"] or 0) != 0:
    raise ValueError("Scoring produced invalid churn probabilities.")

if int(score_validation["invalid_predictions"] or 0) != 0:
    raise ValueError("Scoring produced invalid churn predictions.")

if int(score_validation["minimum_rank"]) != 1:
    raise ValueError("Risk ranking does not start at 1.")

if int(score_validation["maximum_rank"]) != score_row_count:
    raise ValueError("Risk ranking is not complete.")

top_25_current_churn_risk = (
    current_customer_scores
    .orderBy("risk_rank")
    .limit(TOP25_SIZE)
)

top_30_current_churn_risk = (
    current_customer_scores
    .orderBy("risk_rank")
    .limit(TOP30_SIZE)
)

print()
print("=" * 90)
print("CURRENT SCORING RESULT")
print("=" * 90)
print(f"Rows:                  {score_row_count:,}")
print(f"Distinct customers:    {int(score_validation['customers']):,}")
print(f"Snapshot:              {snapshot_month}")
print(f"Invalid probabilities: {int(score_validation['invalid_probabilities'] or 0):,}")
print(f"Invalid predictions:   {int(score_validation['invalid_predictions'] or 0):,}")
print("CURRENT SCORING: PASSED")

# COMMAND ----------

# ============================================================
# CELL 23 - IDEMPOTENT PRODUCTION PUBLICATION
#
# No cache(), persist() or PERSIST TABLE: safe for serverless.
# ============================================================

if PUBLISH_OUTPUTS:
    (
        current_customer_scores
        .write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(TABLE_CURRENT_CUSTOMER_SCORES)
    )

    spark.sql(
        f"""
        CREATE OR REPLACE VIEW {TABLE_TOP25_CHURN_RISK} AS
        SELECT *
        FROM {TABLE_CURRENT_CUSTOMER_SCORES}
        ORDER BY risk_rank
        LIMIT {TOP25_SIZE}
        """
    )

    spark.sql(
        f"""
        CREATE OR REPLACE VIEW {TABLE_TOP30_CHURN_RISK} AS
        SELECT *
        FROM {TABLE_CURRENT_CUSTOMER_SCORES}
        ORDER BY risk_rank
        LIMIT {TOP30_SIZE}
        """
    )

    persisted_validation = (
        spark.table(TABLE_CURRENT_CUSTOMER_SCORES)
        .agg(
            F.count("*").alias("rows"),
            F.countDistinct("customer_id").alias("customers"),
            F.countDistinct("scoring_snapshot").alias("snapshots"),
            F.sum(
                F.when(
                    F.col("churn_probability").isNull()
                    | (F.col("churn_probability") < 0.0)
                    | (F.col("churn_probability") > 1.0),
                    1,
                ).otherwise(0)
            ).alias("invalid_probabilities"),
        )
        .first()
    )

    persisted_top_25_rows = spark.table(
        TABLE_TOP25_CHURN_RISK
    ).count()

    persisted_top_30_rows = spark.table(
        TABLE_TOP30_CHURN_RISK
    ).count()

    expected_top_25_rows = min(TOP25_SIZE, score_row_count)
    expected_top_30_rows = min(TOP30_SIZE, score_row_count)

    if int(persisted_validation["rows"]) != score_row_count:
        raise ValueError("Persisted score row count does not match scoring.")

    if int(persisted_validation["customers"]) != score_row_count:
        raise ValueError("Persisted scores contain duplicate customers.")

    if int(persisted_validation["snapshots"]) != 1:
        raise ValueError("Persisted scores contain more than one snapshot.")

    if int(persisted_validation["invalid_probabilities"] or 0) != 0:
        raise ValueError("Persisted scores contain invalid probabilities.")

    if persisted_top_25_rows != expected_top_25_rows:
        raise ValueError("The persisted Top 25 view has an invalid row count.")

    if persisted_top_30_rows != expected_top_30_rows:
        raise ValueError("The persisted Top 30 view has an invalid row count.")

    publication_status = "PUBLISHED"
else:
    persisted_top_25_rows = min(TOP25_SIZE, score_row_count)
    persisted_top_30_rows = min(TOP30_SIZE, score_row_count)
    publication_status = "VALIDATED_ONLY"

print()
print("=" * 90)
print("PRODUCTION PUBLICATION RESULT")
print("=" * 90)
print(f"Status:          {publication_status}")
print(f"Score table:     {TABLE_CURRENT_CUSTOMER_SCORES}")
print(f"Top 25 view:     {TABLE_TOP25_CHURN_RISK}")
print(f"Top 30 view:     {TABLE_TOP30_CHURN_RISK}")
print(f"Published rows:  {score_row_count:,}")
print("PRODUCTION PUBLICATION: PASSED")

# COMMAND ----------

# ============================================================
# CELL 24 - JOB OUTPUT
# ============================================================

job_result = {
    "status": "SUCCESS",
    "project": PROJECT_NAME,
    "pipeline_run_id": PIPELINE_RUN_ID,
    "latest_invoice_date": str(latest_invoice_date_for_scoring),
    "scoring_snapshot": str(snapshot_month),
    "snapshot_source": snapshot_source,
    "customers_scored": score_row_count,
    "top_25_rows": int(persisted_top_25_rows),
    "top_30_rows": int(persisted_top_30_rows),
    "model_name": REGISTERED_MODEL_NAME,
    "model_alias": REGISTERED_MODEL_ALIAS,
    "model_version": RESOLVED_MODEL_VERSION,
    "model_run_id": RESOLVED_MODEL_RUN_ID,
    "published": PUBLISH_OUTPUTS,
    "score_table": TABLE_CURRENT_CUSTOMER_SCORES,
    "top_25_view": TABLE_TOP25_CHURN_RISK,
    "top_30_view": TABLE_TOP30_CHURN_RISK,
}

job_result_json = json.dumps(
    job_result,
    ensure_ascii=False,
    sort_keys=True,
)

print()
print("=" * 90)
print("DATABRICKS JOB RESULT")
print("=" * 90)
print(job_result_json)

dbutils.notebook.exit(job_result_json)
