from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


# =========================================================
# 0. SPARK
# =========================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-build-current-scoring-features")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. CONFIGURACIÓN
#
# Julio 2026 = último mes completo.
# Agosto 2026 está incompleto y no lo usamos como snapshot.
# =========================================================

CUSTOMER_MONTHLY_PATH = "data/gold/customer_monthly"

OUTPUT_PATH = (
    "data/gold/current_scoring_features"
)

SNAPSHOT_MONTH = "2026-07-01"


# =========================================================
# 2. CARGAR CUSTOMER MONTHLY
# =========================================================

monthly_raw = spark.read.parquet(
    CUSTOMER_MONTHLY_PATH
)


print("\n" + "=" * 70)
print("BUILD CURRENT SCORING FEATURES")
print("=" * 70)

print(
    f"\nSnapshot operativo: "
    f"{SNAPSHOT_MONTH}"
)


# =========================================================
# 3. RESOLVER NOMBRES DE COLUMNAS
# =========================================================

def resolve_column(
    df,
    candidates,
    description
):

    for candidate in candidates:

        if candidate in df.columns:

            print(
                f"{description}: "
                f"{candidate}"
            )

            return candidate

    raise ValueError(
        f"No encuentro columna para "
        f"{description}. "
        f"Probadas: {candidates}. "
        f"Disponibles: {df.columns}"
    )


revenue_col = resolve_column(
    monthly_raw,
    [
        "net_revenue",
        "monthly_net_revenue",
        "monthly_revenue",
        "revenue",
    ],
    "Monthly revenue"
)


purchase_col = resolve_column(
    monthly_raw,
    [
        "purchase_count",
        "monthly_purchase_count",
        "purchases",
    ],
    "Purchase count"
)


credit_note_col = resolve_column(
    monthly_raw,
    [
        "credit_note_count",
        "monthly_credit_note_count",
        "credit_notes",
    ],
    "Credit note count"
)


recency_col = resolve_column(
    monthly_raw,
    [
        "days_since_last_purchase",
        "recency_days",
    ],
    "Recency"
)


# =========================================================
# 4. PREPARAR MONTHLY
#
# MUY IMPORTANTE:
#
# Eliminamos cualquier fila posterior al snapshot.
#
# Por tanto agosto 2026 no puede influir en ninguna
# feature calculada para julio.
# =========================================================

monthly = (
    monthly_raw

    .select(
        F.col("customer_id"),

        F.col("month_start")
        .cast("date")
        .alias("month_start"),

        F.col(revenue_col)
        .cast("double")
        .alias("monthly_revenue"),

        F.col(purchase_col)
        .cast("double")
        .alias("monthly_purchases"),

        F.col(credit_note_col)
        .cast("double")
        .alias("monthly_credit_notes"),

        F.col(recency_col)
        .cast("double")
        .alias("monthly_recency_days"),
    )

    .filter(
        F.col("month_start")
        <=
        F.lit(SNAPSHOT_MONTH)
    )

    .fillna(
        {
            "monthly_revenue": 0.0,
            "monthly_purchases": 0.0,
            "monthly_credit_notes": 0.0,
        }
    )

    .withColumn(
        "monthly_active",

        F.when(
            F.col("monthly_purchases") > 0,
            1.0
        ).otherwise(0.0)
    )
)


# =========================================================
# 5. VALIDAR UNICIDAD CUSTOMER-MONTH
# =========================================================

duplicates = (
    monthly

    .groupBy(
        "customer_id",
        "month_start"
    )

    .count()

    .filter(
        F.col("count") > 1
    )

    .count()
)


if duplicates > 0:

    raise ValueError(
        f"Hay {duplicates} customer-month "
        f"duplicados."
    )


print(
    f"Customer-month duplicates: "
    f"{duplicates}"
)


# =========================================================
# 6. COMPROBAR QUE EXISTE JULIO 2026
# =========================================================

snapshot_rows = (
    monthly

    .filter(
        F.col("month_start")
        ==
        F.lit(SNAPSHOT_MONTH)
    )

    .count()
)


print(
    f"Clientes con fila en snapshot: "
    f"{snapshot_rows:,}"
)


if snapshot_rows == 0:

    raise ValueError(
        f"No existen filas para "
        f"{SNAPSHOT_MONTH}."
    )


# =========================================================
# 7. WINDOWS
# =========================================================

customer_order = (
    Window
    .partitionBy("customer_id")
    .orderBy("month_start")
)


history_window = (
    Window
    .partitionBy("customer_id")
    .orderBy("month_start")
    .rowsBetween(
        Window.unboundedPreceding,
        0
    )
)


window_3m = (
    Window
    .partitionBy("customer_id")
    .orderBy("month_start")
    .rowsBetween(
        -2,
        0
    )
)


window_6m = (
    Window
    .partitionBy("customer_id")
    .orderBy("month_start")
    .rowsBetween(
        -5,
        0
    )
)


window_12m = (
    Window
    .partitionBy("customer_id")
    .orderBy("month_start")
    .rowsBetween(
        -11,
        0
    )
)


# Baseline del target V3:
# t-5, t-4, t-3

window_previous_3m = (
    Window
    .partitionBy("customer_id")
    .orderBy("month_start")
    .rowsBetween(
        -5,
        -3
    )
)


# =========================================================
# 8. HISTORIA DEL CLIENTE
# =========================================================

monthly = monthly.withColumn(
    "history_months",
    F.row_number().over(
        customer_order
    )
)


monthly = monthly.withColumn(
    "first_customer_month",

    F.min(
        "month_start"
    ).over(
        Window.partitionBy(
            "customer_id"
        )
    )
)


# =========================================================
# 9. FEATURES BASE - REVENUE
# =========================================================

monthly = (
    monthly

    .withColumn(
        "revenue_1m",
        F.col("monthly_revenue")
    )

    .withColumn(
        "revenue_3m",
        F.sum(
            "monthly_revenue"
        ).over(window_3m)
    )

    .withColumn(
        "revenue_6m",
        F.sum(
            "monthly_revenue"
        ).over(window_6m)
    )

    .withColumn(
        "revenue_12m",
        F.sum(
            "monthly_revenue"
        ).over(window_12m)
    )
)


# =========================================================
# 10. AVG MONTHLY REVENUE
# =========================================================

monthly = (
    monthly

    .withColumn(
        "avg_monthly_revenue_3m",

        F.avg(
            "monthly_revenue"
        ).over(window_3m)
    )

    .withColumn(
        "avg_monthly_revenue_6m",

        F.avg(
            "monthly_revenue"
        ).over(window_6m)
    )

    .withColumn(
        "avg_monthly_revenue_12m",

        F.avg(
            "monthly_revenue"
        ).over(window_12m)
    )
)


# =========================================================
# 11. PURCHASE COUNTS
# =========================================================

monthly = (
    monthly

    .withColumn(
        "purchases_3m",

        F.sum(
            "monthly_purchases"
        ).over(window_3m)
    )

    .withColumn(
        "purchases_6m",

        F.sum(
            "monthly_purchases"
        ).over(window_6m)
    )

    .withColumn(
        "purchases_12m",

        F.sum(
            "monthly_purchases"
        ).over(window_12m)
    )
)


# =========================================================
# 12. ACTIVE MONTHS
# =========================================================

monthly = (
    monthly

    .withColumn(
        "active_months_3m",

        F.sum(
            "monthly_active"
        ).over(window_3m)
    )

    .withColumn(
        "active_months_6m",

        F.sum(
            "monthly_active"
        ).over(window_6m)
    )

    .withColumn(
        "active_months_12m",

        F.sum(
            "monthly_active"
        ).over(window_12m)
    )
)


# =========================================================
# 13. RECENCY
# =========================================================

monthly = monthly.withColumn(
    "recency_days",
    F.col(
        "monthly_recency_days"
    )
)


# =========================================================
# 14. PREVIOUS 3M VS RECENT 3M
#
# Esta es también la estructura usada por target V3:
#
# baseline:
# t-5, t-4, t-3
#
# recent:
# t-2, t-1, t
# =========================================================

monthly = monthly.withColumn(
    "previous_3m_revenue",

    F.sum(
        "monthly_revenue"
    ).over(
        window_previous_3m
    )
)


monthly = monthly.withColumn(
    "recent_3m_revenue_feature",

    F.sum(
        "monthly_revenue"
    ).over(
        window_3m
    )
)


monthly = monthly.withColumn(
    "revenue_change_3m",

    F.when(
        F.abs(
            F.col(
                "previous_3m_revenue"
            )
        ) > 1e-6,

        (
            F.col(
                "recent_3m_revenue_feature"
            )
            -
            F.col(
                "previous_3m_revenue"
            )
        )
        /
        F.abs(
            F.col(
                "previous_3m_revenue"
            )
        )

    ).otherwise(
        F.lit(0.0)
    )
)


# =========================================================
# 15. VOLATILIDAD
# =========================================================

monthly = (
    monthly

    .withColumn(
        "revenue_std_6m",

        F.stddev(
            "monthly_revenue"
        ).over(window_6m)
    )

    .withColumn(
        "revenue_std_12m",

        F.stddev(
            "monthly_revenue"
        ).over(window_12m)
    )
)


monthly = monthly.withColumn(
    "revenue_cv_6m",

    F.when(
        F.abs(
            F.col(
                "avg_monthly_revenue_6m"
            )
        ) > 1e-6,

        F.col(
            "revenue_std_6m"
        )
        /
        F.abs(
            F.col(
                "avg_monthly_revenue_6m"
            )
        )

    ).otherwise(
        F.lit(0.0)
    )
)


# =========================================================
# 16. AVG TICKET
# =========================================================

monthly = (
    monthly

    .withColumn(
        "avg_ticket_3m",

        F.when(
            F.col("purchases_3m") > 0,

            F.col("revenue_3m")
            /
            F.col("purchases_3m")

        ).otherwise(
            F.lit(0.0)
        )
    )

    .withColumn(
        "avg_ticket_6m",

        F.when(
            F.col("purchases_6m") > 0,

            F.col("revenue_6m")
            /
            F.col("purchases_6m")

        ).otherwise(
            F.lit(0.0)
        )
    )
)


# =========================================================
# 17. CREDIT NOTES
# =========================================================

monthly = (
    monthly

    .withColumn(
        "credit_notes_3m",

        F.sum(
            "monthly_credit_notes"
        ).over(window_3m)
    )

    .withColumn(
        "credit_notes_6m",

        F.sum(
            "monthly_credit_notes"
        ).over(window_6m)
    )
)


# =========================================================
# 18. CUSTOMER AGE
# =========================================================

monthly = monthly.withColumn(
    "customer_age_months",

    (
        F.floor(
            F.months_between(
                F.col("month_start"),
                F.col(
                    "first_customer_month"
                )
            )
        )
        +
        F.lit(1)
    ).cast("double")
)


# =========================================================
# 19. LAGS PARA FEATURES V2
# =========================================================

for lag_value in range(6):

    monthly = monthly.withColumn(
        f"revenue_lag_{lag_value}",

        F.lag(
            "monthly_revenue",
            lag_value
        ).over(customer_order)
    )

    monthly = monthly.withColumn(
        f"purchases_lag_{lag_value}",

        F.lag(
            "monthly_purchases",
            lag_value
        ).over(customer_order)
    )


# =========================================================
# 20. REVENUE SLOPE 6M
# =========================================================

monthly = monthly.withColumn(
    "revenue_slope_6m",

    (
        -2.5 * F.col("revenue_lag_5")
        -1.5 * F.col("revenue_lag_4")
        -0.5 * F.col("revenue_lag_3")
        +0.5 * F.col("revenue_lag_2")
        +1.5 * F.col("revenue_lag_1")
        +2.5 * F.col("revenue_lag_0")
    )
    /
    F.lit(17.5)
)


# =========================================================
# 21. PURCHASE SLOPE 6M
# =========================================================

monthly = monthly.withColumn(
    "purchase_slope_6m",

    (
        -2.5 * F.col("purchases_lag_5")
        -1.5 * F.col("purchases_lag_4")
        -0.5 * F.col("purchases_lag_3")
        +0.5 * F.col("purchases_lag_2")
        +1.5 * F.col("purchases_lag_1")
        +2.5 * F.col("purchases_lag_0")
    )
    /
    F.lit(17.5)
)


# =========================================================
# 22. INACTIVITY STREAK
# =========================================================

monthly = monthly.withColumn(
    "last_active_month",

    F.max(
        F.when(
            F.col("monthly_active") == 1,
            F.col("month_start")
        )
    ).over(history_window)
)


monthly = monthly.withColumn(
    "inactivity_streak_months",

    F.when(
        F.col("monthly_active") == 1,
        F.lit(0.0)
    )

    .when(
        F.col(
            "last_active_month"
        ).isNotNull(),

        F.months_between(
            F.col("month_start"),
            F.col(
                "last_active_month"
            )
        )
    )

    .otherwise(
        F.lit(0.0)
    )
)


# =========================================================
# 23. EXPECTED PURCHASE CADENCE
#
# Igual que V2:
# usamos meses activos, no número de facturas.
# =========================================================

monthly = monthly.withColumn(
    "expected_days_between_purchases_12m",

    F.when(
        F.col("active_months_12m") > 0,

        F.lit(365.25)
        /
        F.col(
            "active_months_12m"
        )

    ).otherwise(
        F.lit(365.25)
    )
)


monthly = monthly.withColumn(
    "expected_days_between_purchases_6m",

    F.when(
        F.col("active_months_6m") > 0,

        F.lit(182.625)
        /
        F.col(
            "active_months_6m"
        )

    ).otherwise(
        F.lit(182.625)
    )
)


# =========================================================
# 24. RECENCY RELATIVA
# =========================================================

monthly = monthly.withColumn(
    "recency_ratio_12m",

    F.col("recency_days")
    /
    F.col(
        "expected_days_between_purchases_12m"
    )
)


monthly = monthly.withColumn(
    "recency_ratio_6m",

    F.col("recency_days")
    /
    F.col(
        "expected_days_between_purchases_6m"
    )
)


# =========================================================
# 25. ACTIVE RATES
# =========================================================

monthly = (
    monthly

    .withColumn(
        "active_rate_3m",

        F.col("active_months_3m")
        /
        F.lit(3.0)
    )

    .withColumn(
        "active_rate_6m",

        F.col("active_months_6m")
        /
        F.lit(6.0)
    )

    .withColumn(
        "active_rate_12m",

        F.col("active_months_12m")
        /
        F.lit(12.0)
    )
)


# =========================================================
# 26. INTENSIDAD EN MESES ACTIVOS
# =========================================================

monthly = monthly.withColumn(
    "purchases_per_active_month_12m",

    F.when(
        F.col(
            "active_months_12m"
        ) > 0,

        F.col("purchases_12m")
        /
        F.col(
            "active_months_12m"
        )

    ).otherwise(
        F.lit(0.0)
    )
)


monthly = monthly.withColumn(
    "revenue_per_active_month_12m",

    F.when(
        F.col(
            "active_months_12m"
        ) > 0,

        F.col("revenue_12m")
        /
        F.col(
            "active_months_12m"
        )

    ).otherwise(
        F.lit(0.0)
    )
)


# =========================================================
# 27. REVENUE MOMENTUM
# =========================================================

monthly = monthly.withColumn(
    "revenue_momentum_1m_vs_6m",

    F.when(
        F.abs(
            F.col(
                "avg_monthly_revenue_6m"
            )
        ) > 1e-6,

        (
            F.col("revenue_1m")
            -
            F.col(
                "avg_monthly_revenue_6m"
            )
        )
        /
        F.abs(
            F.col(
                "avg_monthly_revenue_6m"
            )
        )

    ).otherwise(
        F.lit(0.0)
    )
)


monthly = monthly.withColumn(
    "revenue_momentum_3m_vs_12m",

    F.when(
        F.abs(
            F.col(
                "avg_monthly_revenue_12m"
            )
        ) > 1e-6,

        (
            F.col(
                "avg_monthly_revenue_3m"
            )
            -
            F.col(
                "avg_monthly_revenue_12m"
            )
        )
        /
        F.abs(
            F.col(
                "avg_monthly_revenue_12m"
            )
        )

    ).otherwise(
        F.lit(0.0)
    )
)


# =========================================================
# 28. PURCHASE MOMENTUM
# =========================================================

monthly = monthly.withColumn(
    "purchase_momentum_3m_vs_12m",

    F.when(
        F.col("purchases_12m") > 0,

        (
            F.col("purchases_3m")
            -
            (
                F.col("purchases_12m")
                /
                F.lit(4.0)
            )
        )
        /
        (
            F.col("purchases_12m")
            /
            F.lit(4.0)
        )

    ).otherwise(
        F.lit(0.0)
    )
)


# =========================================================
# 29. AVG TICKET CHANGE
# =========================================================

monthly = monthly.withColumn(
    "avg_ticket_change_3m_vs_6m",

    F.when(
        F.abs(
            F.col("avg_ticket_6m")
        ) > 1e-6,

        (
            F.col("avg_ticket_3m")
            -
            F.col("avg_ticket_6m")
        )
        /
        F.abs(
            F.col("avg_ticket_6m")
        )

    ).otherwise(
        F.lit(0.0)
    )
)


# =========================================================
# 30. CREDIT NOTE RATE
# =========================================================

monthly = monthly.withColumn(
    "credit_note_rate_6m",

    F.when(
        (
            F.col("purchases_6m")
            +
            F.col("credit_notes_6m")
        ) > 0,

        F.col("credit_notes_6m")
        /
        (
            F.col("purchases_6m")
            +
            F.col("credit_notes_6m")
        )

    ).otherwise(
        F.lit(0.0)
    )
)


# =========================================================
# 31. NORMALIZED TRENDS
# =========================================================

monthly = monthly.withColumn(
    "revenue_trend_6m_normalized",

    F.when(
        F.abs(
            F.col(
                "avg_monthly_revenue_6m"
            )
        ) > 1e-6,

        F.col(
            "revenue_slope_6m"
        )
        /
        F.abs(
            F.col(
                "avg_monthly_revenue_6m"
            )
        )

    ).otherwise(
        F.lit(0.0)
    )
)


monthly = monthly.withColumn(
    "purchase_trend_6m_normalized",

    F.when(
        F.col("purchases_6m") > 0,

        F.col(
            "purchase_slope_6m"
        )
        /
        (
            F.col("purchases_6m")
            /
            F.lit(6.0)
        )

    ).otherwise(
        F.lit(0.0)
    )
)


# =========================================================
# 32. CAP DE RATIOS V2
# =========================================================

def clip(
    column_name,
    lower,
    upper
):

    return (
        F.when(
            F.col(column_name) < lower,
            F.lit(lower)
        )

        .when(
            F.col(column_name) > upper,
            F.lit(upper)
        )

        .otherwise(
            F.col(column_name)
        )
    )


monthly = (
    monthly

    .withColumn(
        "recency_ratio_12m",

        clip(
            "recency_ratio_12m",
            0.0,
            10.0
        )
    )

    .withColumn(
        "recency_ratio_6m",

        clip(
            "recency_ratio_6m",
            0.0,
            10.0
        )
    )

    .withColumn(
        "revenue_momentum_1m_vs_6m",

        clip(
            "revenue_momentum_1m_vs_6m",
            -5.0,
            5.0
        )
    )

    .withColumn(
        "revenue_momentum_3m_vs_12m",

        clip(
            "revenue_momentum_3m_vs_12m",
            -5.0,
            5.0
        )
    )

    .withColumn(
        "purchase_momentum_3m_vs_12m",

        clip(
            "purchase_momentum_3m_vs_12m",
            -3.0,
            5.0
        )
    )

    .withColumn(
        "avg_ticket_change_3m_vs_6m",

        clip(
            "avg_ticket_change_3m_vs_6m",
            -5.0,
            5.0
        )
    )

    .withColumn(
        "revenue_trend_6m_normalized",

        clip(
            "revenue_trend_6m_normalized",
            -5.0,
            5.0
        )
    )

    .withColumn(
        "purchase_trend_6m_normalized",

        clip(
            "purchase_trend_6m_normalized",
            -5.0,
            5.0
        )
    )
)


# =========================================================
# 33. CURRENT DROP %
#
# 0.20 = caída del 20%
# -0.20 = crecimiento del 20%
# =========================================================

monthly = monthly.withColumn(
    "current_drop_pct",

    F.when(
        F.col(
            "previous_3m_revenue"
        ) > 0,

        F.lit(1.0)
        -
        (
            F.col(
                "recent_3m_revenue_feature"
            )
            /
            F.col(
                "previous_3m_revenue"
            )
        )

    ).otherwise(
        F.lit(None)
    )
)


# =========================================================
# 34. QUEDARNOS SOLO CON JULIO 2026
# =========================================================

snapshot = (
    monthly

    .filter(
        F.col("month_start")
        ==
        F.lit(SNAPSHOT_MONTH)
    )
)


snapshot_total = snapshot.count()


print(
    f"\nClientes antes de elegibilidad: "
    f"{snapshot_total:,}"
)


# =========================================================
# 35. ELEGIBILIDAD V3
#
# Misma filosofía que nuestro target oficial:
#
# - mínimo 6 meses de historia
# - baseline > 0
# - recent > 0
# - actividad >= 3 meses en últimos 6
# - la caída del 30% TODAVÍA no es visible
#
# recent > 70% baseline
# =========================================================

eligible = (
    snapshot

    .filter(
        F.col("history_months") >= 6
    )

    .filter(
        F.col(
            "previous_3m_revenue"
        ) > 0
    )

    .filter(
        F.col(
            "recent_3m_revenue_feature"
        ) > 0
    )

    .filter(
        F.col(
            "active_months_6m"
        ) >= 3
    )

    .filter(
        F.col(
            "recent_3m_revenue_feature"
        )
        >
        (
            F.lit(0.70)
            *
            F.col(
                "previous_3m_revenue"
            )
        )
    )
)


eligible_count = eligible.count()


print(
    f"Clientes elegibles para scoring: "
    f"{eligible_count:,}"
)


# =========================================================
# 36. FEATURES EXACTAS DEL MODELO
# =========================================================

FEATURE_COLUMNS = [

    # V1
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

    # V2
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


print(
    f"Features del modelo: "
    f"{len(FEATURE_COLUMNS)}"
)


# =========================================================
# 37. NULLS
# =========================================================

eligible = eligible.fillna(
    0.0,
    subset=FEATURE_COLUMNS
)


null_expressions = [

    F.sum(
        F.when(
            F.col(column_name).isNull(),
            1
        ).otherwise(0)
    ).alias(column_name)

    for column_name in FEATURE_COLUMNS
]


null_row = (
    eligible
    .agg(
        *null_expressions
    )
    .first()
)


total_nulls = sum(
    null_row[column_name]
    for column_name
    in FEATURE_COLUMNS
)


print(
    f"Nulls en las 43 features: "
    f"{total_nulls}"
)


if total_nulls > 0:

    raise ValueError(
        "Existen nulls en features "
        "de scoring."
    )


# =========================================================
# 38. DATASET FINAL
#
# Conservamos además algunos campos explicativos
# para construir luego el ranking comercial.
# =========================================================

final_scoring = eligible.select(

    "customer_id",
    "month_start",

    # Contexto negocio
    "current_drop_pct",
    "history_months",

    # Las 43 features
    *FEATURE_COLUMNS
)


# =========================================================
# 39. VALIDACIONES FINALES
# =========================================================

print("\n" + "=" * 70)
print("CURRENT SCORING DATASET")
print("=" * 70)


print(
    f"\nSnapshot: "
    f"{SNAPSHOT_MONTH}"
)

print(
    f"Observaciones elegibles: "
    f"{final_scoring.count():,}"
)


print(
    "\nCURRENT DROP DISTRIBUTION:"
)


final_scoring.select(

    F.min(
        "current_drop_pct"
    ).alias("min"),

    F.avg(
        "current_drop_pct"
    ).alias("avg"),

    F.max(
        "current_drop_pct"
    ).alias("max"),

).show(
    truncate=False
)


print(
    "\nEJEMPLOS DE CLIENTES ELEGIBLES:"
)


final_scoring.select(

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

).orderBy(

    F.desc(
        "revenue_12m"
    )

).show(
    20,
    truncate=False
)


# =========================================================
# 40. GUARDAR
# =========================================================

(
    final_scoring

    .write

    .mode("overwrite")

    .parquet(
        OUTPUT_PATH
    )
)


print(
    f"\nCurrent scoring features "
    f"guardadas en:"
)

print(
    OUTPUT_PATH
)


print(
    "\nIMPORTANTE:"
)

print(
    "Este dataset no contiene target "
    "ni información futura."
)

print(
    "Solo utiliza datos disponibles "
    "hasta julio de 2026."
)


# =========================================================
# 41. FIN
# =========================================================

spark.stop()