from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window


# =========================================================
# 0. SPARK
# =========================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-build-ml-features-v2")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 1. PATHS
# =========================================================

BASE_FEATURES_PATH = "data/gold/ml_features"
CUSTOMER_MONTHLY_PATH = "data/gold/customer_monthly"

OUTPUT_PATH = "data/gold/ml_features_v2"


# =========================================================
# 2. CARGAR DATOS
# =========================================================

base = spark.read.parquet(BASE_FEATURES_PATH)

monthly_raw = spark.read.parquet(
    CUSTOMER_MONTHLY_PATH
)


print("\n" + "=" * 70)
print("BUILD ML FEATURES V2")
print("=" * 70)

print(
    f"\nBase ML observations: "
    f"{base.count():,}"
)


# =========================================================
# 3. RESOLVER NOMBRES DE COLUMNAS
#
# Lo hacemos robusto por si el nombre exacto del campo
# mensual es ligeramente distinto.
# =========================================================

def resolve_column(df, candidates, description):

    for candidate in candidates:

        if candidate in df.columns:
            print(
                f"{description}: "
                f"{candidate}"
            )
            return candidate

    raise ValueError(
        f"No encuentro columna para {description}. "
        f"Probadas: {candidates}. "
        f"Columnas disponibles: {df.columns}"
    )


revenue_col = resolve_column(
    monthly_raw,
    [
        "monthly_net_revenue",
        "net_revenue",
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
    "Monthly purchase count"
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
# 4. PREPARAR CUSTOMER MONTHLY
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

        F.col(recency_col)
        .cast("double")
        .alias("monthly_recency_days"),
    )
    .fillna(
        {
            "monthly_revenue": 0.0,
            "monthly_purchases": 0.0,
        }
    )
    .withColumn(
        "monthly_active",

        F.when(
            F.col("monthly_purchases") > 0,
            1
        ).otherwise(0)
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
        f"Hay {duplicates} customer-month duplicados "
        f"en customer_monthly."
    )


print(
    f"Customer-month duplicates: "
    f"{duplicates}"
)


# =========================================================
# 6. WINDOWS
# =========================================================

customer_window = (
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


# =========================================================
# 7. LAGS DE REVENUE Y COMPRAS
#
# Nos permiten calcular tendencias de 6 meses.
#
# lag_5 = mes más antiguo
# lag_0 = mes actual
# =========================================================

for lag_value in range(6):

    monthly = monthly.withColumn(
        f"revenue_lag_{lag_value}",

        F.lag(
            "monthly_revenue",
            lag_value
        ).over(customer_window)
    )

    monthly = monthly.withColumn(
        f"purchases_lag_{lag_value}",

        F.lag(
            "monthly_purchases",
            lag_value
        ).over(customer_window)
    )


# =========================================================
# 8. REVENUE TREND 6M
#
# Regresión lineal simple sobre:
#
# x = 0,1,2,3,4,5
#
# donde:
# 0 = mes más antiguo
# 5 = mes actual
#
# slope > 0 -> tendencia creciente
# slope < 0 -> tendencia decreciente
# =========================================================

revenue_slope_6m = (

    (
        -2.5 * F.col("revenue_lag_5")
        -1.5 * F.col("revenue_lag_4")
        -0.5 * F.col("revenue_lag_3")
        +0.5 * F.col("revenue_lag_2")
        +1.5 * F.col("revenue_lag_1")
        +2.5 * F.col("revenue_lag_0")
    )
    / F.lit(17.5)
)


monthly = monthly.withColumn(
    "revenue_slope_6m",
    revenue_slope_6m
)


# =========================================================
# 9. PURCHASE TREND 6M
# =========================================================

purchase_slope_6m = (

    (
        -2.5 * F.col("purchases_lag_5")
        -1.5 * F.col("purchases_lag_4")
        -0.5 * F.col("purchases_lag_3")
        +0.5 * F.col("purchases_lag_2")
        +1.5 * F.col("purchases_lag_1")
        +2.5 * F.col("purchases_lag_0")
    )
    / F.lit(17.5)
)


monthly = monthly.withColumn(
    "purchase_slope_6m",
    purchase_slope_6m
)


# =========================================================
# 10. ÚLTIMO MES ACTIVO
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


# =========================================================
# 11. RACHA DE INACTIVIDAD
#
# Ejemplo:
#
# Marzo compra
# Abril 0
# Mayo  0
#
# Mayo -> inactivity_streak_months = 2
# =========================================================

monthly = monthly.withColumn(
    "inactivity_streak_months",

    F.when(
        F.col("monthly_active") == 1,
        F.lit(0.0)
    )
    .when(
        F.col("last_active_month").isNotNull(),

        F.months_between(
            F.col("month_start"),
            F.col("last_active_month")
        )
    )
    .otherwise(
        F.lit(0.0)
    )
)


# =========================================================
# 12. DATOS MENSUALES NECESARIOS PARA JOIN
# =========================================================

monthly_extra = monthly.select(

    "customer_id",
    "month_start",

    "revenue_slope_6m",
    "purchase_slope_6m",
    "inactivity_streak_months"
)


# =========================================================
# 13. JOIN CON FEATURES V1
# =========================================================

features = (
    base
    .withColumn(
        "month_start",
        F.col("month_start").cast("date")
    )
    .join(
        monthly_extra,
        [
            "customer_id",
            "month_start"
        ],
        "left"
    )
)


# =========================================================
# 14. FEATURES PERSONALIZADAS
# =========================================================


# ---------------------------------------------------------
# A. FRECUENCIA ESPERADA DEL CLIENTE - 12M
#
# IMPORTANTE:
#
# Ya NO usamos purchases_12m.
#
# Usamos active_months_12m porque queremos aproximar
# la cadencia habitual del cliente, no el número de
# facturas que genera.
#
# Ejemplos:
#
# 12 meses activos / año -> ~30 días entre meses activos
#  6 meses activos / año -> ~61 días
#  4 meses activos / año -> ~91 días
# ---------------------------------------------------------

features = features.withColumn(
    "expected_days_between_purchases_12m",

    F.when(
        F.col("active_months_12m") > 0,

        F.lit(365.25)
        /
        F.col("active_months_12m")

    ).otherwise(
        F.lit(365.25)
    )
)


# ---------------------------------------------------------
# B. FRECUENCIA ESPERADA DEL CLIENTE - 6M
#
# Igual que arriba, pero usando meses activos durante
# los últimos seis meses.
# ---------------------------------------------------------

features = features.withColumn(
    "expected_days_between_purchases_6m",

    F.when(
        F.col("active_months_6m") > 0,

        F.lit(182.625)
        /
        F.col("active_months_6m")

    ).otherwise(
        F.lit(182.625)
    )
)


# ---------------------------------------------------------
# C. RECENCY RELATIVA AL PATRÓN DEL PROPIO CLIENTE
#
# Ejemplo:
#
# Cliente suele comprar cada ~30 días
# y lleva 45 días:
#
# recency_ratio = 1.5
#
# Cliente suele comprar cada ~60 días
# y lleva 45:
#
# recency_ratio = 0.75
# ---------------------------------------------------------

features = features.withColumn(
    "recency_ratio_12m",

    F.col("recency_days")
    /
    F.col(
        "expected_days_between_purchases_12m"
    )
)


features = features.withColumn(
    "recency_ratio_6m",

    F.col("recency_days")
    /
    F.col(
        "expected_days_between_purchases_6m"
    )
)


# ---------------------------------------------------------
# D. REGULARIDAD
#
# Qué porcentaje de los últimos meses estuvo activo.
# ---------------------------------------------------------

features = (
    features

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


# ---------------------------------------------------------
# E. INTENSIDAD CUANDO ESTÁ ACTIVO
#
# Evita tratar igual:
#
# - cliente que hace 12 compras en 12 meses
# - cliente que hace 12 compras en 4 meses
# ---------------------------------------------------------

features = features.withColumn(
    "purchases_per_active_month_12m",

    F.when(
        F.col("active_months_12m") > 0,

        F.col("purchases_12m")
        /
        F.col("active_months_12m")

    ).otherwise(
        F.lit(0.0)
    )
)


features = features.withColumn(
    "revenue_per_active_month_12m",

    F.when(
        F.col("active_months_12m") > 0,

        F.col("revenue_12m")
        /
        F.col("active_months_12m")

    ).otherwise(
        F.lit(0.0)
    )
)


# ---------------------------------------------------------
# F. MOMENTUM REVENUE 1M VS 6M
#
# Compara el último mes contra el comportamiento
# mensual reciente del propio cliente.
# ---------------------------------------------------------

features = features.withColumn(
    "revenue_momentum_1m_vs_6m",

    F.when(
        F.abs(
            F.col("avg_monthly_revenue_6m")
        ) > 1e-6,

        (
            F.col("revenue_1m")
            -
            F.col("avg_monthly_revenue_6m")
        )
        /
        F.abs(
            F.col("avg_monthly_revenue_6m")
        )

    ).otherwise(
        F.lit(0.0)
    )
)


# ---------------------------------------------------------
# G. MOMENTUM REVENUE 3M VS 12M
# ---------------------------------------------------------

features = features.withColumn(
    "revenue_momentum_3m_vs_12m",

    F.when(
        F.abs(
            F.col("avg_monthly_revenue_12m")
        ) > 1e-6,

        (
            F.col("avg_monthly_revenue_3m")
            -
            F.col("avg_monthly_revenue_12m")
        )
        /
        F.abs(
            F.col("avg_monthly_revenue_12m")
        )

    ).otherwise(
        F.lit(0.0)
    )
)


# ---------------------------------------------------------
# H. MOMENTUM DE FRECUENCIA DE COMPRA
#
# purchases_12m / 4 representa aproximadamente
# las compras esperadas durante un periodo de 3 meses.
# ---------------------------------------------------------

features = features.withColumn(
    "purchase_momentum_3m_vs_12m",

    F.when(
        F.col("purchases_12m") > 0,

        (
            F.col("purchases_3m")
            -
            (
                F.col("purchases_12m")
                / F.lit(4.0)
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


# ---------------------------------------------------------
# I. CAMBIO DE TICKET MEDIO
# ---------------------------------------------------------

features = features.withColumn(
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


# ---------------------------------------------------------
# J. RATIO DE CREDIT NOTES
# ---------------------------------------------------------

features = features.withColumn(
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
# 15. NORMALIZAR TENDENCIAS
#
# La pendiente absoluta depende mucho del tamaño del
# cliente.
#
# También creamos una versión relativa.
# =========================================================

features = features.withColumn(
    "revenue_trend_6m_normalized",

    F.when(
        F.abs(
            F.col("avg_monthly_revenue_6m")
        ) > 1e-6,

        F.col("revenue_slope_6m")
        /
        F.abs(
            F.col("avg_monthly_revenue_6m")
        )

    ).otherwise(
        F.lit(0.0)
    )
)


features = features.withColumn(
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

    ).otherwise(
        F.lit(0.0)
    )
)


# =========================================================
# 16. CAP DE RATIOS EXTREMOS
#
# No eliminamos observaciones.
#
# Solo limitamos ratios extremos generados por
# denominadores pequeños.
# =========================================================

def clip(column_name, lower, upper):

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


features = (
    features

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
# 17. LIMPIAR COLUMNAS AUXILIARES
#
# Conservamos expected_days_between_purchases_12m.
#
# La versión de 6 meses solo la necesitamos para construir
# recency_ratio_6m.
# =========================================================

features = features.drop(
    "expected_days_between_purchases_6m"
)


# =========================================================
# 18. LISTA DE NUEVAS FEATURES
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
# 19. RELLENAR NULLS DE NUEVAS FEATURES
#
# Los primeros meses de un cliente pueden no tener
# suficientes lags para calcular slope.
#
# En esos casos ponemos 0.
# =========================================================

features = features.fillna(
    0.0,
    subset=NEW_FEATURE_COLUMNS
)


# =========================================================
# 20. VALIDACIÓN DE NULLS
# =========================================================

null_expressions = [

    F.sum(
        F.when(
            F.col(column_name).isNull(),
            1
        ).otherwise(0)
    ).alias(column_name)

    for column_name in NEW_FEATURE_COLUMNS
]


null_row = (
    features
    .agg(*null_expressions)
    .first()
)


print("\nNULLS EN NUEVAS FEATURES:")

total_nulls = 0


for column_name in NEW_FEATURE_COLUMNS:

    null_count = null_row[column_name]

    total_nulls += null_count

    print(
        f"{column_name:<38} "
        f"{null_count}"
    )


if total_nulls > 0:

    raise ValueError(
        f"Hay {total_nulls} nulls "
        f"en features V2."
    )


# =========================================================
# 21. VALIDAR FILAS
#
# V2 debe contener exactamente las mismas observaciones
# que V1.
# =========================================================

base_count = base.count()
v2_count = features.count()


print("\nVALIDACIÓN DE FILAS:")

print(
    f"V1 observations: {base_count:,}"
)

print(
    f"V2 observations: {v2_count:,}"
)


if base_count != v2_count:

    raise ValueError(
        "V2 no tiene el mismo número de filas que V1."
    )


# =========================================================
# 22. TARGET DISTRIBUTION
# =========================================================

print("\nTARGET DISTRIBUTION:")

features.groupBy(
    "target_drop_30"
).count().orderBy(
    "target_drop_30"
).show()


# =========================================================
# 23. RESUMEN DE NUEVAS FEATURES
# =========================================================

print("\nRESUMEN DE FEATURES V2:")

features.select(
    NEW_FEATURE_COLUMNS
).summary(
    "count",
    "mean",
    "stddev",
    "min",
    "50%",
    "max"
).show(
    truncate=False
)


# =========================================================
# 24. EJEMPLOS
#
# Aquí podremos comprobar visualmente si ahora
# recency_ratio tiene más sentido comercial.
# =========================================================

print("\nEJEMPLOS V2:")

features.select(

    "customer_id",
    "month_start",
    "target_drop_30",

    "recency_days",

    "purchases_12m",
    "active_months_12m",

    "expected_days_between_purchases_12m",
    "recency_ratio_12m",

    "inactivity_streak_months",

    "revenue_change_3m",
    "revenue_trend_6m_normalized",

    "purchase_momentum_3m_vs_12m",

).orderBy(
    F.desc("recency_ratio_12m")
).show(
    20,
    truncate=False
)


# =========================================================
# 25. GUARDAR
# =========================================================

(
    features
    .write
    .mode("overwrite")
    .parquet(OUTPUT_PATH)
)


print(
    f"\nFeatures V2 guardadas en: "
    f"{OUTPUT_PATH}"
)

print(
    f"Nuevas features añadidas: "
    f"{len(NEW_FEATURE_COLUMNS)}"
)

print(
    f"Features totales aproximadas: "
    f"{len(features.columns) - 3}"
)


# =========================================================
# 26. FIN
# =========================================================

spark.stop()