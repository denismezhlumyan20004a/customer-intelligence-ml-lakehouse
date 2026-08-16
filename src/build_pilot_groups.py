from pathlib import Path

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession

from sklearn.preprocessing import RobustScaler


# =========================================================
# 0. CONFIG
# =========================================================

SCORING_PATH = "data/gold/current_customer_scores"

OUTPUT_PARQUET = "data/gold/pilot_groups"

OUTPUT_DIR = Path("outputs/pilot")


# CAMBIAR SOLO SI EL PILOTO EMPIEZA OTRA FECHA
PILOT_START_DATE = "2026-08-17"

PILOT_DURATION_DAYS = 90


# =========================================================
# EXPERIMENT SIZE
# =========================================================

# Buscamos candidatos entre los 70 clientes
# con mayor risk score.
#
# De estos necesitamos finalmente:
#
# 25 parejas
# = 25 INTERVENTION
# + 25 CONTROL

TOP_N_CANDIDATES = 70

N_PAIRS = 25


# =========================================================
# MATCHING CALIPERS
# =========================================================

# Una pareja no puede diferir más de 10 puntos
# absolutos de risk score.
MAX_RISK_DIFFERENCE = 0.10


# El cliente con mayor revenue_12m no puede
# tener más de 3x el revenue del otro.
MAX_REVENUE_RATIO = 3.0


# Además del caliper, rechazamos pares con una
# distancia multivariable excesiva.
#
# Esto evita casos como:
# 005414 vs un cliente que cumpla revenue/risk
# por poco pero sea muy diferente en frecuencia,
# recencia, tendencia, etc.
MAX_MATCH_DISTANCE = 3.0


# =========================================================
# BALANCE GLOBAL
# =========================================================

# Ideal:
# abs(SMD) < 0.10
#
# Aceptable:
# abs(SMD) < 0.25
#
# >= 0.25:
# NO arrancamos.
MAX_ALLOWED_SMD = 0.25


# =========================================================
# SEARCH SETTINGS
# =========================================================

RANDOM_SEED = 42


# Para encontrar buenas 25 parejas:
N_MATCHING_RESTARTS = 5000


# Después de formar las parejas,
# buscamos una asignación Intervention / Control
# baseline equilibrada.
N_RANDOMIZATIONS = 50000


# =========================================================
# PILOT DEFINITIONS
# =========================================================

INTERVENTION_DEFINITION = (
    "Contacto comercial proactivo motivado por el piloto, "
    "mediante visita presencial o llamada comercial estructurada, "
    "con fecha, motivo detectado, accion realizada y resultado registrado."
)


CONTROL_DEFINITION = (
    "Sin accion comercial especial motivada por el piloto; "
    "el cliente mantiene la operativa comercial habitual."
)


# =========================================================
# 1. SPARK
# =========================================================

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("retailco-build-randomized-matched-pilot-final")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# =========================================================
# 2. LOAD CURRENT SCORES
# =========================================================

scores = spark.read.parquet(
    SCORING_PATH
)


print("\n" + "=" * 82)
print("BUILD RANDOMIZED MATCHED COMMERCIAL PILOT - FINAL DESIGN")
print("=" * 82)


total_customers = scores.count()


print(
    f"\nClientes elegibles disponibles: "
    f"{total_customers}"
)


# =========================================================
# 3. REQUIRED COLUMNS
# =========================================================

required_columns = [
    "customer_id",
    "month_start",

    "risk_rank",
    "risk_probability",

    "revenue_3m",
    "revenue_6m",
    "revenue_12m",

    "purchases_12m",

    "recency_days",

    "revenue_change_3m",

    "current_drop_pct",
]


missing_columns = [
    column
    for column in required_columns
    if column not in scores.columns
]


if missing_columns:

    raise ValueError(
        f"Faltan columnas necesarias: "
        f"{missing_columns}"
    )


# =========================================================
# 4. TO PANDAS
# =========================================================

pdf = (
    scores

    .select(
        *required_columns
    )

    .toPandas()

    .sort_values(
        [
            "risk_rank",
            "customer_id",
        ]
    )

    .reset_index(
        drop=True
    )
)


# =========================================================
# 5. CANDIDATE POPULATION
# =========================================================

candidates = (
    pdf

    .head(
        TOP_N_CANDIDATES
    )

    .copy()

    .reset_index(
        drop=True
    )
)


print(
    f"Pool experimental de alto riesgo: "
    f"{len(candidates)}"
)


print(
    f"Risk rank considerado: "
    f"1 -> "
    f"{int(candidates['risk_rank'].max())}"
)


print(
    f"Riesgo medio del pool: "
    f"{candidates['risk_probability'].mean():.4f}"
)


# =========================================================
# 6. PREP MATCHING FEATURES
# =========================================================

def prepare_matching_features(df):

    output = df.copy()


    output["log_revenue_12m"] = np.log1p(
        np.maximum(
            output["revenue_12m"].astype(float),
            0.0
        )
    )


    output["log_revenue_6m"] = np.log1p(
        np.maximum(
            output["revenue_6m"].astype(float),
            0.0
        )
    )


    output["log_revenue_3m"] = np.log1p(
        np.maximum(
            output["revenue_3m"].astype(float),
            0.0
        )
    )


    output["log_purchases_12m"] = np.log1p(
        np.maximum(
            output["purchases_12m"].astype(float),
            0.0
        )
    )


    output["log_recency_days"] = np.log1p(
        np.maximum(
            output["recency_days"].astype(float),
            0.0
        )
    )


    output["revenue_change_3m_clipped"] = (
        output["revenue_change_3m"]
        .astype(float)
        .clip(
            -3.0,
            3.0
        )
    )


    return output


matching_df = prepare_matching_features(
    candidates
)


# =========================================================
# 7. MATCH FEATURES + WEIGHTS
# =========================================================

MATCH_COLUMNS = [
    "risk_probability",

    "log_revenue_12m",
    "log_revenue_6m",
    "log_revenue_3m",

    "log_purchases_12m",

    "log_recency_days",

    "revenue_change_3m_clipped",
]


MATCH_WEIGHTS = np.array([
    4.0,  # risk
    2.5,  # revenue 12m
    1.0,  # revenue 6m
    1.0,  # revenue 3m
    1.0,  # purchases
    1.0,  # recency
    1.5,  # trend
])


# =========================================================
# 8. SCALE
# =========================================================

scaler = RobustScaler()


scaled = scaler.fit_transform(
    matching_df[
        MATCH_COLUMNS
    ]
)


scaled = (
    scaled
    *
    np.sqrt(
        MATCH_WEIGHTS
    )
)


# =========================================================
# 9. RAW DISTANCE MATRIX
# =========================================================

distance_matrix = np.sqrt(

    (
        (
            scaled[:, None, :]
            -
            scaled[None, :, :]
        )
        ** 2
    )

    .sum(
        axis=2
    )
)


np.fill_diagonal(
    distance_matrix,
    np.inf
)


# =========================================================
# 10. HELPERS
# =========================================================

def revenue_ratio(
    revenue_a,
    revenue_b
):

    revenue_a = float(
        revenue_a
    )

    revenue_b = float(
        revenue_b
    )


    if (
        revenue_a <= 0
        or
        revenue_b <= 0
    ):

        return np.inf


    return (
        max(
            revenue_a,
            revenue_b
        )
        /
        min(
            revenue_a,
            revenue_b
        )
    )


def risk_difference(
    risk_a,
    risk_b
):

    return abs(
        float(risk_a)
        -
        float(risk_b)
    )


# =========================================================
# 11. BUILD VALID PAIR LIST
#
# Antes de emparejar:
#
# 1. risk diff <= 0.10
# 2. revenue ratio <= 3x
# 3. match distance <= 3
# =========================================================

valid_pair_candidates = []


for i in range(
    len(candidates)
):

    for j in range(
        i + 1,
        len(candidates)
    ):

        row_i = (
            candidates
            .iloc[i]
        )

        row_j = (
            candidates
            .iloc[j]
        )


        risk_diff = risk_difference(
            row_i[
                "risk_probability"
            ],
            row_j[
                "risk_probability"
            ],
        )


        if (
            risk_diff
            >
            MAX_RISK_DIFFERENCE
        ):

            continue


        rev_ratio = revenue_ratio(
            row_i[
                "revenue_12m"
            ],
            row_j[
                "revenue_12m"
            ],
        )


        if (
            rev_ratio
            >
            MAX_REVENUE_RATIO
        ):

            continue


        distance = float(
            distance_matrix[
                i,
                j
            ]
        )


        if (
            distance
            >
            MAX_MATCH_DISTANCE
        ):

            continue


        valid_pair_candidates.append(
            {
                "i":
                    i,

                "j":
                    j,

                "distance":
                    distance,

                "risk_difference":
                    risk_diff,

                "revenue_ratio":
                    rev_ratio,
            }
        )


print(
    f"\nParejas candidatas válidas: "
    f"{len(valid_pair_candidates)}"
)


if len(valid_pair_candidates) == 0:

    raise RuntimeError(
        "No existen parejas válidas."
    )


# =========================================================
# 12. MULTI-START MATCHING
#
# Tenemos 70 candidatos y necesitamos 25 pares.
#
# En cada intento:
# - añadimos una pequeña perturbación aleatoria
# - priorizamos pares cercanos
# - impedimos reutilizar clientes
#
# Nos quedamos con el conjunto de 25 parejas
# con menor distancia total/media.
#
# Esto evita que un greedy único nos deje
# atrapados por decisiones tempranas.
# =========================================================

rng_matching = np.random.default_rng(
    RANDOM_SEED
)


best_pairs = None

best_total_distance = np.inf

best_max_distance = np.inf


for restart_id in range(
    N_MATCHING_RESTARTS
):

    scored_pairs = []


    for pair in valid_pair_candidates:

        # Jitter pequeño para explorar
        # diferentes combinaciones.
        jitter = rng_matching.uniform(
            0.0,
            0.20
        )


        search_score = (
            pair[
                "distance"
            ]
            +
            jitter
        )


        scored_pairs.append(
            (
                search_score,
                pair
            )
        )


    scored_pairs.sort(
        key=lambda x: x[0]
    )


    used = set()

    selected = []


    for _, pair in scored_pairs:

        i = pair["i"]
        j = pair["j"]


        if i in used:
            continue

        if j in used:
            continue


        selected.append(
            pair
        )


        used.add(i)
        used.add(j)


        if (
            len(selected)
            ==
            N_PAIRS
        ):

            break


    if (
        len(selected)
        <
        N_PAIRS
    ):

        continue


    total_distance = sum(
        pair[
            "distance"
        ]
        for pair in selected
    )


    max_distance = max(
        pair[
            "distance"
        ]
        for pair in selected
    )


    if (
        total_distance
        <
        best_total_distance
    ) or (
        np.isclose(
            total_distance,
            best_total_distance
        )
        and
        max_distance
        <
        best_max_distance
    ):

        best_pairs = (
            selected.copy()
        )

        best_total_distance = (
            total_distance
        )

        best_max_distance = (
            max_distance
        )


# =========================================================
# 13. VALIDATE MATCHING
# =========================================================

if best_pairs is None:

    raise RuntimeError(
        "No se pudieron formar "
        "25 parejas válidas con los calipers actuales."
    )


print(
    f"Parejas finales encontradas: "
    f"{len(best_pairs)}"
)


print(
    f"Matching restarts evaluados: "
    f"{N_MATCHING_RESTARTS:,}"
)


print(
    f"Distancia media parejas: "
    f"{best_total_distance / N_PAIRS:.4f}"
)


print(
    f"Distancia máxima parejas: "
    f"{best_max_distance:.4f}"
)


# =========================================================
# 14. STANDARDIZED MEAN DIFFERENCE
# =========================================================

def standardized_mean_difference(
    treated_values,
    control_values
):

    treated_values = np.asarray(
        treated_values,
        dtype=float
    )

    control_values = np.asarray(
        control_values,
        dtype=float
    )


    mean_t = np.mean(
        treated_values
    )

    mean_c = np.mean(
        control_values
    )


    std_t = np.std(
        treated_values,
        ddof=1
    )

    std_c = np.std(
        control_values,
        ddof=1
    )


    pooled_std = np.sqrt(
        (
            std_t ** 2
            +
            std_c ** 2
        )
        /
        2.0
    )


    if (
        np.isnan(pooled_std)
        or
        pooled_std < 1e-12
    ):

        return 0.0


    return (
        mean_t
        -
        mean_c
    ) / pooled_std


# =========================================================
# 15. BALANCE VARIABLES
# =========================================================

BALANCE_COLUMNS = [
    "risk_probability",

    "revenue_3m",
    "revenue_6m",
    "revenue_12m",

    "purchases_12m",

    "recency_days",

    "revenue_change_3m",
]


# =========================================================
# 16. CONSTRAINED RANDOMIZATION
#
# MUY IMPORTANTE:
#
# Las parejas YA están formadas.
#
# Ahora sorteamos:
#
# cliente A -> INTERVENTION
# cliente B -> CONTROL
#
# o al revés.
#
# Probamos muchas randomizaciones
# PRE-INTERVENTION y nos quedamos con
# una asignación baseline equilibrada.
#
# No usamos ningún dato futuro.
# =========================================================

rng_randomization = np.random.default_rng(
    RANDOM_SEED + 1000
)


best_assignment = None

best_max_smd = np.inf

best_mean_smd = np.inf


for randomization_id in range(
    N_RANDOMIZATIONS
):

    intervention_indices = []
    control_indices = []


    for pair in best_pairs:

        i = pair["i"]
        j = pair["j"]


        if (
            rng_randomization.random()
            <
            0.5
        ):

            intervention_indices.append(
                i
            )

            control_indices.append(
                j
            )

        else:

            intervention_indices.append(
                j
            )

            control_indices.append(
                i
            )


    intervention_candidate = (
        candidates
        .iloc[
            intervention_indices
        ]
    )


    control_candidate = (
        candidates
        .iloc[
            control_indices
        ]
    )


    abs_smds = []


    for column in BALANCE_COLUMNS:

        smd = standardized_mean_difference(

            intervention_candidate[
                column
            ],

            control_candidate[
                column
            ],
        )


        abs_smds.append(
            abs(
                smd
            )
        )


    max_smd = max(
        abs_smds
    )


    mean_smd = np.mean(
        abs_smds
    )


    # Objetivo:
    # minimizar primero la peor variable.
    # Luego el desequilibrio medio.
    if (
        max_smd
        <
        best_max_smd
    ) or (
        np.isclose(
            max_smd,
            best_max_smd
        )
        and
        mean_smd
        <
        best_mean_smd
    ):

        best_assignment = {
            "randomization_id":
                randomization_id,

            "intervention_indices":
                intervention_indices.copy(),

            "control_indices":
                control_indices.copy(),
        }


        best_max_smd = (
            max_smd
        )


        best_mean_smd = (
            mean_smd
        )


if best_assignment is None:

    raise RuntimeError(
        "No se encontró randomización válida."
    )


# =========================================================
# 17. FINAL GROUP IDS
# =========================================================

intervention_indices = set(
    best_assignment[
        "intervention_indices"
    ]
)


control_indices = set(
    best_assignment[
        "control_indices"
    ]
)


# =========================================================
# 18. FINAL PAIR TABLE
# =========================================================

pair_rows = []


for pair_id, pair in enumerate(
    best_pairs,
    start=1
):

    i = pair["i"]
    j = pair["j"]


    if i in intervention_indices:

        intervention_idx = i
        control_idx = j

    else:

        intervention_idx = j
        control_idx = i


    intervention_row = (
        candidates
        .iloc[
            intervention_idx
        ]
    )


    control_row = (
        candidates
        .iloc[
            control_idx
        ]
    )


    pair_rows.append(
        {
            "pair_id":
                pair_id,

            "intervention_customer_id":
                intervention_row[
                    "customer_id"
                ],

            "control_customer_id":
                control_row[
                    "customer_id"
                ],

            "intervention_risk_rank":
                int(
                    intervention_row[
                        "risk_rank"
                    ]
                ),

            "control_risk_rank":
                int(
                    control_row[
                        "risk_rank"
                    ]
                ),

            "intervention_risk":
                float(
                    intervention_row[
                        "risk_probability"
                    ]
                ),

            "control_risk":
                float(
                    control_row[
                        "risk_probability"
                    ]
                ),

            "risk_difference":
                pair[
                    "risk_difference"
                ],

            "intervention_revenue_3m":
                float(
                    intervention_row[
                        "revenue_3m"
                    ]
                ),

            "control_revenue_3m":
                float(
                    control_row[
                        "revenue_3m"
                    ]
                ),

            "intervention_revenue_6m":
                float(
                    intervention_row[
                        "revenue_6m"
                    ]
                ),

            "control_revenue_6m":
                float(
                    control_row[
                        "revenue_6m"
                    ]
                ),

            "intervention_revenue_12m":
                float(
                    intervention_row[
                        "revenue_12m"
                    ]
                ),

            "control_revenue_12m":
                float(
                    control_row[
                        "revenue_12m"
                    ]
                ),

            "revenue_ratio":
                pair[
                    "revenue_ratio"
                ],

            "intervention_purchases_12m":
                float(
                    intervention_row[
                        "purchases_12m"
                    ]
                ),

            "control_purchases_12m":
                float(
                    control_row[
                        "purchases_12m"
                    ]
                ),

            "intervention_recency_days":
                float(
                    intervention_row[
                        "recency_days"
                    ]
                ),

            "control_recency_days":
                float(
                    control_row[
                        "recency_days"
                    ]
                ),

            "intervention_revenue_change_3m":
                float(
                    intervention_row[
                        "revenue_change_3m"
                    ]
                ),

            "control_revenue_change_3m":
                float(
                    control_row[
                        "revenue_change_3m"
                    ]
                ),

            "match_distance":
                pair[
                    "distance"
                ],
        }
    )


pairs = pd.DataFrame(
    pair_rows
)


# =========================================================
# 19. MATCH QUALITY
# =========================================================

distance_q75 = (
    pairs[
        "match_distance"
    ]
    .quantile(
        0.75
    )
)


distance_q90 = (
    pairs[
        "match_distance"
    ]
    .quantile(
        0.90
    )
)


pairs["match_quality"] = np.where(

    pairs[
        "match_distance"
    ]
    <=
    distance_q75,

    "GOOD",

    np.where(

        pairs[
            "match_distance"
        ]
        <=
        distance_q90,

        "ACCEPTABLE",

        "REVIEW",
    )
)


# =========================================================
# 20. LONG-FORM PILOT TABLE
# =========================================================

pilot_rows = []


for _, pair in pairs.iterrows():

    for group, customer_id in [
        (
            "INTERVENTION",
            pair[
                "intervention_customer_id"
            ],
        ),

        (
            "CONTROL",
            pair[
                "control_customer_id"
            ],
        ),
    ]:

        row = (
            candidates[
                candidates[
                    "customer_id"
                ]
                ==
                customer_id
            ]
            .iloc[0]
        )


        pilot_rows.append(
            {
                "pair_id":
                    int(
                        pair[
                            "pair_id"
                        ]
                    ),

                "group":
                    group,

                "customer_id":
                    row[
                        "customer_id"
                    ],

                "risk_rank":
                    int(
                        row[
                            "risk_rank"
                        ]
                    ),

                "risk_probability":
                    float(
                        row[
                            "risk_probability"
                        ]
                    ),

                "revenue_3m":
                    float(
                        row[
                            "revenue_3m"
                        ]
                    ),

                "revenue_6m":
                    float(
                        row[
                            "revenue_6m"
                        ]
                    ),

                "revenue_12m":
                    float(
                        row[
                            "revenue_12m"
                        ]
                    ),

                "purchases_12m":
                    float(
                        row[
                            "purchases_12m"
                        ]
                    ),

                "recency_days":
                    float(
                        row[
                            "recency_days"
                        ]
                    ),

                "revenue_change_3m":
                    float(
                        row[
                            "revenue_change_3m"
                        ]
                    ),

                "current_drop_pct":
                    float(
                        row[
                            "current_drop_pct"
                        ]
                    ),

                "snapshot_month":
                    str(
                        row[
                            "month_start"
                        ]
                    ),

                "pilot_start_date":
                    PILOT_START_DATE,

                "pilot_duration_days":
                    PILOT_DURATION_DAYS,

                "match_distance":
                    float(
                        pair[
                            "match_distance"
                        ]
                    ),

                "revenue_pair_ratio":
                    float(
                        pair[
                            "revenue_ratio"
                        ]
                    ),

                "risk_pair_difference":
                    float(
                        pair[
                            "risk_difference"
                        ]
                    ),

                "special_pilot_intervention":
                    (
                        1
                        if group
                        ==
                        "INTERVENTION"
                        else
                        0
                    ),

                "intervention_definition":
                    (
                        INTERVENTION_DEFINITION
                        if group
                        ==
                        "INTERVENTION"
                        else
                        CONTROL_DEFINITION
                    ),
            }
        )


pilot = pd.DataFrame(
    pilot_rows
)


# =========================================================
# 21. IDENTIFY EXPERIMENT-EXCLUDED HIGH-RISK CLIENTS
#
# Top-risk clients that were in the pool
# but did not enter the 50 experimental accounts.
#
# They may STILL receive commercial attention
# OUTSIDE the experiment.
# =========================================================

experiment_ids = set(
    pilot[
        "customer_id"
    ]
)


excluded = (
    candidates[
        ~candidates[
            "customer_id"
        ]
        .isin(
            experiment_ids
        )
    ][
        [
            "customer_id",
            "risk_rank",
            "risk_probability",
            "revenue_12m",
        ]
    ]

    .copy()

    .sort_values(
        "risk_rank"
    )
)


excluded["reason"] = (
    "NOT_SELECTED_IN_FINAL_MATCHED_EXPERIMENT"
)


# =========================================================
# 22. BALANCE REPORT
# =========================================================

treated = (
    pilot[
        pilot["group"]
        ==
        "INTERVENTION"
    ]
)


control = (
    pilot[
        pilot["group"]
        ==
        "CONTROL"
    ]
)


balance_rows = []


for column in BALANCE_COLUMNS:

    mean_t = (
        treated[
            column
        ]
        .mean()
    )


    mean_c = (
        control[
            column
        ]
        .mean()
    )


    smd = standardized_mean_difference(

        treated[
            column
        ],

        control[
            column
        ],
    )


    abs_smd = abs(
        smd
    )


    if abs_smd < 0.10:

        status = (
            "EXCELLENT"
        )


    elif abs_smd < MAX_ALLOWED_SMD:

        status = (
            "ACCEPTABLE"
        )


    else:

        status = (
            "REVIEW"
        )


    balance_rows.append(
        {
            "variable":
                column,

            "intervention_mean":
                mean_t,

            "control_mean":
                mean_c,

            "standardized_mean_difference":
                smd,

            "abs_smd":
                abs_smd,

            "balance_status":
                status,
        }
    )


balance = pd.DataFrame(
    balance_rows
)


# =========================================================
# 23. GROUP SUMMARY
# =========================================================

group_summary = (
    pilot

    .groupby(
        "group"
    )

    .agg(

        customers=(
            "customer_id",
            "count"
        ),

        avg_risk=(
            "risk_probability",
            "mean"
        ),

        revenue_3m=(
            "revenue_3m",
            "sum"
        ),

        revenue_6m=(
            "revenue_6m",
            "sum"
        ),

        revenue_12m=(
            "revenue_12m",
            "sum"
        ),

        avg_purchases_12m=(
            "purchases_12m",
            "mean"
        ),

        avg_recency_days=(
            "recency_days",
            "mean"
        ),
    )

    .reset_index()
)


# =========================================================
# 24. FINAL METRICS
# =========================================================

max_abs_smd = (
    balance[
        "abs_smd"
    ]
    .max()
)


mean_abs_smd = (
    balance[
        "abs_smd"
    ]
    .mean()
)


review_count = (
    balance[
        balance[
            "balance_status"
        ]
        ==
        "REVIEW"
    ]
    .shape[0]
)


max_revenue_ratio = (
    pairs[
        "revenue_ratio"
    ]
    .max()
)


max_risk_difference = (
    pairs[
        "risk_difference"
    ]
    .max()
)


max_match_distance = (
    pairs[
        "match_distance"
    ]
    .max()
)


# =========================================================
# 25. EXPERIMENT STATUS
# =========================================================

if (
    max_abs_smd < 0.10
    and
    max_revenue_ratio <= MAX_REVENUE_RATIO
    and
    max_risk_difference <= MAX_RISK_DIFFERENCE
    and
    max_match_distance <= MAX_MATCH_DISTANCE
):

    experiment_status = (
        "EXCELLENT"
    )


elif (
    max_abs_smd < MAX_ALLOWED_SMD
    and
    max_revenue_ratio <= MAX_REVENUE_RATIO
    and
    max_risk_difference <= MAX_RISK_DIFFERENCE
    and
    max_match_distance <= MAX_MATCH_DISTANCE
):

    experiment_status = (
        "READY"
    )


else:

    experiment_status = (
        "NOT_READY"
    )


# =========================================================
# 26. PRINT PAIRS
# =========================================================

print("\n" + "=" * 82)
print("25 FINAL RANDOMIZED MATCHED PAIRS")
print("=" * 82)


display_pairs = (
    pairs[
        [
            "pair_id",

            "intervention_customer_id",
            "control_customer_id",

            "intervention_risk_rank",
            "control_risk_rank",

            "intervention_risk",
            "control_risk",

            "risk_difference",

            "intervention_revenue_12m",
            "control_revenue_12m",

            "revenue_ratio",

            "match_distance",
            "match_quality",
        ]
    ]
    .copy()
)


pd.set_option(
    "display.max_rows",
    100
)

pd.set_option(
    "display.max_columns",
    50
)

pd.set_option(
    "display.width",
    280
)


print(
    display_pairs.to_string(
        index=False
    )
)


# =========================================================
# 27. EXCLUDED
# =========================================================

print("\n" + "=" * 82)
print("HIGH-RISK CLIENTS OUTSIDE THE EXPERIMENT")
print("=" * 82)


if excluded.empty:

    print(
        "\nNinguno."
    )

else:

    print(
        excluded.to_string(
            index=False
        )
    )


print(
    "\nIMPORTANTE:"
)

print(
    "Estar fuera del experimento NO significa "
    "que el cliente no pueda ser atendido comercialmente."
)

print(
    "Solo significa que no forma parte de "
    "la estimacion causal Intervention vs Control."
)


# =========================================================
# 28. SUMMARY
# =========================================================

print("\n" + "=" * 82)
print("BASELINE GROUP SUMMARY")
print("=" * 82)


print(
    group_summary.to_string(
        index=False
    )
)


# =========================================================
# 29. BALANCE
# =========================================================

print("\n" + "=" * 82)
print("RANDOMIZATION BALANCE")
print("=" * 82)


print(
    balance.to_string(
        index=False
    )
)


print(
    f"\nRandomizaciones evaluadas: "
    f"{N_RANDOMIZATIONS:,}"
)


print(
    f"Mejor randomizacion: "
    f"{best_assignment['randomization_id']}"
)


print(
    f"Max abs(SMD): "
    f"{max_abs_smd:.4f}"
)


print(
    f"Mean abs(SMD): "
    f"{mean_abs_smd:.4f}"
)


print(
    f"Variables REVIEW: "
    f"{review_count}"
)


print(
    f"Max revenue ratio: "
    f"{max_revenue_ratio:.2f}x"
)


print(
    f"Max risk difference: "
    f"{max_risk_difference:.4f}"
)


print(
    f"Max match distance: "
    f"{max_match_distance:.4f}"
)


# =========================================================
# 30. STATUS
# =========================================================

print("\n" + "=" * 82)
print("EXPERIMENT STATUS")
print("=" * 82)


print(
    f"\nStatus: "
    f"{experiment_status}"
)


if experiment_status == "EXCELLENT":

    print(
        "El diseño presenta un balance "
        "baseline excelente."
    )


elif experiment_status == "READY":

    print(
        "El diseño presenta un balance "
        "baseline suficientemente bueno "
        "para iniciar el piloto."
    )


else:

    print(
        "NO iniciar todavía el piloto."
    )


# =========================================================
# 31. SAVE
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


pilot_csv_path = (
    OUTPUT_DIR
    /
    "pilot_groups.csv"
)


pairs_csv_path = (
    OUTPUT_DIR
    /
    "pilot_pairs.csv"
)


balance_csv_path = (
    OUTPUT_DIR
    /
    "pilot_balance.csv"
)


summary_csv_path = (
    OUTPUT_DIR
    /
    "pilot_group_summary.csv"
)


excluded_csv_path = (
    OUTPUT_DIR
    /
    "pilot_excluded_clients.csv"
)


pilot.to_csv(
    pilot_csv_path,
    index=False
)


pairs.to_csv(
    pairs_csv_path,
    index=False
)


balance.to_csv(
    balance_csv_path,
    index=False
)


group_summary.to_csv(
    summary_csv_path,
    index=False
)


excluded.to_csv(
    excluded_csv_path,
    index=False
)


# =========================================================
# 32. PARQUET
# =========================================================

pilot_spark = (
    spark
    .createDataFrame(
        pilot
    )
)


(
    pilot_spark

    .write

    .mode(
        "overwrite"
    )

    .parquet(
        OUTPUT_PARQUET
    )
)


# =========================================================
# 33. PROTOCOL
# =========================================================

print("\n" + "=" * 82)
print("PILOT PROTOCOL")
print("=" * 82)


print(
    f"\nInicio propuesto: "
    f"{PILOT_START_DATE}"
)


print(
    f"Seguimiento: "
    f"{PILOT_DURATION_DAYS} dias"
)


print(
    "\nINTERVENTION:"
)

print(
    INTERVENTION_DEFINITION
)


print(
    "\nCONTROL:"
)

print(
    CONTROL_DEFINITION
)


print(
    "\nREGLAS:"
)


print(
    "1. No cambiar Intervention / Control "
    "despues de iniciar el piloto."
)


print(
    "2. Registrar cualquier intervencion "
    "accidental sobre un cliente CONTROL."
)


print(
    "3. Registrar fecha, canal, problema, "
    "accion comercial y resultado."
)


print(
    "4. Registrar coste aproximado "
    "de cada intervencion."
)


print(
    "5. Medir revenue de ambos grupos "
    "a 30, 60 y 90 dias."
)


print(
    "6. Metrica principal: "
    "Revenue Uplift Intervention vs Control."
)


print(
    "7. Metrica economica: "
    "ROI incremental."
)


# =========================================================
# 34. OUTPUTS
# =========================================================

print("\nARCHIVOS GENERADOS:")


print(
    pilot_csv_path
)


print(
    pairs_csv_path
)


print(
    balance_csv_path
)


print(
    summary_csv_path
)


print(
    excluded_csv_path
)


print(
    OUTPUT_PARQUET
)


# =========================================================
# 35. FINAL GUARD
# =========================================================

if experiment_status == "NOT_READY":

    print(
        "\nWARNING:"
    )

    print(
        "NO congelar estos grupos."
    )


else:

    print(
        "\nPILOTO LISTO PARA CONGELAR."
    )

    print(
        "Una vez empezadas las intervenciones, "
        "NO volver a ejecutar este script "
        "para cambiar las asignaciones."
    )


# =========================================================
# 36. END
# =========================================================

spark.stop()