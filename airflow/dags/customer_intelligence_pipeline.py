# ============================================================
# retailco CUSTOMER INTELLIGENCE - AIRFLOW DAG
#
# Airflow = orchestration layer
# Databricks = Spark / ML execution layer
#
# Production state:
# - Quarterly Airflow schedule after each expected S3 delivery.
# - No Databricks schedule: Airflow is the only orchestrator.
# - No credentials hardcoded.
# ============================================================

from datetime import timedelta

import pendulum

from airflow.sdk import DAG
from airflow.providers.databricks.operators.databricks import (
    DatabricksRunNowOperator,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DAG_ID = "retailco_customer_intelligence_pipeline"
DATABRICKS_CONN_ID = "databricks_default"

# Stable Databricks production Job created and validated on
# 2026-08-15. Using the numeric ID avoids ambiguous name lookup.
DATABRICKS_JOB_ID = 592515915108517


# ============================================================
# 2. DEFAULT TASK SETTINGS
# ============================================================

DEFAULT_ARGS = {
    "owner": "retailco-data",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


# ============================================================
# 3. DAG
# ============================================================

with DAG(
    dag_id=DAG_ID,
    description=(
        "Orchestrates the retailco customer intelligence "
        "production pipeline in Databricks."
    ),
    default_args=DEFAULT_ARGS,
    start_date=pendulum.datetime(
        2026,
        8,
        15,
        tz="Europe/Madrid",
    ),
    # Quarterly cadence: February, May, August and November,
    # on day 11 at 06:00 Europe/Madrid. Expected S3 files should
    # be uploaded before this time. Airflow is the only scheduler.
    schedule="0 6 11 2,5,8,11 *",
    catchup=False,
    max_active_runs=1,
    tags=[
        "retailco",
        "customer-intelligence",
        "databricks",
        "ml",
        "production",
    ],
) as dag:

    run_databricks_production_pipeline = DatabricksRunNowOperator(
        task_id="run_databricks_production_pipeline",
        databricks_conn_id=DATABRICKS_CONN_ID,
        job_id=DATABRICKS_JOB_ID,
        notebook_params={
            "snapshot_override": "",
            "publish_outputs": "true",
            "model_alias": "champion",
            "pipeline_run_id": "{{ run_id }}",
        },
        wait_for_termination=True,
        polling_period_seconds=30,
        # Retries for temporary Databricks API connectivity issues.
        databricks_retry_limit=3,
        databricks_retry_delay=5,
        # Store the Databricks run ID and run URL in Airflow XCom.
        do_xcom_push=True,
        # Never cancel another valid production run silently.
        cancel_previous_runs=False,
        execution_timeout=timedelta(hours=2),
    )
