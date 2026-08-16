# Repository cleanup plan

This plan separates the validated production route from modelling history without deleting useful evidence.

## Keep in the main repository

- `airflow/dags/customer_intelligence_pipeline.py`
- `airflow/docker-compose.yaml`
- `airflow/Dockerfile`
- `airflow/requirements.txt`
- `config/best_xgboost_params.json`
- `notebooks/retailco_customer_intelligence_production_job.py`
- `src/build_silver.py`
- `src/build_gold_fact_sales.py`
- `src/build_customer_monthly.py`
- `src/build_churn_target_v3.py`
- `src/build_ml_features.py`
- `src/build_ml_features_v2.py`
- `src/prepare_temporal_split_v2.py`
- `src/evaluate_model_ranking_v2.py`
- `src/evaluate_champion_test.py`
- `src/train_production_model.py`
- `README.md` and `docs/`

## Preserve as modelling history

The earlier scripts document iteration and are useful evidence, but they should not look like competing production paths. Move them later to `archive/experiments/` in a dedicated commit:

- `build_churn_target.py`
- `build_churn_target_v2.py`
- `prepare_temporal_split.py`
- `evaluate_model_ranking.py`
- V1 model-training scripts
- tuned and exploratory XGBoost scripts
- the old Airflow DAG backup

Do not archive anything until the new documentation is committed and the production DAG has been revalidated.

## Keep out of Git

- `.venv/`
- `data/`
- `models/`
- `outputs/`
- `airflow/.env`
- `airflow/logs/`
- Spark `.crc` and `_SUCCESS` files
- Python caches and test caches
- customer-level exports

The registered MLflow model is the production source of truth; committed local Spark model folders are not.

## Safe cleanup order

1. Copy the delivered README, docs, notebook template and `.gitignore` into the repository.
2. Confirm that `airflow/.env`, `data/`, `models/` and `outputs/` are not tracked.
3. Run Python compilation and Airflow DAG import checks.
4. Commit the documentation as a standalone commit.
5. Create `archive/experiments/` and move legacy scripts with `git mv`.
6. Re-run the checks and commit the reorganisation separately.
7. Never use a destructive reset to perform this cleanup.

## Recommended verification commands

```powershell
git status --short
git ls-files airflow/.env data models outputs
python -m compileall -q src airflow/dags notebooks
Set-Location airflow
docker compose exec airflow-dag-processor airflow dags list-import-errors
docker compose ps
```

The `git ls-files` command should return no sensitive or generated paths.
