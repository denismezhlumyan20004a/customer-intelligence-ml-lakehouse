FROM apache/spark:4.1.3-python3

USER root

RUN python3 -m pip install --no-cache-dir numpy pandas pyarrow scikit-learn xgboost

USER spark