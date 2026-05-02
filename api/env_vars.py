import os

MODEL_URI = os.getenv("MODEL_URI", "models:/churn-predictor@champion")
MLFLOW_SERVER = os.getenv(
    "MLFLOW_SERVER",
    os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
)
