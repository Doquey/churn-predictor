import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from xgboost import XGBClassifier

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    os.getenv("MLFLOW_SERVER", "http://127.0.0.1:5000"),
)
EXPERIMENT_NAME = "churn-predictor"
REGISTERED_MODEL_NAME = "churn-predictor"
CHAMPION_ALIAS = "champion"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_X_PATH = DATA_DIR / "X.csv"
DEFAULT_Y_PATH = DATA_DIR / "y.csv"


def should_promote_candidate(
    candidate_roc_auc: float,
    candidate_recall: float,
    candidate_f1: float,
    champion_metrics: dict[str, float] | None = None,
) -> bool:
    minimum_gate = (
        candidate_roc_auc >= 0.84
        and candidate_recall >= 0.65
        and candidate_f1 >= 0.6
    )

    if not minimum_gate:
        return False

    if champion_metrics is None:
        return True

    champion_roc_auc = champion_metrics.get("roc_auc", 0)
    champion_recall = champion_metrics.get("recall", 0)
    champion_f1 = champion_metrics.get("f1", 0)

    return (
        candidate_roc_auc > champion_roc_auc
        and candidate_recall > champion_recall
        and candidate_f1 > champion_f1
    )


def get_champion_metrics(model_name: str, alias: str = CHAMPION_ALIAS) -> dict | None:
    client = mlflow.tracking.MlflowClient()

    try:
        model_version = client.get_model_version_by_alias(
            name=model_name,
            alias=alias,
        )
        run_id = model_version.run_id
        metrics = client.get_run(run_id).data.metrics
        return metrics

    except Exception as e:
        print(f"No champion model found: {e}")
        return None


def get_model_version_for_run(model_name: str, run_id: str):
    client = mlflow.tracking.MlflowClient()

    versions = client.search_model_versions(f"name = '{model_name}'")

    matching_versions = [
        version for version in versions
        if version.run_id == run_id
    ]

    if not matching_versions:
        raise ValueError(f"No model version found for run_id={run_id}")

    return max(matching_versions, key=lambda version: int(version.version))


def promote_model_to_champion(model_name: str, run_id: str):
    client = mlflow.tracking.MlflowClient()

    candidate_version = get_model_version_for_run(
        model_name=model_name,
        run_id=run_id,
    )

    client.set_registered_model_alias(
        name=model_name,
        alias=CHAMPION_ALIAS,
        version=candidate_version.version,
    )

    client.set_model_version_tag(
        name=model_name,
        version=candidate_version.version,
        key="promotion_status",
        value="champion",
    )

    print(
        f"Promoted {model_name} version {candidate_version.version} "
        f"to @{CHAMPION_ALIAS}"
    )

    return candidate_version.version


def load_training_data(
    x_path: Path | str = DEFAULT_X_PATH,
    y_path: Path | str = DEFAULT_Y_PATH,
) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).squeeze()
    return X, y


def run_training(
    x_path: Path | str = DEFAULT_X_PATH,
    y_path: Path | str = DEFAULT_Y_PATH,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_training_data(x_path=x_path, y_path=y_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    threshold = 0.40

    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
    }

    param_grid = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 4],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    with mlflow.start_run(run_name="xgboost_churn_gridsearch") as run:
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])
        mlflow.log_param("num_features", X.shape[1])
        mlflow.log_param("scale_pos_weight", scale_pos_weight)

        # Log the grid itself as a JSON artifact/param-ish record.
        mlflow.log_dict(param_grid, "gridsearch/param_grid.json")

        base_model = XGBClassifier(**base_params)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            verbose=2,
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

        mlflow.log_params({f"best_{key}": value for key, value in best_params.items()})
        mlflow.log_metric("best_cv_roc_auc", best_cv_score)

        # Evaluate on holdout test set
        y_proba = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        roc_auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            report_path = tmpdir / "classification_report.json"
            cm_path = tmpdir / "confusion_matrix.json"
            threshold_path = tmpdir / "threshold.json"
            schema_path = tmpdir / "feature_schema.json"
            best_params_path = tmpdir / "best_params.json"

            report_path.write_text(json.dumps(report, indent=2))
            cm_path.write_text(json.dumps(cm.tolist(), indent=2))
            threshold_path.write_text(json.dumps({"threshold": threshold}, indent=2))
            schema_path.write_text(json.dumps({"columns": X.columns.tolist()}, indent=2))
            best_params_path.write_text(json.dumps(best_params, indent=2))

            mlflow.log_artifact(str(report_path), artifact_path="reports")
            mlflow.log_artifact(str(cm_path), artifact_path="reports")
            mlflow.log_artifact(str(threshold_path), artifact_path="config")
            mlflow.log_artifact(str(schema_path), artifact_path="schema")
            mlflow.log_artifact(str(best_params_path), artifact_path="gridsearch")

        model_info = mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_test.head(5),
        )

        champion_metrics = get_champion_metrics(
            REGISTERED_MODEL_NAME,
            alias=CHAMPION_ALIAS,
        )

        promote = should_promote_candidate(
            candidate_roc_auc=roc_auc,
            candidate_recall=recall,
            candidate_f1=f1,
            champion_metrics=champion_metrics,
        )

        mlflow.log_param("promoted_to_champion", promote)

        promoted_version = None
        if promote:
            promoted_version = promote_model_to_champion(
                model_name=REGISTERED_MODEL_NAME,
                run_id=run.info.run_id,
            )
            mlflow.log_param("promoted_model_version", promoted_version)
        else:
            print("Candidate model was not promoted.")

        training_summary = {
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "promoted_to_champion": bool(promote),
            "promoted_model_version": str(promoted_version) if promoted_version else None,
            "roc_auc": float(roc_auc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        print("MLflow run ID:", run.info.run_id)
        print("Logged model URI:", model_info.model_uri)
        return training_summary


def main():
    run_training()


if __name__ == "__main__":
    main()
