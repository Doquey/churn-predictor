import http.client
import os
import socket as _socket
import sys
from pathlib import Path
from typing import Any

import pendulum

try:
    from airflow.sdk import dag, task
except ImportError:
    from airflow.decorators import dag, task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("CHURN_DATA_DIR", PROJECT_ROOT / "data"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlops.s3_tasks import fetch_data_from_s3

_DOCKER_SOCKET = "/var/run/docker.sock"


class _UnixSocketHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection routed through the Docker Unix socket."""

    def connect(self):
        self.sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        self.sock.connect(_DOCKER_SOCKET)


@task()
def preprocess_training_data(raw_data_path: str) -> dict[str, str]:
    from data.preprocessing import run_preprocessing

    return run_preprocessing(
        input_path=raw_data_path,
        output_dir=DATA_DIR,
    )


@task()
def train_churn_model(data_paths: dict[str, str]) -> dict[str, Any]:
    from src.training import run_training

    return run_training(
        x_path=data_paths["x_path"],
        y_path=data_paths["y_path"],
    )


@task()
def restart_api_if_promoted(training_summary: dict[str, Any]) -> None:
    """Restart the FastAPI container when a new champion model is promoted."""
    if not training_summary.get("promoted_to_champion"):
        print(
            f"Model version {training_summary.get('promoted_model_version', 'N/A')} "
            "did not meet promotion gates — API container restart skipped."
        )
        return

    container_name = os.getenv("API_CONTAINER_NAME", "churn_api")
    version = training_summary.get("promoted_model_version", "unknown")

    conn = _UnixSocketHTTPConnection("localhost")
    # t=15: give the container 15 s to shut down gracefully before SIGKILL
    conn.request("POST", f"/containers/{container_name}/restart?t=15")
    response = conn.getresponse()

    if response.status in (200, 204):
        print(
            f"Container '{container_name}' restarted — "
            f"champion v{version} will be loaded on next request."
        )
    else:
        body = response.read().decode(errors="replace")
        raise RuntimeError(
            f"Docker daemon returned HTTP {response.status} when restarting "
            f"'{container_name}': {body}"
        )


@dag(
    dag_id="c_training_pipeline",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 5, 2, tz="UTC"),
    catchup=False,
    tags=["churn", "training", "mlflow"],
)
def c_training_pipeline():
    raw_data_path = fetch_data_from_s3()
    prepared_data_paths = preprocess_training_data(raw_data_path)
    training_summary = train_churn_model(prepared_data_paths)
    restart_api_if_promoted(training_summary)


c_training_dag = c_training_pipeline()
