#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${ROOT_DIR}/dev.yml"
AIRFLOW_HOME_DIR="${ROOT_DIR}/.airflow"
AIRFLOW_LOG_DIR="${AIRFLOW_HOME_DIR}/logs"
STREAMLIT_PID_FILE="${AIRFLOW_HOME_DIR}/streamlit.pid"
STREAMLIT_LOG_FILE="${AIRFLOW_LOG_DIR}/streamlit.log"

API_URL="${API_URL:-http://localhost:8001/predict}"

DOCKER=()

detect_docker() {
  if docker info >/dev/null 2>&1; then
    DOCKER=(docker)
    return
  fi

  if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
    DOCKER=(sudo docker)
    return
  fi

  echo "ERROR: Docker is not reachable. Start Docker or add your user to the docker group." >&2
  exit 1
}

compose() {
  "${DOCKER[@]}" compose -f "${COMPOSE_FILE}" "$@"
}

is_running() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" >/dev/null 2>&1
}

stop_pid() {
  local pid_file="$1"
  local name="$2"

  if is_running "${pid_file}"; then
    echo "==> Stopping ${name}..."
    kill "$(cat "${pid_file}")" >/dev/null 2>&1 || true
  fi

  rm -f "${pid_file}"
}

export_airflow_env() {
  export AIRFLOW_HOME="${AIRFLOW_HOME_DIR}"
  export AIRFLOW__CORE__EXECUTOR=LocalExecutor
  export AIRFLOW__CORE__LOAD_EXAMPLES=false
  export AIRFLOW__CORE__DAGS_FOLDER="${ROOT_DIR}/mlops"
  export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://mlops:mlops@localhost:5432/mlops"
  export AIRFLOW__WEBSERVER__SECRET_KEY=dev_secret_key
  export AIRFLOW__LOGGING__BASE_LOG_FOLDER="${ROOT_DIR}/logs"

  export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
  export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
  export AWS_ACCESS_KEY_ID=minioadmin
  export AWS_SECRET_ACCESS_KEY=minioadmin
  export AWS_DEFAULT_REGION=us-east-1
  export S3_ENDPOINT_URL=http://localhost:9000
  export S3_BUCKET=churn-dataset
  export S3_RAW_DATA_KEY=WA_Fn-UseC_-Telco-Customer-Churn.csv
  export CHURN_DATA_DIR="${ROOT_DIR}/data"
  export PYTHONPATH="${ROOT_DIR}"
  export API_CONTAINER_NAME=churn_api
}

seed_minio() {
  local csv_path="${ROOT_DIR}/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

  if [[ ! -f "${csv_path}" ]]; then
    echo "ERROR: ${csv_path} not found." >&2
    exit 1
  fi

  echo "==> Seeding MinIO bucket churn-dataset..."
  "${DOCKER[@]}" run --rm \
    --network host \
    --entrypoint sh \
    -v "${ROOT_DIR}/data:/data" \
    minio/mc \
    -c "
      mc alias set local http://localhost:9000 minioadmin minioadmin &&
      mc mb --ignore-existing local/churn-dataset &&
      mc cp /data/WA_Fn-UseC_-Telco-Customer-Churn.csv local/churn-dataset/
    "
}

start_airflow() {
  mkdir -p "${AIRFLOW_HOME_DIR}" "${AIRFLOW_LOG_DIR}" "${ROOT_DIR}/logs"
  export_airflow_env

  echo "==> Migrating Airflow DB..."
  uv run airflow db migrate

  echo "==> Ensuring Airflow admin user exists..."
  uv run airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com >/dev/null 2>&1 || true

  if ! is_running "${AIRFLOW_HOME_DIR}/api-server.pid"; then
    echo "==> Starting Airflow API server on :8080..."
    nohup uv run airflow api-server --port 8080 \
      > "${AIRFLOW_LOG_DIR}/api-server.log" 2>&1 &
    echo $! > "${AIRFLOW_HOME_DIR}/api-server.pid"
  fi

  if ! is_running "${AIRFLOW_HOME_DIR}/scheduler.pid"; then
    echo "==> Starting Airflow scheduler..."
    nohup uv run airflow scheduler \
      > "${AIRFLOW_LOG_DIR}/scheduler.log" 2>&1 &
    echo $! > "${AIRFLOW_HOME_DIR}/scheduler.pid"
  fi

  if ! is_running "${AIRFLOW_HOME_DIR}/dag-processor.pid"; then
    echo "==> Starting Airflow DAG processor..."
    nohup uv run airflow dag-processor \
      > "${AIRFLOW_LOG_DIR}/dag-processor.log" 2>&1 &
    echo $! > "${AIRFLOW_HOME_DIR}/dag-processor.pid"
  fi
}

start_streamlit() {
  mkdir -p "${AIRFLOW_HOME_DIR}" "${AIRFLOW_LOG_DIR}"

  if is_running "${STREAMLIT_PID_FILE}"; then
    echo "==> Streamlit already running."
    return
  fi

  echo "==> Starting Streamlit on :8501..."
  API_URL="${API_URL}" nohup uv run streamlit run "${ROOT_DIR}/main.py" \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true \
    > "${STREAMLIT_LOG_FILE}" 2>&1 &
  echo $! > "${STREAMLIT_PID_FILE}"
}

start() {
  detect_docker
  mkdir -p "${AIRFLOW_HOME_DIR}" "${AIRFLOW_LOG_DIR}"

  echo "==> Installing local Python dependencies..."
  uv sync

  echo "==> Starting Docker services..."
  compose up -d --build postgres minio mlflow api

  seed_minio
  start_airflow
  start_streamlit

  echo ""
  echo "Stack is running:"
  echo "  Streamlit      http://localhost:8501"
  echo "  FastAPI        http://localhost:8001/docs"
  echo "  API health     http://localhost:8001/health"
  echo "  Airflow        http://localhost:8080  (admin / admin)"
  echo "  MLflow         http://localhost:5000"
  echo "  MinIO Console  http://localhost:9001  (minioadmin / minioadmin)"
  echo ""
  echo "Logs:"
  echo "  Streamlit      ${STREAMLIT_LOG_FILE}"
  echo "  Airflow        ${AIRFLOW_LOG_DIR}"
}

stop() {
  detect_docker
  export_airflow_env

  stop_pid "${STREAMLIT_PID_FILE}" "Streamlit"
  stop_pid "${AIRFLOW_HOME_DIR}/api-server.pid" "Airflow API server"
  stop_pid "${AIRFLOW_HOME_DIR}/scheduler.pid" "Airflow scheduler"
  stop_pid "${AIRFLOW_HOME_DIR}/dag-processor.pid" "Airflow DAG processor"
  stop_pid "${AIRFLOW_HOME_DIR}/webserver.pid" "Airflow webserver"

  echo "==> Stopping Docker services..."
  compose down
}

status() {
  detect_docker

  echo "Docker services:"
  compose ps
  echo ""
  echo "Local processes:"
  for item in \
    "${STREAMLIT_PID_FILE}:Streamlit" \
    "${AIRFLOW_HOME_DIR}/api-server.pid:Airflow API server" \
    "${AIRFLOW_HOME_DIR}/scheduler.pid:Airflow scheduler" \
    "${AIRFLOW_HOME_DIR}/dag-processor.pid:Airflow DAG processor"; do
    local pid_file="${item%%:*}"
    local name="${item#*:}"
    if is_running "${pid_file}"; then
      echo "  ${name}: running (pid $(cat "${pid_file}"))"
    else
      echo "  ${name}: stopped"
    fi
  done
}

case "${1:-start}" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  restart)
    stop
    start
    ;;
  status)
    status
    ;;
  *)
    echo "Usage: $0 [start|stop|restart|status]" >&2
    exit 1
    ;;
esac
