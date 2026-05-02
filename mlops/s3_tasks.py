import datetime as dt
import hashlib
import hmac
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

if os.getenv("AIRFLOW_HOME"):
    try:
        from airflow.sdk import task
    except Exception:
        from airflow.decorators import task
else:
    def task(func=None, **_kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.getenv("CHURN_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
S3_BUCKET = os.getenv("S3_BUCKET", "churn-dataset")
S3_RAW_DATA_KEY = os.getenv("S3_RAW_DATA_KEY", RAW_DATA_FILENAME)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def _signing_key(secret_key: str, date_stamp: str, region: str) -> bytes:
    date_key = hmac.new(f"AWS4{secret_key}".encode(), date_stamp.encode(), hashlib.sha256).digest()
    region_key = hmac.new(date_key, region.encode(), hashlib.sha256).digest()
    service_key = hmac.new(region_key, b"s3", hashlib.sha256).digest()
    return hmac.new(service_key, b"aws4_request", hashlib.sha256).digest()


def _s3_url(endpoint_url: str, bucket: str, key: str | None = None) -> str:
    endpoint_url = endpoint_url.rstrip("/")
    bucket_path = quote(bucket, safe="")

    if key is None:
        return f"{endpoint_url}/{bucket_path}"

    object_path = "/".join(quote(part, safe="") for part in key.split("/"))
    return f"{endpoint_url}/{bucket_path}/{object_path}"


def _s3_request(
    method: str,
    url: str,
    body: bytes = b"",
    timeout: int = 60,
):
    parsed_url = urlsplit(url)
    now = dt.datetime.now(dt.UTC)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(body).hexdigest()

    headers = {
        "host": parsed_url.netloc,
        "x-amz-content-sha256": payload_hash,
        "x-amz-date": amz_date,
    }

    canonical_headers = "".join(f"{name}:{headers[name]}\n" for name in sorted(headers))
    signed_headers = ";".join(sorted(headers))
    canonical_request = "\n".join(
        [
            method,
            parsed_url.path or "/",
            parsed_url.query,
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )

    credential_scope = f"{date_stamp}/{AWS_DEFAULT_REGION}/s3/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )
    signature = hmac.new(
        _signing_key(AWS_SECRET_ACCESS_KEY, date_stamp, AWS_DEFAULT_REGION),
        string_to_sign.encode(),
        hashlib.sha256,
    ).hexdigest()

    headers["authorization"] = (
        "AWS4-HMAC-SHA256 "
        f"Credential={AWS_ACCESS_KEY_ID}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, "
        f"Signature={signature}"
    )

    data = body if method not in {"GET", "HEAD"} else None
    request = Request(url, data=data, headers=headers, method=method)
    return urlopen(request, timeout=timeout)


def ensure_bucket_exists(
    bucket: str = S3_BUCKET,
    endpoint_url: str = S3_ENDPOINT_URL,
) -> None:
    bucket_url = _s3_url(endpoint_url, bucket)

    try:
        with _s3_request("HEAD", bucket_url):
            return
    except HTTPError as exc:
        if exc.code != 404:
            raise

    with _s3_request("PUT", bucket_url):
        return


def upload_file_to_s3(
    local_path: Path | str,
    bucket: str = S3_BUCKET,
    key: str = S3_RAW_DATA_KEY,
    endpoint_url: str = S3_ENDPOINT_URL,
) -> str:
    ensure_bucket_exists(bucket=bucket, endpoint_url=endpoint_url)

    local_path = Path(local_path)
    object_url = _s3_url(endpoint_url, bucket, key)

    with _s3_request("PUT", object_url, body=local_path.read_bytes()):
        pass

    return f"s3://{bucket}/{key}"


def download_file_from_s3(
    bucket: str = S3_BUCKET,
    key: str = S3_RAW_DATA_KEY,
    destination_path: Path | str | None = None,
    endpoint_url: str = S3_ENDPOINT_URL,
) -> str:
    if destination_path is None:
        destination_path = DATA_DIR / Path(key).name

    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    object_url = _s3_url(endpoint_url, bucket, key)
    with _s3_request("GET", object_url) as response:
        destination_path.write_bytes(response.read())

    return str(destination_path)


@task()
def fetch_data_from_s3() -> str:
    return download_file_from_s3()
