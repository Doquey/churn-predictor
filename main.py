from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from data.preprocessing import FEATURE_COLUMNS, transform_raw_customers

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8001/predict")
SAMPLE_UPLOAD_PATH = Path(__file__).resolve().parent / "data" / "churn_upload_sample.csv"


def _blank_features() -> dict[str, int | float]:
    return {feature: 0 for feature in FEATURE_COLUMNS}


def build_features(
    senior_citizen: bool,
    partner: bool,
    dependents: bool,
    tenure: int,
    paperless_billing: bool,
    monthly_charges: float,
    total_charges: float,
    internet_service: str,
    online_security: str,
    online_backup: str,
    device_protection: str,
    tech_support: str,
    streaming_tv: str,
    streaming_movies: str,
    contract: str,
    payment_method: str,
) -> dict[str, int | float]:
    features = _blank_features()

    features["SeniorCitizen"] = int(senior_citizen)
    features["Partner"] = int(partner)
    features["Dependents"] = int(dependents)
    features["tenure"] = tenure
    features["PaperlessBilling"] = int(paperless_billing)
    features["MonthlyCharges"] = monthly_charges
    features["TotalCharges"] = total_charges

    features["InternetService_Fiber optic"] = int(internet_service == "Fiber optic")
    features["InternetService_No"] = int(internet_service == "No")
    features["OnlineSecurity_No internet service"] = int(online_security == "No internet service")
    features["OnlineSecurity_Yes"] = int(online_security == "Yes")
    features["OnlineBackup_No internet service"] = int(online_backup == "No internet service")
    features["DeviceProtection_No internet service"] = int(device_protection == "No internet service")
    features["TechSupport_No internet service"] = int(tech_support == "No internet service")
    features["TechSupport_Yes"] = int(tech_support == "Yes")
    features["StreamingTV_No internet service"] = int(streaming_tv == "No internet service")
    features["StreamingMovies_No internet service"] = int(streaming_movies == "No internet service")
    features["Contract_One year"] = int(contract == "One year")
    features["Contract_Two year"] = int(contract == "Two year")
    features["PaymentMethod_Credit card (automatic)"] = int(
        payment_method == "Credit card (automatic)"
    )
    features["PaymentMethod_Electronic check"] = int(payment_method == "Electronic check")

    return features


def request_prediction(api_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    import requests

    try:
        response = requests.post(api_url, json=payload, timeout=30)
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise RuntimeError(
            f"Could not connect to the API at {api_url}. "
            "Start the local stack with `./run_local.sh` or point the sidebar endpoint "
            "at a running API."
        ) from exc
    except requests.HTTPError as exc:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        raise RuntimeError(f"API returned {response.status_code}: {detail}") from exc

    return response.json()


def render_response(response: dict[str, Any]) -> None:
    import streamlit as st

    st.subheader("Response")
    st.json(response)


def main():
    try:
        import requests
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise SystemExit("Install dependencies with `uv sync` before running Streamlit.") from exc

    st.set_page_config(page_title="Churn Predictor", page_icon=None, layout="wide")
    st.title("Churn Predictor")

    api_url = st.sidebar.text_input("API endpoint", value=DEFAULT_API_URL)

    upload_tab, form_tab, json_tab = st.tabs(["CSV Upload", "Form", "JSON"])

    with upload_tab:
        if SAMPLE_UPLOAD_PATH.exists():
            st.download_button(
                "Download sample CSV",
                data=SAMPLE_UPLOAD_PATH.read_bytes(),
                file_name=SAMPLE_UPLOAD_PATH.name,
                mime="text/csv",
            )

        uploaded_file = st.file_uploader("Raw Telco customer CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                raw_customers = pd.read_csv(uploaded_file)
                features_df = transform_raw_customers(raw_customers)
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(str(exc))
            else:
                st.subheader("Uploaded Rows")
                st.dataframe(raw_customers.head(20), use_container_width=True)

                payload = {"instances": features_df.to_dict(orient="records")}
                st.subheader("Request Preview")
                st.json({"instances": payload["instances"][:3]})

                if st.button("Predict CSV"):
                    try:
                        response = request_prediction(api_url, payload)
                    except (requests.RequestException, RuntimeError) as exc:
                        st.error(str(exc))
                    else:
                        predictions = response.get("predictions")
                        if isinstance(predictions, list) and len(predictions) == len(raw_customers):
                            results = raw_customers.copy()
                            results["Predicted Churn"] = [
                                "Yes" if p == 1 else "No" for p in predictions
                            ]

                            st.subheader("Predictions")
                            has_ground_truth = "Churn" in results.columns

                            if has_ground_truth:
                                def _style_row(row):
                                    match = (
                                        str(row["Churn"]).strip().lower()
                                        == str(row["Predicted Churn"]).strip().lower()
                                    )
                                    cell = (
                                        "background-color: #c3e6cb; color: #155724"
                                        if match else
                                        "background-color: #f5c6cb; color: #721c24"
                                    )
                                    return [
                                        cell if col == "Predicted Churn" else ""
                                        for col in row.index
                                    ]

                                st.dataframe(
                                    results.style.apply(_style_row, axis=1),
                                    use_container_width=True,
                                )
                            else:
                                st.dataframe(results, use_container_width=True)

                            st.download_button(
                                "Download predictions CSV",
                                data=results.to_csv(index=False).encode("utf-8"),
                                file_name="churn_predictions.csv",
                                mime="text/csv",
                            )

    with form_tab:
        with st.form("churn_prediction_form"):
            customer_col, service_col, billing_col = st.columns(3)

            with customer_col:
                senior_citizen = st.checkbox("Senior citizen")
                partner = st.checkbox("Partner")
                dependents = st.checkbox("Dependents")
                tenure = st.number_input("Tenure", min_value=0, max_value=120, value=12, step=1)

            with service_col:
                internet_service = st.selectbox(
                    "Internet service",
                    ["DSL", "Fiber optic", "No"],
                )
                online_security = st.selectbox(
                    "Online security",
                    ["No", "Yes", "No internet service"],
                )
                online_backup = st.selectbox(
                    "Online backup",
                    ["No", "Yes", "No internet service"],
                )
                device_protection = st.selectbox(
                    "Device protection",
                    ["No", "Yes", "No internet service"],
                )
                tech_support = st.selectbox(
                    "Tech support",
                    ["No", "Yes", "No internet service"],
                )
                streaming_tv = st.selectbox(
                    "Streaming TV",
                    ["No", "Yes", "No internet service"],
                )
                streaming_movies = st.selectbox(
                    "Streaming movies",
                    ["No", "Yes", "No internet service"],
                )

            with billing_col:
                paperless_billing = st.checkbox("Paperless billing", value=True)
                monthly_charges = st.number_input(
                    "Monthly charges",
                    min_value=0.0,
                    value=70.0,
                    step=1.0,
                )
                total_charges = st.number_input(
                    "Total charges",
                    min_value=0.0,
                    value=850.0,
                    step=10.0,
                )
                contract = st.selectbox(
                    "Contract",
                    ["Month-to-month", "One year", "Two year"],
                )
                payment_method = st.selectbox(
                    "Payment method",
                    [
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                        "Electronic check",
                        "Mailed check",
                    ],
                )

            submitted = st.form_submit_button("Predict")

        if submitted:
            features = build_features(
                senior_citizen=senior_citizen,
                partner=partner,
                dependents=dependents,
                tenure=int(tenure),
                paperless_billing=paperless_billing,
                monthly_charges=float(monthly_charges),
                total_charges=float(total_charges),
                internet_service=internet_service,
                online_security=online_security,
                online_backup=online_backup,
                device_protection=device_protection,
                tech_support=tech_support,
                streaming_tv=streaming_tv,
                streaming_movies=streaming_movies,
                contract=contract,
                payment_method=payment_method,
            )
            payload = {"instances": [features]}

            st.subheader("Request")
            st.json(payload)

            try:
                render_response(request_prediction(api_url, payload))
            except (requests.RequestException, RuntimeError) as exc:
                st.error(str(exc))

    with json_tab:
        default_payload = {"instances": [_blank_features()]}
        payload_text = st.text_area(
            "Payload",
            value=json.dumps(default_payload, indent=2),
            height=420,
        )

        if st.button("Send JSON"):
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
            else:
                try:
                    render_response(request_prediction(api_url, payload))
                except (requests.RequestException, RuntimeError) as exc:
                    st.error(str(exc))


if __name__ == "__main__":
    main()
