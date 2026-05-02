from pathlib import Path

import pandas as pd


BINARY_COLUMNS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "gender"]
MULTICLASS_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]
TARGET_COLUMN = "Churn"
DATA_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_X_PATH = DATA_DIR / "X.csv"
DEFAULT_Y_PATH = DATA_DIR / "y.csv"
FEATURE_COLUMNS = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PaperlessBilling",
    "MonthlyCharges",
    "TotalCharges",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "DeviceProtection_No internet service",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingMovies_No internet service",
    "Contract_One year",
    "Contract_Two year",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
]
RAW_INPUT_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def map_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in BINARY_COLUMNS:
        if column not in df.columns:
            continue

        if column == "gender":
            df[column] = df[column].map({"Male": 1, "Female": 0})
            continue

        df[column] = df[column].map({"Yes": 1, "No": 0})

    return df


def map_multiclass_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in MULTICLASS_COLUMNS if column in df.columns]
    return pd.get_dummies(
        df,
        columns=columns,
        drop_first=True,
    )


def map_target_column(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    return df


def validate_raw_customer_columns(df: pd.DataFrame) -> None:
    missing_columns = [column for column in RAW_INPUT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Uploaded CSV is missing required columns: " + ", ".join(missing_columns)
        )


def transform_raw_customers(df: pd.DataFrame) -> pd.DataFrame:
    validate_raw_customer_columns(df)

    raw = df.copy()
    features = pd.DataFrame(index=raw.index)

    features["SeniorCitizen"] = pd.to_numeric(
        raw["SeniorCitizen"],
        errors="coerce",
    ).fillna(0).astype(int)
    features["Partner"] = raw["Partner"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    features["Dependents"] = raw["Dependents"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    features["tenure"] = pd.to_numeric(raw["tenure"], errors="coerce").fillna(0)
    features["PaperlessBilling"] = (
        raw["PaperlessBilling"].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    )
    features["MonthlyCharges"] = pd.to_numeric(
        raw["MonthlyCharges"],
        errors="coerce",
    ).fillna(0)

    total_charges = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    features["TotalCharges"] = total_charges.fillna(total_charges.median()).fillna(0)

    features["InternetService_Fiber optic"] = (raw["InternetService"] == "Fiber optic").astype(int)
    features["InternetService_No"] = (raw["InternetService"] == "No").astype(int)
    features["OnlineSecurity_No internet service"] = (
        raw["OnlineSecurity"] == "No internet service"
    ).astype(int)
    features["OnlineSecurity_Yes"] = (raw["OnlineSecurity"] == "Yes").astype(int)
    features["OnlineBackup_No internet service"] = (
        raw["OnlineBackup"] == "No internet service"
    ).astype(int)
    features["DeviceProtection_No internet service"] = (
        raw["DeviceProtection"] == "No internet service"
    ).astype(int)
    features["TechSupport_No internet service"] = (
        raw["TechSupport"] == "No internet service"
    ).astype(int)
    features["TechSupport_Yes"] = (raw["TechSupport"] == "Yes").astype(int)
    features["StreamingTV_No internet service"] = (
        raw["StreamingTV"] == "No internet service"
    ).astype(int)
    features["StreamingMovies_No internet service"] = (
        raw["StreamingMovies"] == "No internet service"
    ).astype(int)
    features["Contract_One year"] = (raw["Contract"] == "One year").astype(int)
    features["Contract_Two year"] = (raw["Contract"] == "Two year").astype(int)
    features["PaymentMethod_Credit card (automatic)"] = (
        raw["PaymentMethod"] == "Credit card (automatic)"
    ).astype(int)
    features["PaymentMethod_Electronic check"] = (
        raw["PaymentMethod"] == "Electronic check"
    ).astype(int)

    return features[FEATURE_COLUMNS]


def run_preprocessing(
    input_path: Path | str = DEFAULT_RAW_DATA_PATH,
    output_dir: Path | str = DATA_DIR,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(input_path)
    validate_raw_customer_columns(dataset)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Training data must include the {TARGET_COLUMN!r} column.")

    y = map_target_column(dataset.copy())[TARGET_COLUMN]
    X = transform_raw_customers(dataset)

    x_path = output_dir / DEFAULT_X_PATH.name
    y_path = output_dir / DEFAULT_Y_PATH.name

    X.to_csv(x_path, index=False)
    y.to_csv(y_path, index=False)

    return {
        "x_path": str(x_path),
        "y_path": str(y_path),
    }


def main():
    run_preprocessing()


if __name__ == "__main__":
    main()
