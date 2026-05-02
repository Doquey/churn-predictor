from typing import Any

import mlflow
import pandas as pd

from ..env_vars import MLFLOW_SERVER, MODEL_URI
from .base import ChurnPredictorProtocol


class ChurnPredictor(ChurnPredictorProtocol):
    def __init__(
        self,
        model_uri: str = MODEL_URI,
        tracking_uri: str = MLFLOW_SERVER,
    ) -> None:
        self.model_uri = model_uri
        self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(self.tracking_uri)
        self._model = mlflow.pyfunc.load_model(self.model_uri)

    def _preprocess(self, payload: Any) -> pd.DataFrame:
        if isinstance(payload, pd.DataFrame):
            return payload

        if isinstance(payload, list):
            if not payload:
                raise ValueError("Prediction payload must not be empty.")
            return pd.DataFrame(payload)

        if isinstance(payload, dict):
            instances = payload.get("instances")
            if isinstance(instances, list):
                if not instances:
                    raise ValueError("Prediction payload instances must not be empty.")
                return pd.DataFrame(instances)
            return pd.DataFrame([payload])

        raise TypeError("Prediction payload must be a dict, list, or DataFrame.")

    def _postprocess(self, predictions: Any) -> dict[str, Any]:
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()

        if not isinstance(predictions, list):
            predictions = [predictions]

        return {"predictions": predictions}

    def predict(self, payload: Any) -> dict[str, Any]:
        features = self._preprocess(payload)
        predictions = self._model.predict(features)
        return self._postprocess(predictions)
