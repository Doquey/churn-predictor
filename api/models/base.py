from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ChurnPredictorProtocol(Protocol):
    def _preprocess(self, payload: Any) -> pd.DataFrame:
        ...

    def _postprocess(self, predictions: Any) -> dict[str, Any]:
        ...

    def predict(self, payload: Any) -> dict[str, Any]:
        ...
