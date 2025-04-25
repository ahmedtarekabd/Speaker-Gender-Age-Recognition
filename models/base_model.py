import mlflow
from typing import Any, Dict
from numpy import ndarray
from sklearn.base import BaseEstimator
from modules.evaluate import PerformanceAnalyzer

# === Base Model Interface ===
class BaseModel:
    def __init__(self) -> None:
        self.model: BaseEstimator = None
        self.best_params: Dict[str, Any] = {}

    def train(self, X_train: ndarray, y_train: ndarray, X_val: ndarray, y_val: ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: ndarray) -> ndarray:
        return self.model.predict(X)

    def log_mlflow(self, y_val: ndarray, y_pred: ndarray) -> Dict[str, Any]:
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.evaluate(y_val, y_pred)
        mlflow.log_params(self.best_params or {})
        mlflow.log_metrics({k: v for k, v in metrics["weighted avg"].items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(self.model, "model")
        return metrics
