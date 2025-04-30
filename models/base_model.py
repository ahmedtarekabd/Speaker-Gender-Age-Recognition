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
    
    def score(self, X: ndarray, y: ndarray) -> float:
        return self.model.score(X, y)

    def log_mlflow(self, y_val: ndarray, y_pred: ndarray):
        """
        Logs model performance metrics and the trained model to MLflow.

        This method evaluates the model's performance using the provided true 
        and predicted values, logs the evaluation metrics to MLflow, and saves 
        the trained model to MLflow for tracking and reproducibility.

        Args:
            y_val (ndarray): The ground truth target values.
            y_pred (ndarray): The predicted target values from the model.

        Returns:
            str | dict: A string representation of the evaluation metrics or 
            a dictionary containing the metrics.

        Input Example:
            y_val = np.array([1, 0, 1, 1, 0])
            y_pred = np.array([1, 0, 1, 0, 0])
        """
        analyzer = PerformanceAnalyzer()
        metrics, metrics_str = analyzer.evaluate(y_val, y_pred)
        mlflow.log_params(self.best_params or {})
        for category, category_metrics in metrics.items():
            if isinstance(category_metrics, dict):
                mlflow.log_metrics({f"{category}_{k}": v for k, v in category_metrics.items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(self.model, "model")
        return metrics_str
