import mlflow
from modules.evaluate import PerformanceAnalyzer

# === Base Model Interface ===
class BaseModel:
    def __init__(self):
        self.model = None
        self.best_params = {}

    def train(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError

    def predict(self, X):
        return self.model.predict(X)

    def log_mlflow(self, y_val, y_pred):
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.evaluate(y_val, y_pred)
        mlflow.log_params(self.best_params or {})
        mlflow.log_metrics({k: v for k, v in metrics["weighted avg"].items() if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(self.model, "model")
        return metrics
