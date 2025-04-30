import mlflow
from datetime import datetime
from models.lightgbm import LightGBMModel
from sklearn.model_selection import train_test_split
from models.base_model import BaseModel
from typing import Tuple
import numpy as np
from typing import Dict

# === Unified Model Pipeline ===
class ModelPipeline:
    def __init__(self, model: BaseModel = LightGBMModel) -> None:
        self.model_name = model.__class__.__name__
        self.model = model()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        use_optuna: bool = False, 
        n_trials: int = 20,
        train_size: float = 0.75,
        val_size: float = 0.1,
    ) -> (str | dict):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_size/(1-train_size), random_state=42)

        with mlflow.start_run(run_name=f"{self.model_name}_{self.run_id}"):
            self.model.train(X_train, y_train, X_val, y_val, use_optuna=use_optuna, n_trials=n_trials)
            y_pred_val = self.model.predict(X_val)
            metrics = self.model.log_mlflow(y_val, y_pred_val)

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return self.model.score(X, y)
