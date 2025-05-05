import mlflow
from datetime import datetime
from models.lightgbm import LightGBMModel
from sklearn.model_selection import train_test_split
from models.base_model import BaseModel
from typing import Tuple
import numpy as np
from typing import Dict, Optional
from modules.evaluate import PerformanceAnalyzer

# === Unified Model Pipeline ===
class ModelPipeline:
    def __init__(self, model: BaseModel = LightGBMModel) -> None:
        self.model = model()
        self.model_name = self.model.__class__.__name__
        self.best_params = {}
        self.metrics = {}
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_model(self, run_id: str = None, experiment_id: str = None, experiment_name: str = None, best_metric: str = None, maximize: bool = True, additional_tags: Dict[str, str] = None) -> None:
        self.model.load_model_from_run(run_id, experiment_id, experiment_name, best_metric, maximize, additional_tags)
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        use_optuna: bool = False, 
        n_trials: int = 20,
        split: bool = True,
        train_size: float = 0.75,
        val_size: float = 0.1,
        save_run: bool = True,
        experiment_name: Optional[str] = None,
        run_name: str = None,
        mlflow_tags: Optional[Dict[str, str]] = None,
    ) -> (str | dict):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_size/(1-train_size), random_state=42)

        try:
            experiment_id = mlflow.set_experiment(experiment_name).experiment_id if experiment_name else None
        except mlflow.exceptions.RestException:
            experiment_id = None
            
        with mlflow.start_run(run_name=run_name or f"{self.model_name}_{self.run_id}", experiment_id=experiment_id):
            self.model.train(X_train, y_train, X_val, y_val, use_optuna=use_optuna, n_trials=n_trials)

            ## If X_test and y_test are not provided, use X_val and y_val for testing
            if X_test is None or y_test is None:
                X_test, y_test = X_val, y_val

            y_pred_test = self.model.predict(X_test)
            if save_run:
                metrics = self.model.log_mlflow(y_test, y_pred_test)
                mlflow.set_tags(mlflow_tags or {})
            else:
                metrics = self.model.classification_report(y_test, y_pred_test)

        return metrics


    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return self.model.score(X, y)

    def classification_report(self, X: np.ndarray, y: np.ndarray) -> str:
        evaluator = PerformanceAnalyzer()
        y_pred = self.model.predict(X)
        report, report_str = evaluator.evaluate(y, y_pred)
        return report_str