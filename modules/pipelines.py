import mlflow
from datetime import datetime
from models.lightgbm import LightGBMModel
from modules.preprocessing import AudioPreprocessor
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
        self.preprocessor = AudioPreprocessor()

    def load_model(self, run_id: str = None, experiment_id: str = None, experiment_name: str = None, best_metric: str = None, maximize: bool = True, additional_tags: Dict[str, str] = None) -> None:
        self.model.load_model_from_run(run_id, experiment_id, experiment_name, best_metric, maximize, additional_tags)
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None, 
        y_test: Optional[np.ndarray] = None, 
        use_optuna: bool = False, 
        n_trials: int = 20,
        class_weight_type: str = "",
        save_run: bool = True,
        experiment_name: Optional[str] = None,
        run_name: str = None,
        mlflow_tags: Optional[Dict[str, str]] = None,
    ) -> (str | dict):

        try:
            experiment_id = mlflow.set_experiment(experiment_name).experiment_id if experiment_name else None
        except mlflow.exceptions.RestException:
            experiment_id = None
            
        with mlflow.start_run(run_name=run_name or f"{self.model_name}_{self.run_id}", experiment_id=experiment_id):
            self.model.train(X_train, y_train, X_val, y_val, use_optuna=use_optuna, n_trials=n_trials, class_weight_type=class_weight_type)

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

    def load_model_from_registry(self, model_name: str, version: int = None) -> None:
        self.model.load_model_from_registry(model_name, version)
    
    def register_model(
        self, 
        run_id: str, 
        model_name: str = None, 
        tags: Dict[str, str] = None
    ) -> None:
        self.model.register_model(run_id, model_name, tags)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return self.model.score(X, y)

    def classification_report(self, X: np.ndarray, y: np.ndarray) -> str:
        evaluator = PerformanceAnalyzer()
        y_pred = self.model.predict(X)
        report, report_str = evaluator.evaluate(y, y_pred)
        return report_str