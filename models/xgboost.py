from models.base_model import BaseModel
from xgboost import XGBClassifier
import optuna
from numpy import ndarray
import numpy as np
import cupy as cp
from sklearn.utils import class_weight


# === XGBoost Implementation ===
class XGBoostModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    def objective(
        self, 
        trial: optuna.trial.Trial, 
        X_train: ndarray, 
        y_train: ndarray, 
        X_val: ndarray, 
        y_val: ndarray,
        class_weight_type: str = "",
    ) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 15, 20),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
            "device": "cuda" if cp.cuda.is_available() else "cpu",
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric="mlogloss")
        if class_weight_type:
            class_weights = class_weight.compute_sample_weight(class_weight_type, y=y_train)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False, sample_weight=class_weights)
        else:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model.score(X_val, y_val)

    def train(
        self, 
        X_train: ndarray, 
        y_train: ndarray, 
        X_val: ndarray, 
        y_val: ndarray, 
        use_optuna: bool = False, 
        n_trials: int = 20,
        class_weight_type: str = "",
    ) -> None:
        if use_optuna:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val, class_weight_type), n_trials=n_trials, n_jobs=2, show_progress_bar=True)
            self.best_params = study.best_params
            self.model = XGBClassifier(**self.best_params, use_label_encoder=False, eval_metric="mlogloss")
        else:
            self.model = XGBClassifier(**self.best_params, use_label_encoder=False, eval_metric="mlogloss")
        X, y = np.vstack([X_train, X_val]), np.hstack([y_train, y_val])
        if class_weight_type:
            # Source: https://stackoverflow.com/questions/42192227/xgboost-python-classifier-class-weight-option
            class_weights = class_weight.compute_sample_weight(class_weight_type, y=y)
            self.model.fit(X, y, sample_weight=class_weights)
        else:
            self.model.fit(X, y)