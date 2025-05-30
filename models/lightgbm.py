import optuna
from models.base_model import BaseModel
import lightgbm as lgb
from numpy import ndarray
import numpy as np
from sklearn.utils import class_weight


# === LightGBM Implementation ===
class LightGBMModel(BaseModel):
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
            "objective": "multiclass",
            "num_class": len(set(y_train)),
            "metric": "multi_logloss",
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 50, 130),
            "max_depth": trial.suggest_int("max_depth", 20, 30),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "n_jobs": -1,
            "verbosity": -1
        }
        model = lgb.LGBMClassifier(**params)
        # Compute class weights if specified
        if class_weight_type:
            class_weights = class_weight.compute_sample_weight(class_weight_type, y=y_train)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=class_weights)
        else:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)]) # Fit without class weights
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
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val, class_weight_type), n_trials=n_trials, n_jobs=-1, show_progress_bar=True)
            self.best_params = study.best_params
            self.model = lgb.LGBMClassifier(**self.best_params, verbosity=-1)
        else:
            self.model = lgb.LGBMClassifier(**self.best_params)
        X, y = np.vstack([X_train, X_val]), np.hstack([y_train, y_val])
        if class_weight_type:
            # Compute class weights if specified
            class_weights = class_weight.compute_sample_weight(class_weight_type, y=y)
            self.model.fit(X, y, sample_weight=class_weights)
        else:
            # Fit without class weights
            self.model.fit(X, y)
