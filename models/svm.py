from models.base_model import BaseModel
from sklearn.svm import SVC
from sklearn.utils import class_weight
import optuna
from numpy import ndarray
import numpy as np


# === SVM Implementation ===
class SVMModel(BaseModel):
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
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
        model = SVC(**params, probability=False)
        if class_weight_type:
            class_weights = class_weight.compute_sample_weight(class_weight_type, y=y_train)
            model.fit(X_train, y_train, sample_weight=class_weights)
        else:
            model.fit(X_train, y_train)
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
            self.model = SVC(**self.best_params, probability=False)
        else:
            self.model = SVC(**self.best_params, probability=False)
        X, y = np.vstack([X_train, X_val]), np.hstack([y_train, y_val])
        if class_weight_type:
            class_weights = class_weight.compute_sample_weight(class_weight_type, y=y)
            self.model.fit(X, y, sample_weight=class_weights)
        else:
            self.model.fit(X, y)