from catboost import CatBoostClassifier
import optuna
import cupy as cp
from models.base_model import BaseModel

# === CatBoost Implementation ===
class CatBoostModel(BaseModel):
    def __init__(self):
        super().__init__()

    def objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
            "task_type": trial.suggest_categorical("task_type", ["GPU" if cp.cuda.is_available() else "CPU"]),
            "verbose": False
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        return model.score(X_val, y_val)

    def train(self, X_train, y_train, X_val, y_val, use_optuna=False, n_trials=20):
        if use_optuna:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
            self.best_params = study.best_params
            self.model = CatBoostClassifier(**self.best_params, verbose=False)
        else:
            self.model = CatBoostClassifier(verbose=0)
        self.model.fit(X_train, y_train)
