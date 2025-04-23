import optuna
from models.base_model import BaseModel
import lightgbm as lgb

# === LightGBM Implementation ===
class LightGBMModel(BaseModel):
    def __init__(self):
        super().__init__()

    def objective(self, trial, X_train, y_train, X_val, y_val):
        params = {
            "objective": "multiclass",
            "num_class": len(set(y_train)),
            "metric": "multi_logloss",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "n_jobs": -1,
            "verbosity": -1
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        return model.score(X_val, y_val)

    def train(self, X_train, y_train, X_val, y_val, use_optuna=False, n_trials=20):
        if use_optuna:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials, n_jobs=-1)
            self.best_params = study.best_params
            self.model = lgb.LGBMClassifier(**self.best_params, verbosity=-1)
        else:
            self.model = lgb.LGBMClassifier()
        self.model.fit(X_train, y_train)
