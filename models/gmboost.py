from sklearn.ensemble import GradientBoostingClassifier
import optuna
from models.base_model import BaseModel
from numpy import ndarray

# === GradientBoosting Implementation ===
class GradientBoostingModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    def objective(
        self, 
        trial: optuna.trial.Trial, 
        X_train: ndarray, 
        y_train: ndarray, 
        X_val: ndarray, 
        y_val: ndarray
    ) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0)
        }
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)

    def train(
        self, 
        X_train: ndarray, 
        y_train: ndarray, 
        X_val: ndarray, 
        y_val: ndarray, 
        use_optuna: bool = False, 
        n_trials: int = 20
    ) -> None:
        if use_optuna:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
            self.best_params = study.best_params
            self.model = GradientBoostingClassifier(**self.best_params)
        else:
            self.model = GradientBoostingClassifier()
        self.model.fit(X_train, y_train)
