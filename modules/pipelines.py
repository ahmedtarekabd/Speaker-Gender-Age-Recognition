import mlflow
from datetime import datetime
from models.catboost import CatBoostModel
from models.lightgbm import LightGBMModel
from sklearn.model_selection import train_test_split

# === Unified Model Pipeline ===
class ModelPipeline:
    def __init__(self, model_name="catboost"):
        self.model_name = model_name
        self.model_wrapper = CatBoostModel() if model_name == "catboost" else LightGBMModel()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def train(self, X, y, use_optuna=False, n_trials=20):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        with mlflow.start_run(run_name=f"{self.model_name}_{self.run_id}"):
            self.model_wrapper.train(X_train, y_train, X_val, y_val, use_optuna=use_optuna, n_trials=n_trials)
            y_pred_val = self.model_wrapper.predict(X_val)
            metrics = self.model_wrapper.log_mlflow(y_val, y_pred_val)

        return X_val, y_val, X_test, y_test

    def predict(self, X):
        return self.model_wrapper.predict(X)
