import numpy as np
import mlflow
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from modules.pipelines import ModelPipeline
from models.xgboost import XGBoostModel
from typing import Optional


class StackPipeline:
    def __init__(self):
        # Initialize two XGBoost pipelines for version 1 and version 2
        self.model_v1 = ModelPipeline(model=XGBoostModel)
        self.model_v2 = ModelPipeline(model=XGBoostModel)
        self.stacking_clf = None
        self.run_id = None
        self.model_name = "Stacked_XGBoost"

    def load_models(self):
        # Load XGBoost model version 1
        self.model_v1.load_model_from_registry(model_name="XGBoost", version=1)
        print("✅ Loaded XGBoost model version 1")

        # Load XGBoost model version 2
        self.model_v2.load_model_from_registry(model_name="XGBoost", version=2)
        print("✅ Loaded XGBoost model version 2")

    def train(self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None, 
        y_test: Optional[np.ndarray] = None, 
        experiment_name: Optional[str] = None,
        passthrough: bool = False,
    ):
        # Define base classifiers
        base_classifiers = [
            ('xgb_v1', XGBClassifier(
                colsample_bytree=0.7019871292400243,
                gamma=2.26312455814313, 
                learning_rate=0.033500535606589985, 
                max_depth=20,
                min_child_weight=9, 
                n_estimators=593, 
                scale_pos_weight=4.507748417170937, 
                subsample=0.6687408408803283,
                random_state=42,
                device="cuda",
            )),  # Model from version 1
            ('xgb_v2', XGBClassifier(
                colsample_bytree=0.9930481843526001,
                gamma=2.0025706931421947,
                learning_rate=0.06965046825646097,
                max_depth=20,
                min_child_weight =9,
                n_estimators = 424,
                scale_pos_weight = 1.064274380022721,
                subsample = 0.6792632413542896,
                random_state=42,
                device="cuda",
            ))  # Model from version 2
        ]

        # Define the stacking classifier with a meta-model
        self.stacking_clf = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=XGBClassifier(n_estimators=50, random_state=42),  # Meta-model
            stack_method='predict_proba',  # Use probabilities as input to the meta-model
            passthrough=passthrough,  # Pass original features to the meta-model
        )

        # Concatenate training and validation data
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])

        # Train the stacking classifier
        self.stacking_clf.fit(X_combined, y_combined)
        print("✅ Stacking classifier trained successfully")

        # If test data is provided, evaluate the model
        if X_test is not None and y_test is not None:
            self.log_mlflow(X_test, y_test, experiment_name=experiment_name)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Ensure the stacking classifier is trained
        if self.stacking_clf is None:
            raise ValueError("Stacking classifier is not trained. Call `train_stacking_classifier` first.")

        # Predict using the stacking classifier
        return self.stacking_clf.predict(X)

    def classification_report(self, X: np.ndarray, y: np.ndarray) -> str:
        # Generate predictions
        preds = self.predict(X)

        # Evaluate performance
        report = classification_report(y, preds)
        print("Classification Report:")
        print(report)
        return report

    def log_mlflow(self, X_test: np.ndarray, y_test: np.ndarray, experiment_name: str = None, run_name: str = None):
        # Log the stacking classifier to MLflow
        try:
            experiment_id = mlflow.set_experiment(experiment_name).experiment_id if experiment_name else None
        except mlflow.exceptions.RestException:
            experiment_id = None

        with mlflow.start_run(run_name=run_name or f"{self.model_name}", experiment_id=experiment_id) as run:
            self.run_id = run.info.run_id

            # Log model parameters
            mlflow.log_param("base_estimators", ["xgb_v1", "xgb_v2"])
            mlflow.log_param("meta_model", "XGBClassifier")

            # Evaluate and log metrics
            preds = self.predict(X_test)
            report = classification_report(y_test, preds, output_dict=True)
            for key, value in report.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        mlflow.log_metric(f"{key}_{sub_key}", sub_value)
                else:
                    mlflow.log_metric(key, value)

            # Log the stacking classifier
            mlflow.sklearn.log_model(self.stacking_clf, "model")
            print(f"✅ Stacking model logged to MLflow with run ID: {self.run_id}")

    def load_model_from_registry(self, model_name: str, version: int = None):
        # Load the stacking model from MLflow registry
        model_uri = f"models:/{model_name}/{version}" if version else f"models:/{model_name}/latest"
        self.stacking_clf: StackingClassifier = mlflow.sklearn.load_model(model_uri)
        self.model_v1.model = self.stacking_clf.named_estimators_['xgb_v1']
        self.model_v2.model = self.stacking_clf.named_estimators_['xgb_v2']
        self.stacking_clf = self.stacking_clf.final_estimator_
        print(f"✅ Loaded stacking model from MLflow registry: {model_uri}")

    def register_model(self, model_name: str, tags: dict = None):
        # Register the stacking model to MLflow registry
        if not self.run_id:
            raise ValueError("No run ID found. Train and log the model before registering.")
        mlflow.register_model(f"runs:/{self.run_id}/stacking_model", model_name)
        print(f"✅ Registered stacking model to MLflow registry with name: {model_name}")