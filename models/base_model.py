import mlflow
from typing import Any, Dict
from numpy import ndarray
from sklearn.base import BaseEstimator
from modules.evaluate import PerformanceAnalyzer

# === Base Model Interface ===
class BaseModel:
    def __init__(self) -> None:
        self.model: BaseEstimator = None
        self.best_params: Dict[str, Any] = {}
        self.model_name: str = self.__class__.__name__  # Automatically set model name

    def train(self, X_train: ndarray, y_train: ndarray, X_val: ndarray, y_val: ndarray) -> None:
        raise NotImplementedError

    def predict(self, X: ndarray) -> ndarray:
        return self.model.predict(X)
    
    def score(self, X: ndarray, y: ndarray) -> float:
        return self.model.score(X, y)

    def log_mlflow(self, y_val: ndarray, y_pred: ndarray):
        """
        Logs model performance metrics and the trained model to MLflow.

        This method evaluates the model's performance using the provided true 
        and predicted values, logs the evaluation metrics to MLflow, and saves 
        the trained model to MLflow for tracking and reproducibility.

        Args:
            y_val (ndarray): The ground truth target values.
            y_pred (ndarray): The predicted target values from the model.

        Returns:
            str | dict: A string representation of the evaluation metrics or 
            a dictionary containing the metrics.

        Input Example:
            y_val = np.array([1, 0, 1, 1, 0])
            y_pred = np.array([1, 0, 1, 0, 0])
        """
        analyzer = PerformanceAnalyzer()
        metrics, metrics_str = analyzer.evaluate(y_val, y_pred)
        mlflow.log_params(self.best_params or {})

        for category, category_metrics in metrics.items():
            if isinstance(category_metrics, dict):
                mlflow.log_metrics({f"{category}_{k}": v for k, v in category_metrics.items() if isinstance(v, (int, float))})

        mlflow.sklearn.log_model(self.model, "model")
        mlflow.set_tag("model_name", self.model_name)  # Add model name as a tag
        return metrics_str

    def load_model_from_run(
        self, 
        run_id: str = None, 
        experiment_id: str = None, 
        experiment_name: str = None, 
        best_metric: str = None, 
        maximize: bool = True, 
        additional_tags: Dict[str, str] = None
    ) -> None:
        """
        Loads a model from a specific MLflow run, the last run, or the best run based on a metric.
    
        Args:
            run_id (str, optional): The ID of the MLflow run from which to load the model. Defaults to None.
            experiment_id (str, optional): The ID of the MLflow experiment to search for runs. Defaults to None.
            experiment_name (str, optional): The name of the MLflow experiment to search for runs. Required if run_id is not provided.
            best_metric (str, optional): The metric to use for selecting the best run. Defaults to None. Example: "weighted avg_f1-score
            maximize (bool, optional): Whether to maximize or minimize the metric when selecting the best run. Defaults to True.
            additional_tags (dict, optional): Additional tags to filter runs. Defaults to None.
    
        Raises:
            ValueError: If neither `run_id` nor `experiment_name` is provided.
        """
        if run_id:
            # Load model from the specified run ID
            run = mlflow.get_run(run_id)
        # elif experiment_id or experiment_name:
        else:
            # Default to the first experiment if not provided
            if not (experiment_id or experiment_name): experiment_id = "0"

            # Determine the order_by clause
            if best_metric:
                metric_order = f"metrics.'{best_metric}' {'DESC' if maximize else 'ASC'}"
                order_by = [metric_order]
            else:
                order_by = ["start_time DESC"]
    
            # Build the filter string
            filter_string = f"attributes.run_name LIKE '{self.model_name}%'"
            if additional_tags:
                for key, value in additional_tags.items():
                    filter_string += f" and tags.{key} = '{value}'"
    
            # Search for the most relevant run with the model name and additional tags as filters
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id] if experiment_id else None,
                experiment_names=[experiment_name] if experiment_name else None,
                filter_string=filter_string,
                order_by=order_by,
                max_results=1
            )
    
            if runs.empty:
                raise ValueError(f"No runs found in experiment '{experiment_name}' with the specified criteria.")
    
            # Get the best or last run
            run = mlflow.get_run(runs.iloc[0]["run_id"])
        # else:
        #     raise ValueError("Either 'run_id' or 'experiment_id' or 'experiment_name' must be provided.")
    
        # Load the model and metadata
        # self.model = mlflow.pyfunc.load_model(mlflow.get_tracking_uri() + f"/{experiment_id}/{run.info.run_id}/artifacts/model")
        self.model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")
        self.best_params = run.data.params
        self.metrics = run.data.metrics
        self.model_name = run.info.run_name
        self.run_id = run.info.run_id
