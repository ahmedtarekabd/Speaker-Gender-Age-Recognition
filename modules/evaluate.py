from numpy import ndarray # For type hinting
from sklearn.metrics import classification_report

# === Evaluation ===
class PerformanceAnalyzer:
    def evaluate(self, y_true: ndarray, y_pred: ndarray) -> dict:
        report = classification_report(y_true, y_pred, output_dict=True)
        return report