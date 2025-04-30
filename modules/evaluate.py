from numpy import ndarray # For type hinting
from sklearn.metrics import classification_report

# === Evaluation ===
class PerformanceAnalyzer:
    def evaluate(self, y_true: ndarray, y_pred: ndarray):
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        report_str = classification_report(y_true, y_pred, zero_division=0)
        return report, report_str