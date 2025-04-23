from sklearn.metrics import classification_report

# === Evaluation ===
class PerformanceAnalyzer:
    def evaluate(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        return report