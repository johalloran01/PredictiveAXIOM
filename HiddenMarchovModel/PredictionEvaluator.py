from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

class PredictionEvaluator:
    def __init__(self):
        """Initialize the evaluator with no preset metrics."""
        self.metrics = {}

    def evaluate_accuracy(self, predictions, actual, verbose=True):
        """Evaluate and store the accuracy of predictions."""
        accuracy = accuracy_score(actual, predictions)
        self.metrics['accuracy'] = accuracy
        if verbose:
            print(f"Accuracy: {accuracy}")
        return accuracy

    def evaluate_confusion_matrix(self, predictions, actual, verbose=True):
        """Generate a confusion matrix for the predictions."""
        cm = confusion_matrix(actual, predictions)
        self.metrics['confusion_matrix'] = cm
        if verbose:
            print("Confusion Matrix:\n", cm)
        return cm

    def evaluate_classification_report(self, predictions, actual, verbose=True):
        """Generate a classification report showing precision, recall, and F1-score."""
        report = classification_report(
            actual,
            predictions,
            output_dict=True,
            zero_division=0  # Avoid warnings for undefined precision/recall
        )
        self.metrics['classification_report'] = report
        if verbose:
            print("Classification Report:\n", classification_report(actual, predictions, zero_division=0))
        return report

    def compare_models(self, model_predictions, verbose=True):
        """Compare multiple models by accuracy, given a dictionary of model predictions."""
        comparisons = {model_name: self.evaluate_accuracy(pred, actual, verbose=False)
                       for model_name, (pred, actual) in model_predictions.items()}
        self.metrics['model_comparisons'] = comparisons
        if verbose:
            print("Model Comparisons:")
            for model_name, accuracy in comparisons.items():
                print(f"{model_name}: {accuracy}")
        return comparisons