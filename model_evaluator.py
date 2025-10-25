import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
)

class ModelEvaluator:
    @staticmethod
    def evaluate_classifier(test_x: np.ndarray, test_y: np.ndarray, model) -> dict[str, float]:
        y_pred = model.predict(test_x) # Maybe passing predictions instead of model would be better
        results = {
            "accuracy": accuracy_score(test_y, y_pred),
            "precision": precision_score(test_y, y_pred, zero_division=0),
            "recall": recall_score(test_y, y_pred, zero_division=0),
            "f1_score": f1_score(test_y, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(test_y, y_pred).tolist(),
        }
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(test_x)
            results["roc_auc"] = roc_auc_score(test_y, y_pred_proba)
        return results
    
    @staticmethod
    def evaluate_regressor(test_x: np.ndarray, test_y: np.ndarray, model) -> dict[str, float]:
        return {}
    
    @staticmethod
    def evaluate_clusterer(test_x: np.ndarray, test_y: np.ndarray, model) -> dict[str, float]:
        return {}