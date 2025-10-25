import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

class ModelEvaluator:
    @staticmethod
    def evaluate_classifier(test_x: np.ndarray, test_y: np.ndarray, model) -> dict[str, float]:
        y_pred = model.predict(test_x) # Maybe passing predictions instead of model would be better
        results = {
            "accuracy": accuracy_score(test_y, y_pred),
            "precision_weighted": precision_score(test_y, y_pred, zero_division=0, average='weighted'),
            "recall_weighted": recall_score(test_y, y_pred, zero_division=0, average='weighted'),
            "f1_score_weighted": f1_score(test_y, y_pred, zero_division=0, average='weighted'),
            "confusion_matrix": confusion_matrix(test_y, y_pred).tolist(),
        }
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(test_x)
            results["roc_auc"] = roc_auc_score(test_y, y_pred_proba, multi_class='ovr')
        return results
    
    @staticmethod
    def evaluate_regressor(test_x: np.ndarray, test_y: np.ndarray, model) -> dict[str, float]:
        y_pred = model.predict(test_x)
        results = {
            "mae": mean_absolute_error(test_y, y_pred),
            "mse": mean_squared_error(test_y, y_pred),
            "r2": r2_score(test_y, y_pred),
            "rmse": root_mean_squared_error(test_y, y_pred) # Shiny new function
        }
        return results
    
    @staticmethod
    def evaluate_clusterer(train_x: np.ndarray, model) -> dict[str, float]:
        # Takes training data since clustering is unsupervised
        results = {
            "silhouette": silhouette_score(train_x, model.labels_),
            "davies_bouldin": davies_bouldin_score(train_x, model.labels_),
            "calinski_harabasz": calinski_harabasz_score(train_x, model.labels_)
        }
        return results
