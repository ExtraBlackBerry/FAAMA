import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

class ModelEvaluatorInterface:
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

# TEST 
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target # type: ignore
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classification
    rf = RandomForestClassifier(random_state=69)
    rf.fit(X_train, y_train)
    rf_results = ModelEvaluatorInterface.evaluate_classifier(X_test, y_test, rf)
    print('rf results:', rf_results)

    # Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_results = ModelEvaluatorInterface.evaluate_regressor(X_test, y_test, lr)
    print('lr results:', lr_results)

    # Clustering
    km = KMeans(n_clusters=3, random_state=69)
    km.fit(X)
    km_results = ModelEvaluatorInterface.evaluate_clusterer(X, km)
    print('km results:', km_results)