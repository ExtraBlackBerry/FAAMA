import pandas as pd
import numpy as np
import joblib
from pathlib import Path
# Model Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor


class MachineLearningInterface:
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = None
    
    def load_data(self, filepath: str, file_type: str = "csv"):
        if file_type == "csv":
            pd.read_csv(filepath)
        elif file_type == "json":
            pd.read_json(filepath)
    
    def preprocess_text(self, text_data: list[str], fit: bool = True):
        if fit:
            self.vectorizer.fit_transform(text_data).toarray()
        self.vectorizer.transform(text_data).toarray()
    
    def preprocess_numeric(self, data: np.ndarray, fit: bool = True):
        if fit:
            self.scaler.fit_transform(data)
        self.scaler.transform(data)
    
    def encode_labels(self, labels: list[str], fit: bool = True): # Assuming label encoder, 1D list for labels.
        if fit:
            self.encoder.fit_transform(labels)
        self.encoder.transform(labels)
    
    def split_data(self, X, y, test_size: float = 0.2, random_state: int = 42):
        train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train models
    def train_classifier(self, X_train: np.ndarray, y_train: pd.Series, model_type: str, n_estimators: int = 100):
        match model_type:
            case 'RandomForestClassifier':
                self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                self.model.fit(X_train, y_train)
            case 'XGBoostClassifier':
                self.model = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss', random_state=42)
                self.model.fit(X_train, y_train)
            case 'LogisticRegression':
                self.model = LogisticRegression(max_iter=200, random_state=42)
                self.model.fit(X_train, y_train)
            case 'SVC':
                self.model = SVC(probability=True, random_state=42)
                self.model.fit(X_train, y_train)
    
    def train_regressor(self, X_train: np.ndarray, y_train: pd.Series, model_type: str, n_estimators: int = 100):
        match model_type:
            case 'RandomForestRegressor':
                self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                self.model.fit(X_train, y_train)
            case 'XGBoostRegressor':
                self.model = XGBRegressor(n_estimators=n_estimators, random_state=42)
                self.model.fit(X_train, y_train)
            case 'LinearRegression':
                self.model = LinearRegression()
                self.model.fit(X_train, y_train)
            case 'DecisionTreeRegressor':
                self.model = DecisionTreeRegressor(random_state=42)
                self.model.fit(X_train, y_train)
    
    def train_clusterer(self, X_train: np.ndarray, model_type: str, n_clusters: int = 3):
        match model_type:
            case 'KMeans':
                    self.model = KMeans(n_clusters=n_clusters, random_state=42)
                    self.model.fit(X_train)
            case 'DBSCAN':
                self.model = DBSCAN(eps=0.5, min_samples=5)
                self.model.fit(X_train)
            case 'AgglomerativeClustering':
                self.model = AgglomerativeClustering(n_clusters=n_clusters)
                self.model.fit(X_train)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            print("Current model does not support probability predictions.")
    
    def get_cluster_labels(self, X=None):
        if hasattr(self.model, 'labels_'):
            if X is None:
                return self.model.labels_
            else:
                return self.model.predict(X)
        else:
            print("Current model is not a clustering model")


    def save_model(self, model_name: str):
        model_path = self.models_dir / f"{model_name}.joblib"
        preprocessors_path = self.models_dir / f"{model_name}_preprocessors.joblib"
        
        joblib.dump(self.model, model_path)
        joblib.dump({
            'scaler': self.scaler,
            'encoder': self.encoder,
            'vectorizer': self.vectorizer
        }, preprocessors_path)
    
    def load_model(self, model_name: str):
        model_path = self.models_dir / f"{model_name}.joblib"
        preprocessors_path = self.models_dir / f"{model_name}_preprocessors.joblib"
        
        self.model = joblib.load(model_path)
        preprocessors = joblib.load(preprocessors_path)
        
        self.scaler = preprocessors['scaler']
        self.encoder = preprocessors['encoder']
        self.vectorizer = preprocessors['vectorizer']
        return self.model
    
    def get_available_models(self) -> list[str]:
        model_files = self.models_dir.glob("*.joblib")
        model_names = [file.stem.replace("_preprocessors", "") for file in model_files if "_preprocessors" not in file.stem]
        return model_names
    



    # UNIT TEST
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

# Init interface
ml_interface = MachineLearningInterface()



print("Available models before training:", ml_interface.get_available_models())


# Clkassification Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = ml_interface.preprocess_numeric(X_train, fit=True)
X_test_scaled = ml_interface.preprocess_numeric(X_test, fit=False)

ml_interface.train_classifier(X_train_scaled, y_train, 'RandomForestClassifier', n_estimators=50)
predictions = ml_interface.predict(X_test_scaled)
accuracy = (predictions == y_test).mean()
print(f"Accuracy: {accuracy:.2f}")
print(f"Predictions: {predictions[:5]}")

ml_interface.save_model("iris_rf_classifier")
print("Available models after saving:", ml_interface.get_available_models())

# Regression Test
X_reg = X[:, 1:]  # Use features 2-4 as input
y_reg = X[:, 0]   # Use first feature (sepal length) as target

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_train_reg_scaled = ml_interface.preprocess_numeric(X_train_reg, fit=True)
X_test_reg_scaled = ml_interface.preprocess_numeric(X_test_reg, fit=False)

ml_interface.train_regressor(X_train_reg_scaled, y_train_reg, 'RandomForestRegressor', n_estimators=50)
predictions_reg = ml_interface.predict(X_test_reg_scaled)
mse = np.mean((predictions_reg - y_test_reg) ** 2)
print(f"Mean Squared Error: {mse:.4f}")
print(f"Sample predictions: {predictions_reg[:5]}")
print(f"Actual values: {y_test_reg[:5]}")

ml_interface.save_model("iris_rf_regressor")
print("Available models after saving regressor:", ml_interface.get_available_models())

# Clulstering Test
X_scaled = ml_interface.preprocess_numeric(X, fit=True)
ml_interface.train_clusterer(X_scaled, 'KMeans', n_clusters=3)
cluster_labels = ml_interface.get_cluster_labels()
print(f"Cluster labels (first 10): {cluster_labels[:10]}")
print(f"Unique clusters: {np.unique(cluster_labels)}")
print(f"Cluster distribution: {np.bincount(cluster_labels)}")

ml_interface.save_model("iris_kmeans_cluster")
print("Available models after saving clusterer:", ml_interface.get_available_models())


