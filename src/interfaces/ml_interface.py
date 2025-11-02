import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans

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
        filepath = Path(filepath) if Path(filepath).is_absolute() else self.data_dir / filepath
        
        if file_type == "csv":
            return pd.read_csv(filepath)
        elif file_type == "json":
            return pd.read_json(filepath)
    
    def preprocess_text(self, text_data, fit: bool = True):
        if fit:
            return self.vectorizer.fit_transform(text_data).toarray()
        return self.vectorizer.transform(text_data).toarray()
    
    def preprocess_numeric(self, data, fit: bool = True):
        if fit:
            return self.scaler.fit_transform(data)
        return self.scaler.transform(data)
    
    def encode_labels(self, labels, fit: bool = True):
        if fit:
            return self.encoder.fit_transform(labels)
        return self.encoder.transform(labels)
    
    def split_data(self, X, y, test_size: float = 0.2, random_state: int = 42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def train_classifier(self, X_train, y_train, n_estimators: int = 100):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.model.fit(X_train, y_train)
        return self.model
    
    def train_regressor(self, X_train, y_train, n_estimators: int = 100):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.model.fit(X_train, y_train)
        return self.model
    
    def train_clusterer(self, X_train, n_clusters: int = 3):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(X_train)
        return self.model
    
    def train_custom(self, model, X_train, y_train=None):
        self.model = model
        if y_train is not None:
            # Supervised learning
            self.model.fit(X_train, y_train)
        else:
            # Unsupervised learning
            self.model.fit(X_train)
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("Current model doesn't support probability predictions")
    
    def get_cluster_labels(self, X=None):
        if hasattr(self.model, 'labels_'):
            if X is None:
                return self.model.labels_
            else:
                return self.model.predict(X)
        else:
            raise AttributeError("Current model is not a clustering model")


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
