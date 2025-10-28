import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


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
    
    def train(self, model, X_train, y_train):
        self.model = model
        self.model.fit(X_train, y_train)
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
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


# # Example
# if __name__ == "__main__":
#     from sklearn.ensemble import RandomForestClassifier
    
#     # Initialize
#     ml = MachineLearningInterface()
    
#     # Create sample data
#     data = pd.DataFrame({
#         'text': ["Great product!", "Terrible experience", "Good value"],
#         'rating': [5, 1, 4],
#         'label': ['positive', 'negative', 'positive']
#     })
    
#     # Preprocess
#     X_text = ml.preprocess_text(data['text'])
#     X_numeric = ml.preprocess_numeric(data[['rating']])
#     X = np.hstack([X_text, X_numeric])
#     y = ml.encode_labels(data['label'])
    
#     # Split
#     X_train, X_test, y_train, y_test = ml.split_data(X, y)
    
#     # Train
#     model = RandomForestClassifier()
#     ml.train(model, X_train, y_train)
    
#     # Save
#     ml.save_model("sentiment_model")