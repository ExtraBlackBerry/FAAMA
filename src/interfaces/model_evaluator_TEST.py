from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from interfaces.model_evaluator import ModelEvaluator

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target # type: ignore
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Classification
rf = RandomForestClassifier(random_state=69)
rf.fit(X_train, y_train)
rf_results = ModelEvaluator.evaluate_classifier(X_test, y_test, rf)
print('rf results:', rf_results)

# Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_results = ModelEvaluator.evaluate_regressor(X_test, y_test, lr)
print('lr results:', lr_results)

# Clustering
km = KMeans(n_clusters=3, random_state=69)
km.fit(X)
km_results = ModelEvaluator.evaluate_clusterer(X, km)
print('km results:', km_results)