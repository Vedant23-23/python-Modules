# mnist/classifier.py

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
import numpy as np

def load_mnist_sample(binary=False, classes=(0, 1)):
    digits = load_digits()
    X, y = digits.data, digits.target
    if binary:
        mask = np.isin(y, classes)
        X, y = X[mask], y[mask]
    return X, y

def train_classifier(X, y, poly_degree=1):
    if poly_degree > 1:
        poly = PolynomialFeatures(degree=poly_degree)
        X = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc, clf, X_test, y_test, y_pred
