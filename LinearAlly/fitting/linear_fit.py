# fitting/linear_fit.py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def generate_noisy_data(n=50, noise=1.0, seed=0):
    np.random.seed(seed)
    x = np.linspace(0, 10, n)
    y = 2 * x + 3 + np.random.normal(0, noise, size=n)
    return x.reshape(-1, 1), y

def linear_fit(x, y):
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    return y_pred, model

def polynomial_fit(x, y, degree=2):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    y_pred = model.predict(x_poly)
    return y_pred, model
