# modules/regularization.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from skimage import color, restoration
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from scipy.signal import convolve2d

def generate_regression_data(n_samples=100, noise=0.1):
    np.random.seed(0)
    X = np.sort(2 * np.random.rand(n_samples, 1) - 1, axis=0)
    y = np.sin(2 * np.pi * X).ravel() + noise * np.random.randn(n_samples)
    return X, y

def regularized_fit(X, y, model_type='Ridge', alpha=1.0):
    if model_type == 'Ridge':
        model = Ridge(alpha=alpha)
    else:
        model = Lasso(alpha=alpha, max_iter=10000)

    model.fit(X, y)
    X_plot = np.linspace(-1, 1, 200).reshape(-1, 1)
    y_pred = model.predict(X_plot)
    return X, y, X_plot, y_pred

def deblur_image(uploaded_image):
    from skimage import data, img_as_float

    if uploaded_image is not None:
        img = img_as_float(imread(uploaded_image))  # Keep original RGB image as float
    else:
        img = img_as_float(data.astronaut())

    img = img[:256, :256]  # Crop to a fixed size
    psf = np.ones((5, 5)) / 25  # Point spread function

    # Apply the blur to each color channel separately
    blurred = np.stack([convolve2d(img[:, :, c], psf, mode='same') for c in range(3)], axis=-1)
    deconvolved = np.stack([restoration.wiener(blurred[:, :, c], psf, 0.1) for c in range(3)], axis=-1)

    return img, blurred, deconvolved

