"""Synthetic dataset generators with knobs for n, p, scaling, and conditioning."""

import numpy as np


def make_regression(n, p, noise=0.1, seed=0, feature_scales=None, condition_number=None):
    """Generate y = X @ theta_true + noise.

    feature_scales : array of shape (p,) — multiplies each column. Use to create
        wildly different feature magnitudes (e.g., [1, 1e3, 1e-2, ...]) so that
        the unscaled GD experiment is dramatic.

    condition_number : if set, X is constructed via SVD with a controlled
        spectrum so kappa(X^T X) ~= condition_number^2. Overrides feature_scales.
    """
    rng = np.random.default_rng(seed)

    if condition_number is not None:
        U, _ = np.linalg.qr(rng.standard_normal((n, p)))
        V, _ = np.linalg.qr(rng.standard_normal((p, p)))
        s = np.logspace(0, -np.log10(condition_number), p)
        X = U @ np.diag(s) @ V.T
    else:
        X = rng.standard_normal((n, p))
        if feature_scales is not None:
            X = X * np.asarray(feature_scales).reshape(1, -1)

    theta_true = rng.standard_normal(p)
    y = X @ theta_true + noise * rng.standard_normal(n)
    return X, y, theta_true


def make_collinear(n, p, seed=0):
    """Rank-deficient X: last column is a copy of the first. Breaks normal eq."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    X[:, -1] = X[:, 0]
    theta_true = rng.standard_normal(p)
    y = X @ theta_true + 0.1 * rng.standard_normal(n)
    return X, y, theta_true


def add_bias(X):
    """Prepend a column of ones for the intercept term."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def standardize(X):
    """Zero-mean, unit-variance per column. Returns (X_scaled, mean, std)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    return (X - mean) / std_safe, mean, std_safe
