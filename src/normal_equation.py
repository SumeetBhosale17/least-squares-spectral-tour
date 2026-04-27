"""Closed-form linear regression solvers, ordered from numerically worst to best.

The naive textbook recipe is theta* = (X^T X)^-1 X^T y. Forming X^T X is fast
but squares the condition number: kappa(X^T X) = kappa(X)^2. That is why this
file exposes four variants -- so the blog can show, side by side, that "the
normal equation" is really a family of methods that trade speed for stability.

Recommended order of preference for production code:
    fit_lstsq (SVD)  >  fit_qr  >  fit_cholesky  >  fit_normal_equation (LU on XtX)
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def fit_normal_equation(X, y):
    """Naive textbook NE: build X^T X, solve via LU. Fastest, least stable.

    Doubles the effective condition number because solve() works on X^T X,
    whose kappa is kappa(X)^2. Use only when you know X is well-conditioned.
    """
    XtX = X.T @ X
    Xty = X.T @ y
    return np.linalg.solve(XtX, Xty)


def fit_cholesky(X, y):
    """Same Gram-matrix recipe as fit_normal_equation, but uses Cholesky
    instead of LU. X^T X is symmetric positive (semi)-definite, so Cholesky
    is the right factorization: ~2x faster than LU and half the memory.

    Still suffers from the kappa(X)^2 penalty -- that is a property of forming
    X^T X, not of the solver applied to it.
    """
    XtX = X.T @ X
    Xty = X.T @ y
    c, low = cho_factor(XtX, lower=True)
    return cho_solve((c, low), Xty)


def fit_qr(X, y):
    """Stable closed-form via QR. Works on X directly, never builds X^T X,
    so the effective condition number stays at kappa(X), not kappa(X)^2.
    Cost: O(n p^2). The right default for a hand-rolled exact solver.
    """
    Q, R = np.linalg.qr(X, mode="reduced")
    return np.linalg.solve(R, Q.T @ y)


def fit_lstsq(X, y):
    """SVD-based least squares. Most stable, handles rank deficiency, what
    scikit-learn's LinearRegression uses internally. Cost: O(n p^2) with a
    larger constant than QR.
    """
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta


def fit_pinv(X, y, rcond=None):
    """Pseudo-inverse via SVD. Equivalent answer to lstsq for well-posed
    problems; useful when you want explicit control of the rcond cutoff to
    truncate tiny singular values.
    """
    return np.linalg.pinv(X, rcond=rcond) @ y
