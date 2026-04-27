"""Batch gradient descent for ordinary least squares.

Loss:  L(theta) = (1/2n) * ||X theta - y||^2
Grad:  dL/dtheta = (1/n) * X^T (X theta - y)
Update: theta <- theta - eta * grad
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class GDResult:
    theta: np.ndarray
    n_iters: int
    converged: bool
    loss_history: list = field(default_factory=list)
    theta_history: list = field(default_factory=list)
    grad_norm_history: list = field(default_factory=list)


def fit_gradient_descent(
    X,
    y,
    lr=0.01,
    max_iters=10_000,
    tol=1e-8,
    theta_init=None,
    record_trajectory=False,
):
    """Vanilla batch GD.

    record_trajectory=True keeps theta and loss at every step (for plotting).
    Set False for benchmarking to avoid memory/time overhead.
    """
    n, p = X.shape
    theta = np.zeros(p) if theta_init is None else theta_init.copy()

    loss_hist, theta_hist, grad_hist = [], [], []
    converged = False

    for it in range(max_iters):
        residual = X @ theta - y
        grad = (X.T @ residual) / n

        if record_trajectory:
            loss_hist.append(0.5 * np.mean(residual ** 2))
            theta_hist.append(theta.copy())
            grad_hist.append(np.linalg.norm(grad))

        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            converged = True
            break

        theta = theta - lr * grad

    return GDResult(
        theta=theta,
        n_iters=it + 1,
        converged=converged,
        loss_history=loss_hist,
        theta_history=theta_hist,
        grad_norm_history=grad_hist,
    )


def safe_lr(X):
    """A theoretically safe step size for OLS GD: 1 / L where L is the largest
    eigenvalue of (1/n) X^T X. Below this, GD is guaranteed to converge."""
    n = X.shape[0]
    sigma_max = np.linalg.svd(X, compute_uv=False).max()
    L = (sigma_max ** 2) / n
    return 1.0 / L
