"""Sanity check: GD and Normal Equation converge to the same theta on
well-conditioned data.

Run from the repo root:
    python -m experiments.01_correctness
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data_gen import make_regression, standardize
from src.normal_equation import fit_normal_equation
from src.gradient_descent import fit_gradient_descent, safe_lr


def main():
    X, y, theta_true = make_regression(n=2000, p=20, noise=0.1, seed=42)
    X_s, _, _ = standardize(X)

    theta_ne = fit_normal_equation(X_s, y)
    res = fit_gradient_descent(X_s, y, lr=safe_lr(X_s) * 0.9, max_iters=50_000, tol=1e-12)
    theta_gd = res.theta

    diff = np.linalg.norm(theta_gd - theta_ne)
    rel = diff / np.linalg.norm(theta_ne)

    print(f"GD iterations:           {res.n_iters} (converged={res.converged})")
    print(f"||theta_GD - theta_NE||: {diff:.2e}")
    print(f"relative diff:           {rel:.2e}")
    print(f"||theta_NE - theta_true||: {np.linalg.norm(theta_ne - theta_true):.4f}")
    assert rel < 1e-4, "GD and NE diverged — something is broken."
    print("\nOK: GD matches Normal Equation.")


if __name__ == "__main__":
    main()
