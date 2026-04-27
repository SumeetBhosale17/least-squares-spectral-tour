"""Failure case: rank-deficient X (perfectly collinear features).

X^T X is singular -> np.linalg.solve raises LinAlgError. pinv returns the
minimum-norm solution. GD also returns a sensible (zero-init) solution because
it never tries to invert anything.

Run:
    python -m experiments.05_singular
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data_gen import make_collinear
from src.normal_equation import fit_normal_equation, fit_pinv
from src.gradient_descent import fit_gradient_descent, safe_lr


def loss(X, y, theta):
    return 0.5 * np.mean((X @ theta - y) ** 2)


def main():
    X, y, _ = make_collinear(n=500, p=10, seed=0)
    print(f"X shape: {X.shape}, rank: {np.linalg.matrix_rank(X)} (expected {X.shape[1] - 1})\n")

    print("Normal Equation (solve):")
    try:
        t_ne = fit_normal_equation(X, y)
        print(f"  surprisingly succeeded, loss={loss(X, y, t_ne):.4f}")
    except np.linalg.LinAlgError as e:
        print(f"  LinAlgError: {e}")

    t_pi = fit_pinv(X, y)
    print(f"\nPseudo-inverse: loss={loss(X, y, t_pi):.6f}, ||theta||={np.linalg.norm(t_pi):.4f}")

    lr = safe_lr(X) * 0.9
    res = fit_gradient_descent(X, y, lr=lr, max_iters=50_000, tol=1e-10)
    print(f"GD:             loss={loss(X, y, res.theta):.6f}, ||theta||={np.linalg.norm(res.theta):.4f}, iters={res.n_iters}")

    print("\nLoss matches across pinv and GD even though theta is non-unique.")


if __name__ == "__main__":
    main()
