"""Conjugate Gradient and LSMR vs Gradient Descent on ill-conditioned data.

The classical answer to "GD is slow when kappa is bad" is CG: it converges
in O(sqrt(kappa)) iterations instead of O(kappa). LSMR is the production-grade
Krylov method for sparse least squares.

This experiment runs all three on synthetic regression with controlled kappa
and reports iteration counts to a fixed accuracy.

Run:
    python -m experiments.09_cg_lsmr
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from src.data_gen import make_regression
from src.gradient_descent import fit_gradient_descent, safe_lr
from src.utils import setup_plot_style, savefig


def cg_normal_eq(X, y, tol=1e-6, max_iters=20_000):
    """Conjugate Gradient on the normal equations (X^T X) theta = X^T y.
    We never form X^T X -- we use X.T @ (X @ p) inside the matvec."""
    n, p = X.shape
    A = spla.LinearOperator(
        shape=(p, p), matvec=lambda v: X.T @ (X @ v), dtype=X.dtype
    )
    b = X.T @ y
    iters = [0]

    def cb(_):
        iters[0] += 1

    theta, info = spla.cg(A, b, rtol=tol, maxiter=max_iters, callback=cb)
    return theta, iters[0]


def lsmr_solve(X, y, tol=1e-6, max_iters=20_000):
    """LSMR works directly on X (no Gram matrix). Returns iteration count."""
    out = spla.lsmr(X, y, atol=tol, btol=tol, maxiter=max_iters)
    theta = out[0]
    iters = out[2]
    return theta, iters


def main():
    setup_plot_style()
    n, p = 1000, 50
    kappa_grid = np.logspace(1, 6, 6)
    iters_gd, iters_cg, iters_lsmr = [], [], []
    sqrt_kappa = []

    print(f"{'kappa':>8}  {'sqrt(kappa)':>11}  {'GD iters':>9}  {'CG iters':>9}  {'LSMR iters':>11}")
    for kappa in kappa_grid:
        X, y, _ = make_regression(n=n, p=p, condition_number=kappa, seed=0)
        lr = safe_lr(X) * 0.9

        res = fit_gradient_descent(X, y, lr=lr, max_iters=50_000, tol=1e-6)
        _, n_cg = cg_normal_eq(X, y, tol=1e-6, max_iters=50_000)
        _, n_lsmr = lsmr_solve(X, y, tol=1e-6, max_iters=50_000)

        iters_gd.append(res.n_iters)
        iters_cg.append(n_cg)
        iters_lsmr.append(n_lsmr)
        sqrt_kappa.append(np.sqrt(kappa))
        print(f"{kappa:8.0e}  {np.sqrt(kappa):11.1f}  {res.n_iters:>9}  {n_cg:>9}  {n_lsmr:>11}")

    fig, ax = plt.subplots()
    ax.loglog(kappa_grid, iters_gd, marker="o", label="Gradient Descent (~kappa)")
    ax.loglog(kappa_grid, iters_cg, marker="s", label="Conjugate Gradient (~sqrt(kappa))")
    ax.loglog(kappa_grid, iters_lsmr, marker="^", label="LSMR (~sqrt(kappa))")
    ax.loglog(kappa_grid, kappa_grid, linestyle=":", color="grey", alpha=0.6, label=r"$\kappa$ reference")
    ax.loglog(kappa_grid, sqrt_kappa, linestyle="--", color="grey", alpha=0.6, label=r"$\sqrt{\kappa}$ reference")
    ax.set_xlabel(r"condition number $\kappa(X)$")
    ax.set_ylabel("iterations to gradient norm 1e-6")
    ax.set_title("CG and LSMR scale with sqrt(kappa); GD scales with kappa")
    ax.legend(fontsize=9)
    savefig("cg_lsmr_vs_gd.png")


if __name__ == "__main__":
    main()
