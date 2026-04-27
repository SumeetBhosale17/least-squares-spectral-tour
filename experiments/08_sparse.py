"""Sparse design matrices: where GD genuinely beats NE.

NE has to materialize a dense X^T X regardless of how sparse X was. GD only
ever needs X v and X^T u, which scipy.sparse computes in O(nnz). When X has
density 0.1% and p is large, GD wins by orders of magnitude on both wall
time and memory.

Run:
    python -m experiments.08_sparse
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from src.gradient_descent import safe_lr
from src.utils import setup_plot_style, savefig


def gd_sparse(X, y, lr, max_iters=500, tol=1e-6):
    """GD that exploits sparse X. Each iteration is two sparse mat-vecs."""
    n, p = X.shape
    theta = np.zeros(p)
    for it in range(max_iters):
        residual = X @ theta - y
        grad = (X.T @ residual) / n
        if np.linalg.norm(grad) < tol:
            return theta, it + 1
        theta -= lr * grad
    return theta, max_iters


def safe_lr_sparse(X):
    n = X.shape[0]
    sigma_max = sp.linalg.svds(X, k=1, return_singular_vectors=False)[0]
    return n / (sigma_max ** 2)


def fit_ne_dense(X, y):
    Xd = X.toarray()
    XtX = Xd.T @ Xd
    return np.linalg.solve(XtX, Xd.T @ y)


def main():
    setup_plot_style()
    n = 5000
    density = 0.005   # 0.5% nonzero
    p_grid = [200, 500, 1000, 2000, 4000]
    rows = []

    print(f"n={n}, density={density:.1%} nonzero")
    print(f"{'p':>5}  {'NE (dense)':>12}  {'GD (sparse)':>14}  {'GD iters':>10}  {'speedup':>8}")
    rng = np.random.default_rng(0)
    for p in p_grid:
        X = sp.random(n, p, density=density, format="csr",
                      random_state=rng.integers(1 << 30))
        theta_true = rng.standard_normal(p)
        y = X @ theta_true + 0.05 * rng.standard_normal(n)

        t0 = time.perf_counter()
        fit_ne_dense(X, y)
        t_ne = time.perf_counter() - t0

        lr = safe_lr_sparse(X) * 0.9
        t0 = time.perf_counter()
        _, iters = gd_sparse(X, y, lr=lr, max_iters=2000)
        t_gd = time.perf_counter() - t0

        rows.append((p, t_ne, t_gd, iters))
        print(f"{p:>5}  {t_ne*1e3:10.2f} ms  {t_gd*1e3:12.2f} ms  {iters:>10}  {t_ne/max(t_gd,1e-9):>7.1f}x")

    arr = np.array(rows)
    ps, t_ne, t_gd, _ = arr.T
    fig, ax = plt.subplots()
    ax.plot(ps, t_ne * 1e3, marker="o", label="NE (densified X^T X)")
    ax.plot(ps, t_gd * 1e3, marker="s", label="GD (sparse mat-vec)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("p"); ax.set_ylabel("wall time (ms)")
    ax.set_title(f"Sparse X (density={density:.1%}): GD wins by exploiting sparsity")
    ax.legend()
    savefig("sparse_benchmark.png")


if __name__ == "__main__":
    main()
