"""Peak memory: NE has to materialize a p x p matrix (X^T X). At p ~ 30k that
matrix alone is ~7 GB in float64. GD never builds it.

Run:
    python -m experiments.07_memory_profile
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
from src.data_gen import make_regression, standardize
from src.normal_equation import fit_normal_equation
from src.gradient_descent import fit_gradient_descent, safe_lr
from src.utils import setup_plot_style, savefig


def measure(fn, *args, **kwargs):
    """Peak Python-level allocation during the call, in MB."""
    gc.collect()
    tracemalloc.start()
    fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 ** 2)


def main():
    setup_plot_style()
    n = 5000
    p_grid = [100, 250, 500, 1000, 2000, 4000, 6000]
    mem_ne, mem_gd, theory = [], [], []

    for p in p_grid:
        X, y, _ = make_regression(n=n, p=p, seed=0)
        X, _, _ = standardize(X)
        lr = safe_lr(X) * 0.9

        m_ne = measure(fit_normal_equation, X, y)
        m_gd = measure(fit_gradient_descent, X, y, lr=lr, max_iters=500, tol=1e-8)
        t = (p * p * 8) / (1024 ** 2)  # X^T X in MB, float64
        mem_ne.append(m_ne); mem_gd.append(m_gd); theory.append(t)
        print(f"p={p:>5}  peak alloc  NE={m_ne:8.1f} MB   GD={m_gd:8.1f} MB   theoretical X^T X={t:8.1f} MB")

    fig, ax = plt.subplots()
    ax.plot(p_grid, mem_ne, marker="o", label="NE peak allocation")
    ax.plot(p_grid, mem_gd, marker="s", label="GD peak allocation")
    ax.plot(p_grid, theory, marker="x", linestyle="--", label=r"theoretical $p^2 \cdot 8$ bytes")
    ax.set_xlabel("p (features)"); ax.set_ylabel("memory (MB)")
    ax.set_title(f"Peak memory vs p  (n={n})")
    ax.legend()
    savefig("memory_vs_p.png")


if __name__ == "__main__":
    main()
