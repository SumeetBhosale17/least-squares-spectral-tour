"""Wall-clock benchmark: sweep n_samples with p fixed.

NE complexity: O(n p^2 + p^3). With small p, NE stays cheap as n grows
(only the X^T X build scales with n). GD is O(k n p) per training run.

Run:
    python -m experiments.02_benchmark_n
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.data_gen import make_regression, standardize
from src.normal_equation import fit_normal_equation
from src.gradient_descent import fit_gradient_descent, safe_lr
from src.utils import time_call, setup_plot_style, savefig


def main():
    setup_plot_style()
    p = 50
    n_grid = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000]
    rows = []

    for n in n_grid:
        X, y, _ = make_regression(n=n, p=p, seed=0)
        X, _, _ = standardize(X)
        lr = safe_lr(X) * 0.9

        t_ne, s_ne, _ = time_call(fit_normal_equation, X, y, repeats=3)
        t_gd, s_gd, _ = time_call(
            fit_gradient_descent, X, y,
            lr=lr, max_iters=2000, tol=1e-8, repeats=3,
        )
        rows.append((n, t_ne, s_ne, t_gd, s_gd))
        print(f"n={n:>7}  NE={t_ne*1e3:8.2f} ms   GD(2000 it)={t_gd*1e3:8.2f} ms")

    arr = np.array(rows)
    ns, tne, sne, tgd, sgd = arr.T

    fig, ax = plt.subplots()
    ax.errorbar(ns, tne * 1e3, yerr=sne * 1e3, marker="o", label="Normal Equation")
    ax.errorbar(ns, tgd * 1e3, yerr=sgd * 1e3, marker="s", label="Gradient Descent (2000 iters)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n (samples)"); ax.set_ylabel("wall time (ms)")
    ax.set_title(f"Solver wall time vs n  (p={p} fixed)")
    ax.legend()
    savefig("wallclock_vs_n.png")


if __name__ == "__main__":
    main()
