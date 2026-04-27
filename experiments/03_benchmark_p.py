"""Wall-clock benchmark: sweep p with n fixed.

Two experiments here:
  (A) NE (LU on XtX) vs Cholesky vs QR vs GD across the full p range.
      Uses fixed-iteration GD as a worst-case proxy.
  (B) The actual crossover hunt: extend p out far enough that the textbook
      O(p^3) cost beats GD's O(k n p) on a single laptop. We use Cholesky
      (fastest stable NE variant) and convergence-based GD.

Caveats: results are hardware-dependent (CPU, BLAS implementation, threading,
caches). Sparse X or GPU execution would shift the crossover dramatically.

Run:
    python -m experiments.03_benchmark_p
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.data_gen import make_regression, standardize
from src.normal_equation import fit_normal_equation, fit_cholesky, fit_qr
from src.gradient_descent import fit_gradient_descent, safe_lr
from src.utils import time_call, setup_plot_style, savefig


def part_A_solver_family(n=5000):
    """All four solvers, fixed-iteration GD."""
    p_grid = [10, 50, 100, 250, 500, 1000, 2000, 4000]
    out = {"NE (LU on XtX)": [], "Cholesky": [], "QR": [], "GD (1000 iters)": []}
    err = {k: [] for k in out}

    print(f"Part A: solver family across p (n={n}, GD fixed at 1000 iters)\n")
    print(f"{'p':>5}  {'NE':>10}  {'Chol':>10}  {'QR':>10}  {'GD':>12}")
    for p in p_grid:
        X, y, _ = make_regression(n=n, p=p, seed=0)
        X, _, _ = standardize(X)
        lr = safe_lr(X) * 0.9

        t_ne, s_ne, _ = time_call(fit_normal_equation, X, y, repeats=3)
        t_ch, s_ch, _ = time_call(fit_cholesky, X, y, repeats=3)
        t_qr, s_qr, _ = time_call(fit_qr, X, y, repeats=3)
        t_gd, s_gd, _ = time_call(fit_gradient_descent, X, y,
                                  lr=lr, max_iters=1000, tol=1e-8, repeats=3)
        out["NE (LU on XtX)"].append((t_ne, s_ne))
        out["Cholesky"].append((t_ch, s_ch))
        out["QR"].append((t_qr, s_qr))
        out["GD (1000 iters)"].append((t_gd, s_gd))
        print(f"{p:>5}  {t_ne*1e3:8.2f} ms  {t_ch*1e3:8.2f} ms  {t_qr*1e3:8.2f} ms  {t_gd*1e3:10.2f} ms")

    fig, ax = plt.subplots()
    for name, vals in out.items():
        ts = np.array([v[0] for v in vals])
        ss = np.array([v[1] for v in vals])
        ax.errorbar(p_grid, ts * 1e3, yerr=ss * 1e3, marker="o", label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("p (features)"); ax.set_ylabel("wall time (ms)")
    ax.set_title(f"Solver family wall time vs p (n={n})")
    ax.legend(fontsize=9)
    savefig("wallclock_vs_p_family.png")


def part_B_crossover_hunt(n=10000):
    """Push p far enough to find the actual crossover. Cholesky vs
    convergence-based GD on standardized data. Keep n > p so X^T X is PD."""
    p_grid = [500, 1000, 2000, 4000, 8000]
    t_ch, t_gd, iters_gd = [], [], []

    print(f"\nPart B: crossover hunt (n={n}, GD runs until convergence)\n")
    print(f"{'p':>6}  {'Chol':>10}  {'GD':>12}  {'GD iters':>10}")
    for p in p_grid:
        X, y, _ = make_regression(n=n, p=p, seed=0)
        X, _, _ = standardize(X)
        lr = safe_lr(X) * 0.9

        try:
            t1, _, _ = time_call(fit_cholesky, X, y, repeats=2)
        except np.linalg.LinAlgError:
            t1 = np.nan
        t2, _, res = time_call(fit_gradient_descent, X, y,
                               lr=lr, max_iters=5000, tol=1e-6, repeats=2)
        t_ch.append(t1); t_gd.append(t2); iters_gd.append(res.n_iters)
        print(f"{p:>6}  {t1*1e3 if not np.isnan(t1) else float('nan'):8.2f} ms  "
              f"{t2*1e3:10.2f} ms  {res.n_iters:>10}")

    crossover = None
    for i in range(1, len(p_grid)):
        if t_gd[i - 1] > t_ch[i - 1] and t_gd[i] < t_ch[i]:
            crossover = p_grid[i]; break
        if t_ch[i - 1] < t_gd[i - 1] and t_ch[i] > t_gd[i]:
            crossover = p_grid[i]; break

    fig, ax = plt.subplots()
    ax.plot(p_grid, np.array(t_ch) * 1e3, marker="o", label="Cholesky on XtX")
    ax.plot(p_grid, np.array(t_gd) * 1e3, marker="s", label="GD (until grad-norm 1e-6)")
    ax.set_xscale("log"); ax.set_yscale("log")
    if crossover is not None:
        ax.axvline(crossover, color="grey", linestyle="--", alpha=0.6)
        ax.text(crossover, max(max(t_ch), max(t_gd)) * 500,
                f"  crossover ~ p={crossover}", color="grey", fontsize=9)
    ax.set_xlabel("p (features)"); ax.set_ylabel("wall time (ms)")
    ax.set_title(f"Crossover hunt (n={n}, hardware-dependent)")
    ax.legend()
    savefig("wallclock_vs_p_crossover.png")

    if crossover is None:
        print("\nNo crossover observed in this p range -- on this hardware, "
              "Cholesky NE wins out to p={}. Push p higher or use sparse X "
              "to find one.".format(p_grid[-1]))
    else:
        print(f"\nCrossover found at p ~ {crossover}.")


def main():
    setup_plot_style()
    part_A_solver_family()
    part_B_crossover_hunt()


if __name__ == "__main__":
    main()
