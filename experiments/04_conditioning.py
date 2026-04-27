"""Failure case: ill-conditioned X.

The headline finding here is the kappa^2 penalty. fit_normal_equation and
fit_cholesky both build X^T X, which doubles the effective condition number.
fit_qr and fit_lstsq operate on X directly and pay only kappa, not kappa^2.
GD with early stopping is implicitly regularized -- it never has time to
amplify noise along the small singular directions.

Run:
    python -m experiments.04_conditioning
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.data_gen import make_regression
from src.normal_equation import (
    fit_normal_equation, fit_cholesky, fit_qr, fit_lstsq,
)
from src.gradient_descent import fit_gradient_descent, safe_lr
from src.utils import setup_plot_style, savefig


GD_MAX_ITERS = 20_000
GD_TOL = 1e-10


def safe_call(fn, X, y):
    try:
        return fn(X, y)
    except np.linalg.LinAlgError:
        return None


def run_sweep(noise, label, savename):
    n, p = 1000, 30
    kappa_grid = np.logspace(1, 9, 9)
    series = {"NE (LU on XtX)": [], "Cholesky on XtX": [], "QR on X": [],
              "lstsq (SVD)": [], f"GD (early stop, max {GD_MAX_ITERS} iters)": []}

    print(f"\n=== {label}  (noise={noise}) ===")
    print(f"GD hyperparameters: lr=safe_lr(X)*0.5, max_iters={GD_MAX_ITERS}, tol={GD_TOL}")
    header = f"{'kappa':>10}  {'NE':>10}  {'Chol':>10}  {'QR':>10}  {'lstsq':>10}  {'GD':>10}"
    print(header)
    print("-" * len(header))

    for kappa in kappa_grid:
        X, y, theta_true = make_regression(n=n, p=p, noise=noise, condition_number=kappa, seed=0)

        for name, fn in [
            ("NE (LU on XtX)", fit_normal_equation),
            ("Cholesky on XtX", fit_cholesky),
            ("QR on X", fit_qr),
            ("lstsq (SVD)", fit_lstsq),
        ]:
            theta = safe_call(fn, X, y)
            err = np.linalg.norm(theta - theta_true) if theta is not None else np.nan
            series[name].append(err)

        lr = safe_lr(X) * 0.5
        res = fit_gradient_descent(X, y, lr=lr, max_iters=GD_MAX_ITERS, tol=GD_TOL)
        gd_err = np.linalg.norm(res.theta - theta_true)
        series[f"GD (early stop, max {GD_MAX_ITERS} iters)"].append(gd_err)

        print(f"{kappa:10.0e}  "
              f"{series['NE (LU on XtX)'][-1]:10.2e}  "
              f"{series['Cholesky on XtX'][-1]:10.2e}  "
              f"{series['QR on X'][-1]:10.2e}  "
              f"{series['lstsq (SVD)'][-1]:10.2e}  "
              f"{gd_err:10.2e}")

    fig, ax = plt.subplots()
    markers = {"NE (LU on XtX)": "o", "Cholesky on XtX": "v", "QR on X": "s",
               "lstsq (SVD)": "D", f"GD (early stop, max {GD_MAX_ITERS} iters)": "^"}
    for name, errs in series.items():
        ax.loglog(kappa_grid, errs, marker=markers[name], label=name)
    ax.set_xlabel(r"condition number $\kappa(X)$")
    ax.set_ylabel(r"$\|\hat\theta - \theta^*\|_2$")
    ax.set_title(label)
    ax.legend(fontsize=8)
    savefig(savename)


def main():
    setup_plot_style()
    run_sweep(noise=0.1, label="Recovery error vs kappa (with noise)",
              savename="condition_number_vs_error.png")
    run_sweep(noise=0.0, label="Floating-point error vs kappa (zero noise) -- "
                               "the kappa^2 penalty isolated",
              savename="condition_number_vs_error_noiseless.png")


if __name__ == "__main__":
    main()
