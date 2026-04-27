"""Scaled vs unscaled features: the geometry of the loss surface and how it
breaks (or doesn't break) each solver.

Math intuition
--------------
For OLS, the Hessian is H = (1/n) X^T X. Its eigenvalues set the curvature
along each axis of the loss bowl. If feature columns have wildly different
scales, H's eigenvalues span many orders of magnitude -> the bowl is a long,
narrow ellipsoid. GD with a single learning rate must use the *smallest*
safe step (set by the largest eigenvalue), then crawl along the flat
direction (set by the smallest eigenvalue). Convergence rate ~ kappa.

The Normal Equation, in *exact* arithmetic, is scale-invariant -- it just
solves a linear system. But the same kappa that hurts GD also hurts the
numerical accuracy of solve().

This script produces three figures:
  1. loss contours, scaled vs unscaled (the geometric intuition)
  2. GD trajectories overlaid on those contours
  3. iterations-to-converge as feature scale spread grows

Run:
    python -m experiments.06_scaling_effect
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from src.data_gen import make_regression, standardize
from src.gradient_descent import fit_gradient_descent, safe_lr
from src.normal_equation import fit_normal_equation
from src.utils import setup_plot_style, savefig, condition_number


def loss_grid(X, y, theta_star, span=3.0, n=120):
    """Compute L(theta) on a 2D grid centered at theta_star."""
    t1 = np.linspace(theta_star[0] - span, theta_star[0] + span, n)
    t2 = np.linspace(theta_star[1] - span, theta_star[1] + span, n)
    T1, T2 = np.meshgrid(t1, t2)
    L = np.zeros_like(T1)
    for i in range(n):
        for j in range(n):
            r = X @ np.array([T1[i, j], T2[i, j]]) - y
            L[i, j] = 0.5 * np.mean(r ** 2)
    return T1, T2, L


def fig1_contours():
    """Side-by-side contours: unscaled (skewed) vs scaled (round)."""
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n) * 8.0
    X_u = np.column_stack([x1, x2])
    theta_true = np.array([2.0, -1.0])
    y = X_u @ theta_true + 0.05 * rng.standard_normal(n)
    X_s, _, _ = standardize(X_u)

    theta_u = fit_normal_equation(X_u, y)
    theta_s = fit_normal_equation(X_s, y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, X, ts, title in [
        (axes[0], X_u, theta_u, f"Unscaled  ($\\kappa$={condition_number(X_u):.1e})"),
        (axes[1], X_s, theta_s, f"Standardized  ($\\kappa$={condition_number(X_s):.1e})"),
    ]:
        T1, T2, L = loss_grid(X, y, ts, span=2.5, n=140)
        ax.contour(T1, T2, np.log10(L + 1e-12), levels=22, cmap="viridis")
        ax.plot(*ts, "*", color="red", markersize=18,
                markeredgecolor="white", label=r"$\theta^*$")
        ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
        ax.set_title(title)
        ax.set_xlim(ts[0] - 2.7, ts[0] + 2.7)
        ax.set_ylim(ts[1] - 2.7, ts[1] + 2.7)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Loss contours: scaling turns ellipses into circles",
                 y=1.00, fontsize=13)
    fig.tight_layout()
    savefig("loss_contours_scaled_vs_unscaled.png")
    plt.close(fig)


def fig2_trajectories():
    """GD trajectories on unscaled vs scaled data. Both panels use box aspect
    so the unscaled zig-zag isn't compressed by matplotlib's auto-scaling."""
    rng = np.random.default_rng(1)
    n = 400
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n) * 6.0
    X_u = np.column_stack([x1, x2])
    theta_true = np.array([1.5, -0.8])
    y = X_u @ theta_true + 0.05 * rng.standard_normal(n)
    X_s, _, _ = standardize(X_u)

    theta_u = fit_normal_equation(X_u, y)
    theta_s = fit_normal_equation(X_s, y)

    init_u = theta_u + np.array([2.0, 2.0])
    init_s = theta_s + np.array([2.0, 2.0])

    res_u = fit_gradient_descent(X_u, y, lr=safe_lr(X_u) * 0.9,
                                 max_iters=200, theta_init=init_u,
                                 record_trajectory=True, tol=0)
    res_s = fit_gradient_descent(X_s, y, lr=safe_lr(X_s) * 0.9,
                                 max_iters=200, theta_init=init_s,
                                 record_trajectory=True, tol=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    panels = [
        (axes[0], X_u, theta_u, res_u, f"Unscaled  (zig-zag, {res_u.n_iters} iters)"),
        (axes[1], X_s, theta_s, res_s, f"Scaled  (straight, {res_s.n_iters} iters)"),
    ]
    for ax, X, ts, res, title in panels:
        T1, T2, L = loss_grid(X, y, ts, span=3.0, n=140)
        ax.contour(T1, T2, np.log10(L + 1e-12), levels=22, cmap="viridis", alpha=0.7)
        traj = np.array(res.theta_history)
        ax.plot(traj[:, 0], traj[:, 1], "o-", color="crimson",
                markersize=3, linewidth=1.2, alpha=0.85, label="GD path")
        ax.plot(traj[0, 0], traj[0, 1], "o", color="white",
                markeredgecolor="crimson", markersize=10, label="start")
        ax.plot(*ts, "*", color="black", markersize=18,
                markeredgecolor="white", label=r"$\theta^*$")
        ax.set_xlabel(r"$\theta_1$"); ax.set_ylabel(r"$\theta_2$")
        ax.set_title(title)
        ax.set_xlim(ts[0] - 3.2, ts[0] + 3.2)
        ax.set_ylim(ts[1] - 3.2, ts[1] + 3.2)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("GD trajectory: scaling decides whether you walk straight or zig-zag",
                 y=1.00, fontsize=13)
    fig.tight_layout()
    savefig("gd_trajectory_overlay.png")
    plt.close(fig)


def fig3_iters_vs_scale_spread():
    """As feature-scale spread grows, GD on unscaled data needs many more
    iterations. GD on scaled data stays flat. NE timing barely moves."""
    spreads = np.logspace(0, 5, 11)
    iters_unscaled, iters_scaled = [], []
    n, p = 1000, 10

    for spread in spreads:
        scales = np.logspace(0, np.log10(spread), p)
        X, y, _ = make_regression(n=n, p=p, seed=0, feature_scales=scales)
        X_s, _, _ = standardize(X)

        res_u = fit_gradient_descent(X, y, lr=safe_lr(X) * 0.9,
                                     max_iters=200_000, tol=1e-6)
        res_s = fit_gradient_descent(X_s, y, lr=safe_lr(X_s) * 0.9,
                                     max_iters=200_000, tol=1e-6)
        iters_unscaled.append(res_u.n_iters)
        iters_scaled.append(res_s.n_iters)
        print(f"spread={spread:.0e}  unscaled={res_u.n_iters:>7}  scaled={res_s.n_iters:>5}")

    fig, ax = plt.subplots()
    ax.loglog(spreads, iters_unscaled, marker="o", label="GD on unscaled X")
    ax.loglog(spreads, iters_scaled, marker="s", label="GD on standardized X")
    ax.set_xlabel("max/min feature scale ratio")
    ax.set_ylabel("iterations to reach grad-norm 1e-6")
    ax.set_title("Scaling collapses convergence cost by orders of magnitude")
    ax.legend()
    savefig("iters_vs_scale_spread.png")
    plt.close(fig)


def main():
    setup_plot_style()
    print("Figure 1: loss contours")
    fig1_contours()
    print("\nFigure 2: GD trajectories")
    fig2_trajectories()
    print("\nFigure 3: iterations vs scale spread")
    fig3_iters_vs_scale_spread()


if __name__ == "__main__":
    main()
