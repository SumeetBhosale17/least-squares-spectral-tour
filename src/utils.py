"""Shared helpers: timing, conditioning, plotting style."""

import time
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def time_call(fn, *args, repeats=5, warmup=1, **kwargs):
    """Time a callable. Returns (mean_seconds, std_seconds, result_of_last_call)."""
    for _ in range(warmup):
        result = fn(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times)), result


def peak_memory_mb():
    """Resident set size in MB for the current process. Returns NaN if unavailable."""
    if not _HAS_PSUTIL:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def condition_number(X):
    """kappa(X^T X) via singular values."""
    s = np.linalg.svd(X, compute_uv=False)
    s = s[s > 0]
    return (s.max() / s.min()) ** 2


def setup_plot_style():
    """Consistent style for all figures in the repo."""
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.frameon": False,
    })


FIGDIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "figures",
)


def savefig(name):
    """Save current figure into the figures/ directory."""
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    plt.savefig(path)
    print(f"  saved -> {path}")
