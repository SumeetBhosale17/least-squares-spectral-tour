# A Spectral Tour of Least-Squares Solvers

The textbook frames linear regression as **Gradient Descent vs Normal Equation**.
That binary is a pedagogical artifact. The real candidate set is seven
algorithms — Normal Equation, Cholesky, QR, SVD, Gradient Descent,
Conjugate Gradient, LSMR, and L-BFGS — and the right one depends on the
**singular-value spectrum of X**, not on which method your textbook reached
for first.

This repo is the reproducible companion to the blog post. It builds the
GD-vs-NE comparison the long way (so you can see exactly where the binary
breaks), then walks out into the broader landscape.

> Full write-up: [`blog/post.md`](blog/post.md).
> Code, eight experiments, twelve figures, all reproducible from
> `requirements.txt`.

---

## The unifying principle

Every result in this repo reduces to **what the algorithm does to the
singular values of X** (equivalently, the eigenvalues of `H = (1/n) XᵀX`,
which are `σᵢ²/n`):

| Phenomenon | Where it lives in the spectrum |
|---|---|
| GD convergence rate | `σ_max / σ_min` — that is, κ |
| GD optimal step size | `2 / (L + μ) = 2n / (σ_max² + σ_min²)` |
| NE / Cholesky accuracy | scales as `1/σ_min²` (the κ² penalty) |
| QR / lstsq accuracy | scales as `1/σ_min` only |
| Standardization | shrinks the σᵢ spread (κ down) |
| Ridge regression | replaces `σᵢ²` with `σᵢ² + λ` |
| Early-stopped GD | per-direction shrinkage `[1 − (1 − ησᵢ²/n)^k]` — same shape as ridge |
| Conjugate Gradient | converges in `√κ` steps, not κ |
| SGD | rate driven by `Tr(H)/μ`, not just κ |

If you can answer *"what does this do to the singular values of X?"* you
can predict everything else.

---

## TL;DR — the seven solvers, with the headline finding for each

| Solver | When it wins | Killer fact |
|---|---|---|
| **Normal Equation (LU on XᵀX)** | Never as the default. | At κ(X)=10⁸, error is ~10⁻¹ vs QR's ~10⁻⁹ — eight orders of magnitude. |
| **Cholesky on XᵀX** | Small dense p, well-conditioned, you trust your data. | Beats convergence-based GD by ~5× across every p tested up to 12,000. |
| **QR on X** | The right hand-rolled exact solver. | Avoids the κ² penalty entirely. |
| **SVD / `lstsq`** | Suspected rank deficiency or collinearity. | What scikit-learn ships. |
| **Gradient Descent** | Sparse / streaming / memory-bound. | **16 → 200,000+ iterations** from one bad column scale. |
| **Conjugate Gradient** | Anywhere you'd reach for GD on a least-squares problem. | `√κ` instead of `κ` iterations. Measured: 1,050 vs ~9,000 at κ=10⁶. |
| **LSMR / LSQR** | Sparse least squares, in production. | SciPy ships it. Beats both dense NE and sparse GD. |

Plus: **L-BFGS** for dense smooth non-quadratic losses, **SGD** for
`n = 10⁸`-scale ML training. The blog covers both.

---

## The three findings the rest of the repo defends

1. **The κ² penalty is the most under-taught fact in linear regression.**
   At κ(X)=10⁸ with zero noise, NE / Cholesky give relative error ~10⁻¹;
   QR / lstsq give ~10⁻⁹. The textbook recipe `solve(X.T @ X, X.T @ y)` is
   malpractice. Use QR or SVD.

2. **The GD-vs-NE binary is a false dichotomy.** For ill-conditioned
   iterative least squares, **use Conjugate Gradient** (`√κ` instead of
   `κ`). For sparse least squares, **use LSMR**. Batch GD is rarely the
   right production tool — it's pedagogically central because it's the
   cleanest place to see scaling, conditioning, and implicit
   regularization at work.

3. **Early-stopped GD is implicit ridge regression** with effective
   λ = 1/(η·k). This is why GD's recovery error stays bounded as κ grows
   while every closed-form solver explodes. Path- and init-dependent —
   not a drop-in for ridge in a reproducible pipeline. The same mechanism
   underlies SGD's implicit regularization in deep learning.

---

## Headline plots

- `figures/iters_vs_scale_spread.png` — GD on standardized data converges
  in ~25 iterations regardless of feature spread. GD on raw data needs
  200k+ once the scale ratio crosses ~30.
- `figures/condition_number_vs_error_noiseless.png` — the κ² penalty,
  isolated. NE / Cholesky climb at slope 2 in log-log; QR / lstsq at
  slope 1. At κ=10⁸ the slopes are 8 decades apart.
- `figures/loss_contours_scaled_vs_unscaled.png` — the geometric reason.
  Scaling turns long ellipses into circles.
- `figures/gd_trajectory_overlay.png` — gradient-orthogonal-to-contour
  made visible: GD zig-zags across the narrow valley on unscaled data,
  walks straight on standardized data.
- `figures/cg_lsmr_vs_gd.png` — CG's iteration count tracking the `√κ`
  reference line; GD tracking κ.
- `figures/sparse_benchmark.png` — where GD legitimately beats NE.
- `figures/wallclock_vs_p_crossover.png` — where it doesn't.
- `figures/memory_vs_p.png` — NE's `O(p²)` allocation tracked by
  tracemalloc to within fractions of a MB.

---

## Repo layout

```
src/
  data_gen.py           synthetic regression with knobs for n, p, scaling, kappa
  normal_equation.py    four exact solvers: LU on XtX, Cholesky on XtX, QR on X, lstsq (SVD)
  gradient_descent.py   vanilla batch GD with trajectory recording, safe_lr() helper
  utils.py              timing, conditioning, plotting helpers
experiments/
  00_run_all.py         single-command regenerate-everything
  01_correctness.py     all five methods agree to ~1e-13
  02_benchmark_n.py     wall-clock vs n
  03_benchmark_p.py     wall-clock vs p (solver family + crossover hunt)
  04_conditioning.py    kappa^2 penalty, with-noise and noiseless variants
  05_singular.py        rank-deficient X
  06_scaling_effect.py  contours, trajectories, iters-vs-scale-spread
  07_memory_profile.py  tracemalloc shows NE = p^2 * 8 exactly
  08_sparse.py          sparse X: where GD legitimately wins
  09_cg_lsmr.py         CG and LSMR vs GD: sqrt(kappa) vs kappa scaling
figures/                all PNGs, regenerated by experiments
notebooks/walkthrough.ipynb   interactive companion
blog/post.md            full write-up
```

---

## Reproducing

```bash
pip install -r requirements.txt
python -m experiments.00_run_all     # regenerate every figure (~5 min)
```

Or run any experiment individually — each prints its numbers to stdout
and writes its figure to `figures/`.

---

## Decision flowchart (from the blog)

```
Production code, dense X, p < ~10k → sklearn.linear_model.LinearRegression (uses lstsq).

Want a hand-rolled exact solver?
  Well-conditioned dense          → QR on X.
  Suspect rank deficiency         → SVD-based lstsq with explicit rcond.
  Don't form X^T X by default. Don't use solve() on the Gram matrix.

Need an iterative method?
  Dense or sparse, ill-conditioned → Conjugate Gradient (CGNR / CGLS).
                                     ~sqrt(kappa) iterations vs GD's kappa.
  Sparse, large p, exact-flavor    → LSMR / LSQR (scipy.sparse.linalg).
  Smooth non-quadratic loss        → L-BFGS.
  ML-style training, n huge        → SGD or mini-batch with a schedule.
  L1 / elastic-net                 → coordinate descent or proximal GD.

Always: standardize. It's the cheap diagonal preconditioner that makes every
iterative method finish in a reasonable number of steps.
```

See the [blog post](blog/post.md) for derivations, geometry, the spectral
unifying principle (§1.5), the CG / LSMR / L-BFGS landscape (§7), the
early-stopping ≈ ridge equivalence (§8), the SGD distinction (§9), and
references.
