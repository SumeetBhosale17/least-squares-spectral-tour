# Least-Squares Spectral Tour

A reproducible tour of seven least-squares solvers — Normal Equation,
Cholesky, QR, SVD, Gradient Descent, Conjugate Gradient, LSMR, L-BFGS —
with the textbook **GD-vs-NE binary** dismantled along the way.

> **Read the blog post:** [`docs/blog/post.md`](docs/blog/post.md)
> **Live site:** *(coming soon — GitHub Pages URL after first deploy)*

---

## Three findings the repo defends

1. **The κ² penalty.** At κ(X)=10⁸ with zero noise, `solve(XᵀX, Xᵀy)` gives
   relative error ~10⁻¹; QR / lstsq give ~10⁻⁹. **Eight orders of
   magnitude.** The textbook recipe is malpractice.
2. **GD-vs-NE is a false dichotomy.** For ill-conditioned iterative least
   squares, use **Conjugate Gradient** (`√κ` instead of `κ` iterations).
   For sparse, use **LSMR**. Batch GD is rarely the right production tool.
3. **Early-stopped GD is implicit ridge regression** with effective
   λ = 1/(η·k). Same mechanism behind SGD's implicit regularization in
   deep learning.

Full derivations, twelve figures, eight experiments — in the
[blog post](docs/blog/post.md).

---

## Reproducing

```bash
pip install -r requirements.txt
python -m experiments.00_run_all     # regenerate every figure (~5 min)
```

Figures land in `docs/figures/`. Each `experiments/*.py` script can also
be run individually — each prints its numbers to stdout.

## Building the site locally

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

## Repo layout

```
src/             four exact solvers + GD + helpers
experiments/     nine reproducible scripts (numbered 01-09)
notebooks/       interactive walkthrough.ipynb
docs/
  index.md       site homepage (long-form overview)
  blog/post.md   the full write-up
  figures/       all twelve PNGs
mkdocs.yml       site config (Material theme, MathJax)
```
