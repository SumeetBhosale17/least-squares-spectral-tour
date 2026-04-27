"""Regenerate every figure and reprint every numeric finding in one shot.

Run from the repo root:
    python -m experiments.00_run_all
"""

import sys, os, time, importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPERIMENTS = [
    "experiments.01_correctness",
    "experiments.02_benchmark_n",
    "experiments.03_benchmark_p",
    "experiments.04_conditioning",
    "experiments.05_singular",
    "experiments.06_scaling_effect",
    "experiments.07_memory_profile",
    "experiments.08_sparse",
    "experiments.09_cg_lsmr",
]


def main():
    grand_t0 = time.perf_counter()
    for name in EXPERIMENTS:
        print("\n" + "=" * 72)
        print(f"  RUNNING: {name}")
        print("=" * 72)
        t0 = time.perf_counter()
        mod = importlib.import_module(name)
        mod.main()
        print(f"  -> done in {time.perf_counter() - t0:.1f} s")

    print("\n" + "=" * 72)
    print(f"All experiments completed in {time.perf_counter() - grand_t0:.1f} s")
    print("Figures written to ./docs/figures/")
    print("=" * 72)


if __name__ == "__main__":
    main()
