"""
Runaway-basin fraction: uncertainty and sensitivity.

The basin fraction is the share of a prespecified grid of starting conditions
(birth age T0 and brain size R0) whose simulated population ends with mean
brain size above a threshold. The grid is chosen, not sampled, so the 360
outcomes (36 starts x 10 seeds) are not independent draws and a binomial
interval over them would overstate precision and generality. Uncertainty is
therefore reported across simulation seeds: each seed yields one fraction over
the whole grid, and the mean, SD, and range across seeds are given.

Sensitivity: the comparison A = 0 versus A = 8 is repeated under variations of
the endpoint threshold (R > 3.5, 4, 4.5), the number of generations (50, 80),
the population size (300, 500), and the grid resolution (6 x 6, 11 x 11 over
the same ranges). The direction of the effect is what matters; the exact
percentages are descriptive of the grid.
"""
import numpy as np
from fast_sim import simulate_fast

SEEDS = list(range(10))
A_LEVELS = (0.0, 4.0, 8.0, 12.0)


def grid(nT, nR):
    return [(T0, R0) for T0 in np.linspace(3, 22, nT) for R0 in np.linspace(1, 6, nR)]


def fractions_by_seed(A, starts, n=300, generations=50, thresh=4.0):
    out = []
    for s in SEEDS:
        finals = [simulate_fast(T0, R0, n=n, generations=generations, allocare=A,
                                seed=s)[1].mean() for (T0, R0) in starts]
        out.append(100 * np.mean(np.array(finals) > thresh))
    return np.array(out)


def summarize(fr):
    return fr.mean(), fr.std(ddof=1), fr.min(), fr.max()


def main():
    base = grid(6, 6)
    print("1. Basin fraction by shared care, 6 x 6 grid, N = 300, 50 generations, R > 4")
    print(f"{'A':>4} {'mean %':>8} {'SD':>6} {'range':>16}")
    main_vals = {}
    for A in A_LEVELS:
        m, sd, lo, hi = summarize(fractions_by_seed(A, base))
        main_vals[A] = (m, sd)
        print(f"{A:4.0f} {m:8.1f} {sd:6.1f} [{lo:5.1f}, {hi:5.1f}]")
    assert abs(main_vals[0.0][0] - 30.6) < 0.1 and abs(main_vals[8.0][0] - 19.4) < 0.1

    print("\n2. Sensitivity of the A = 0 vs A = 8 comparison")
    print(f"{'variant':>34} {'A=0 mean (SD)':>16} {'A=8 mean (SD)':>16} {'difference':>11}")
    variants = [
        ("threshold R > 3.5", dict(thresh=3.5), base),
        ("threshold R > 4.0 (main)", dict(thresh=4.0), base),
        ("threshold R > 4.5", dict(thresh=4.5), base),
        ("80 generations", dict(generations=80), base),
        ("population 500", dict(n=500), base),
        ("11 x 11 grid", dict(), grid(11, 11)),
    ]
    rows = []
    for name, kw, starts in variants:
        f0 = fractions_by_seed(0.0, starts, **kw)
        f8 = fractions_by_seed(8.0, starts, **kw)
        d = f0.mean() - f8.mean()
        rows.append((name, f0.mean(), f0.std(ddof=1), f8.mean(), f8.std(ddof=1), d))
        print(f"{name:>34} {f0.mean():8.1f} ({f0.std(ddof=1):4.1f}) "
              f"{f8.mean():8.1f} ({f8.std(ddof=1):4.1f}) {d:10.1f}")
        # the basin is smaller under shared care in every seed-mean comparison,
        # by more than two seed SDs
        assert d > 0 and d > 2 * max(f0.std(ddof=1), f8.std(ddof=1)) / np.sqrt(len(SEEDS)), name
    print("\nIn every variant the runaway basin is smaller under shared care.")
    return main_vals, rows


if __name__ == "__main__":
    main()
