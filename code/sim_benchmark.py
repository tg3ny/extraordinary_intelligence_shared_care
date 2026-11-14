"""
Validation of the vectorized simulation (fast_sim.py) against the reference
sequential implementation (pk_model.simulate_population).

The fast routine generates offspring in batches rather than one birth at a time,
so it can overwrite duplicated replacement slots within a generation and does
not let the population change within a generation. That is an approximation to
the sequential birth-and-replace dynamics, and it should not be assumed to
reproduce identical population behaviour. This script tests the assumption in
three ways:

  1. endpoint classification (runaway or not) on paired runs with the same
     start, care level, and seed;
  2. the final mean brain size and birth age of those paired runs (correlation,
     mean absolute difference, and the same statistics for runs that end near
     the classification boundary);
  3. the basin fraction over the full starting grid for both implementations,
     summarized across seeds (the grid is prespecified, so a binomial interval
     over start x seed outcomes is not appropriate; see basin_sensitivity.py).

Set SIM_BENCHMARK_FAST=1 in the environment to skip part 3, which runs the slow
sequential reference over the whole grid.
"""
import os
import numpy as np
from pk_model import simulate_population
from fast_sim import simulate_fast

RUNAWAY_R = 4.0          # endpoint classification threshold used in the manuscript
N = 300
GENERATIONS = 50
SEEDS = list(range(10))


def classify(R_final):
    return float(np.mean(R_final) > RUNAWAY_R)


def endpoint(T0, R0, A, seed, ref=False):
    sim = simulate_population if ref else simulate_fast
    T, R, I = sim(T0, R0, n=N, generations=GENERATIONS, allocare=A, seed=seed)
    return float(np.mean(T)), float(np.mean(R)), classify(R)


def main():
    starts = [(4, 3), (6, 4), (8, 5), (12, 3), (16, 2), (20, 4)]
    print("=" * 78)
    print("1. ENDPOINT AGREEMENT on paired runs: fast_sim vs reference simulate_population")
    print("=" * 78)
    print(f"{'start (T,R)':>12} {'A':>4} {'ref meanR':>10} {'fast meanR':>11} "
          f"{'ref cls':>8} {'fast cls':>9} {'match':>6}")
    pairs = []            # (A, T0, R0, seed, Tr, Rr, cr, Tf, Rf, cf)
    for A in (0.0, 8.0):
        for (T0, R0) in starts:
            for seed in SEEDS[:5]:
                Tr, Rr, cr = endpoint(T0, R0, A, seed, ref=True)
                Tf, Rf, cf = endpoint(T0, R0, A, seed, ref=False)
                pairs.append((A, T0, R0, seed, Tr, Rr, cr, Tf, Rf, cf))
            print(f"{str((T0,R0)):>12} {A:4.0f} {Rr:10.2f} {Rf:11.2f} "
                  f"{cr:8.0f} {cf:9.0f} {'yes' if cr == cf else 'NO':>6}")
    P = np.array([p[4:] for p in pairs])     # Tr, Rr, cr, Tf, Rf, cf
    agree = int(np.sum(P[:, 2] == P[:, 5])); tot = len(pairs)
    print(f"\n  endpoint classification agreement: {agree}/{tot} ({100*agree/tot:.1f}%)")

    print()
    print("=" * 78)
    print("2. FINAL TRAIT VALUES on the same paired runs")
    print("=" * 78)
    Tr, Rr, Tf, Rf = P[:, 0], P[:, 1], P[:, 3], P[:, 4]
    rR = np.corrcoef(Rr, Rf)[0, 1]; rT = np.corrcoef(Tr, Tf)[0, 1]
    dR = np.mean(np.abs(Rr - Rf)); dT = np.mean(np.abs(Tr - Tf))
    bR = np.mean(Rf - Rr); bT = np.mean(Tf - Tr)
    print(f"  final mean brain size R : r = {rR:.3f}, mean |diff| = {dR:.2f}, "
          f"mean (fast - ref) = {bR:+.2f}  (ref range {Rr.min():.1f} to {Rr.max():.1f})")
    print(f"  final mean birth age  T : r = {rT:.3f}, mean |diff| = {dT:.2f}, "
          f"mean (fast - ref) = {bT:+.2f}  (ref range {Tr.min():.1f} to {Tr.max():.1f})")
    near = np.abs(Rr - RUNAWAY_R) < 1.0
    if near.any():
        print(f"  runs ending within 1 unit of the R > {RUNAWAY_R:g} boundary (n = {near.sum()}): "
              f"mean |diff R| = {np.mean(np.abs(Rr[near]-Rf[near])):.2f}, "
              f"classification agreement {int(np.sum(P[near,2]==P[near,5]))}/{near.sum()}")
    sd_R = np.std(Rr, ddof=1)
    print(f"  (between-run SD of final R in the reference: {sd_R:.2f})")
    assert rR > 0.95 and rT > 0.95, "fast and reference final traits diverge"
    assert dR < 0.25 * sd_R, "fast-reference difference in R is not small relative to run-to-run spread"

    if os.environ.get("SIM_BENCHMARK_FAST"):
        print("\n  [SIM_BENCHMARK_FAST set: skipping part 3]")
        return
    print()
    print("=" * 78)
    print("3. BASIN FRACTION over the 6 x 6 grid, both implementations, summarized across seeds")
    print("=" * 78)
    T0s = np.linspace(3, 22, 6)
    R0s = np.linspace(1, 6, 6)
    print(f"{'A':>5} {'impl':>6} {'seeds':>6} {'mean %':>8} {'SD':>6} {'range':>16}")
    for A in (0.0, 4.0, 8.0, 12.0):
        for impl, ref in (("fast", False), ("ref", True)):
            seeds = SEEDS[:3] if ref else SEEDS
            fr = []
            for seed in seeds:
                k = sum(endpoint(T0, R0, A, seed, ref=ref)[2] for T0 in T0s for R0 in R0s)
                fr.append(100 * k / (len(T0s) * len(R0s)))
            fr = np.array(fr)
            print(f"{A:5.0f} {impl:>6} {len(seeds):6d} {fr.mean():8.1f} "
                  f"{fr.std(ddof=1):6.1f} [{fr.min():5.1f}, {fr.max():5.1f}]")
    print("\n  Note: the reference implementation is run with fewer seeds because it")
    print("  is sequential and far slower.")


if __name__ == "__main__":
    main()
