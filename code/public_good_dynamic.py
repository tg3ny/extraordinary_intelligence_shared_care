"""
Dynamic public-good variant of shared care.

In model_tightening.py the public-good formulation A = kappa * I_group is
evaluated statically: group intelligence is held fixed while the gradient on
a focal individual's brain size is computed. That shows that no individual can
buy its offspring more care by growing a larger brain, but it is equivalent to
a fixed A at the evaluation point and says nothing about the evolutionary
dynamics of group-level care.

Here the public good is made dynamic: in every generation the care available
to all offspring is A_t = kappa * mean(I_t) of the resident population, so care
rises and falls with the group's own evolving intelligence (a positive group-
level feedback that could in principle sustain the runaway). The simulation
otherwise follows fast_sim.simulate_fast exactly, including the order of
random draws, so kappa = 0 reproduces the A = 0 baseline run for run.

Reported: the share of the 6 x 6 grid of starting conditions that runs away
(final mean R > 4) as mean and SD across 10 seeds, and the mean final brain
size, for kappa in {0, 0.5, 1, 2, 4}.
"""
import numpy as np
from pk_model import p_survive_birth, p_survive_childhood
from fast_sim import simulate_fast

N, GENERATIONS, SEEDS = 300, 50, list(range(10))
T0s, R0s = np.linspace(3, 22, 6), np.linspace(1, 6, 6)
KAPPAS = (0.0, 0.5, 1.0, 2.0, 4.0)
RUNAWAY_R = 4.0


def simulate_public_good(T0, R0, kappa, n=N, generations=GENERATIONS, sigma=1.0,
                         rho=0.9, seed=None):
    rng = np.random.default_rng(seed)
    T = np.full(n, float(T0)); R = np.full(n, float(R0)); I = np.full(n, float(R0))
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    for _ in range(generations):
        A = kappa * I.mean()                       # public good tracks the group
        a = rng.integers(0, n, size=n); b = rng.integers(0, n, size=n)
        Tp = 0.5*(T[a]+T[b]); Rp = 0.5*(R[a]+R[b]); Ip = 0.5*(I[a]+I[b])
        dT = rng.normal(0, sigma, n)
        z = rng.normal(0, 1, size=(2, n)); corr = L @ z
        Tc = np.maximum(Tp + dT, 0.1)
        Rc = np.maximum(Rp + sigma*corr[0], 0.01)
        Ic = np.maximum(Ip + sigma*corr[1], 0.01)
        pb = p_survive_birth(Tc, Rc)
        pc = p_survive_childhood(Tc, Ip, allocare=A)
        survive = (rng.random(n) < pb) & (rng.random(n) < pc)
        idx = np.where(survive)[0]
        if idx.size:
            slots = rng.integers(0, n, size=idx.size)
            T[slots] = Tc[idx]; R[slots] = Rc[idx]; I[slots] = Ic[idx]
    return T, R, I


def basin_by_seed(kappa):
    fr, finalR = [], []
    for s in SEEDS:
        big = 0
        for T0 in T0s:
            for R0 in R0s:
                _, R, _ = simulate_public_good(T0, R0, kappa, seed=s)
                big += R.mean() > RUNAWAY_R; finalR.append(R.mean())
        fr.append(100 * big / (len(T0s) * len(R0s)))
    return np.array(fr), float(np.mean(finalR))


def main():
    print("Dynamic public good: A_t = kappa * mean(I_t), updated every generation")
    print(f"grid {len(T0s)} x {len(R0s)} starts, N = {N}, {GENERATIONS} generations, "
          f"{len(SEEDS)} seeds, runaway = final mean R > {RUNAWAY_R:g}")
    print(f"{'kappa':>6} {'% runaway':>10} {'SD (seeds)':>11} {'range':>14} {'mean final R':>13}")
    results = {}
    for k in KAPPAS:
        fr, mR = basin_by_seed(k)
        results[k] = (fr.mean(), fr.std(ddof=1), fr.min(), fr.max(), mR)
        print(f"{k:6.1f} {fr.mean():9.1f}% {fr.std(ddof=1):10.1f} "
              f"[{fr.min():5.1f}, {fr.max():5.1f}] {mR:13.2f}")
    # kappa = 0 must reproduce the fixed-A baseline exactly (same RNG stream)
    base = []
    for s in SEEDS:
        big = 0
        for T0 in T0s:
            for R0 in R0s:
                _, R, _ = simulate_fast(T0, R0, n=N, generations=GENERATIONS,
                                        allocare=0.0, seed=s)
                big += R.mean() > RUNAWAY_R
        base.append(100 * big / 36)
    assert abs(np.mean(base) - results[0.0][0]) < 1e-9, "kappa = 0 should equal A = 0"
    # values quoted in the manuscript
    assert abs(results[0.0][0] - 30.6) < 0.1, results[0.0]
    assert abs(results[1.0][0] - 24.4) < 0.1, results[1.0]
    assert abs(results[2.0][0] - 20.3) < 0.1, results[2.0]
    assert abs(results[4.0][0] - 17.8) < 0.1, results[4.0]
    assert results[0.0][0] > results[1.0][0] > results[2.0][0] > results[4.0][0]
    assert results[0.0][4] > results[4.0][4]
    print("\nThe runaway basin narrows as the public good strengthens, even though")
    print("care now grows with the group's own intelligence.")


if __name__ == "__main__":
    main()
