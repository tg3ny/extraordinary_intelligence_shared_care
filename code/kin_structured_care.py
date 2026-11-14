"""
Kin-structured shared care: does the runaway basin still narrow when helpers are
relatives of the infant?

In public_good_dynamic.py the care available to every infant is kappa times the
population's mean intelligence, so helper intelligence is uncorrelated with the
infant's own lineage (unrelated helpers). If helpers are kin, their intelligence is
correlated with the infant's parents'. For a helper related to the infant by r, the
expected helper intelligence given the infant's parents regresses on the parental
midpoint with slope rho = 2r (full sibling: rho = 1; grandmother, aunt or half-
sibling: rho = 1/2; unrelated: rho = 0). Care is therefore modeled as

    A_i = kappa * [ rho * Ip_i + (1 - rho) * mean(I_t) ],

where Ip_i is the midparent intelligence of offspring i. At rho = 0 this is the
unrelated dynamic public good; at rho = 1 it is equivalent to dividing the
childhood-mortality parameter gamma by (1 + kappa) for the whole lineage.

Two demographic schemes are run. "replace" is the published birth-and-replace
scheme used throughout the manuscript (survivors overwrite random members, so mean
survival also sets turnover). "wf" is a standard Wright-Fisher scheme with non-
overlapping generations (the whole population is resampled from the survivors), in
which only relative fitness matters. The random streams are matched to fast_sim so
that rho = 0 reproduces public_good_dynamic.py and kappa = 0 reproduces A = 0.

Reported: the share of the 6 x 6 grid of starting conditions that runs away (final
mean R > 4), mean and SD across seeds, and the mean final brain size.
"""
import numpy as np
from pk_model import p_survive_birth, p_survive_childhood

N, GENERATIONS = 300, 50
T0s, R0s = np.linspace(3, 22, 6), np.linspace(1, 6, 6)
RHOS = (0.0, 0.5, 1.0)
KAPPAS = (0.0, 1.0, 2.0, 4.0)
RUNAWAY_R = 4.0


def simulate_kin(T0, R0, kappa, rho, scheme="replace", n=N, generations=GENERATIONS,
                 sigma=1.0, corr=0.9, seed=None):
    rng = np.random.default_rng(seed)
    T = np.full(n, float(T0)); R = np.full(n, float(R0)); I = np.full(n, float(R0))
    L = np.linalg.cholesky(np.array([[1.0, corr], [corr, 1.0]]))
    for _ in range(generations):
        Ibar = I.mean()
        a = rng.integers(0, n, size=n); b = rng.integers(0, n, size=n)
        Tp = 0.5*(T[a]+T[b]); Rp = 0.5*(R[a]+R[b]); Ip = 0.5*(I[a]+I[b])
        dT = rng.normal(0, sigma, n)
        z = rng.normal(0, 1, size=(2, n)); c = L @ z
        Tc = np.maximum(Tp + dT, 0.1)
        Rc = np.maximum(Rp + sigma*c[0], 0.01)
        Ic = np.maximum(Ip + sigma*c[1], 0.01)
        A = kappa * (rho * Ip + (1.0 - rho) * Ibar)
        survive = (rng.random(n) < p_survive_birth(Tc, Rc)) & \
                  (rng.random(n) < p_survive_childhood(Tc, Ip, allocare=A))
        idx = np.where(survive)[0]
        if idx.size == 0:
            continue
        if scheme == "replace":
            slots = rng.integers(0, n, size=idx.size)
            T[slots] = Tc[idx]; R[slots] = Rc[idx]; I[slots] = Ic[idx]
        else:
            pick = rng.choice(idx, size=n, replace=True)
            T = Tc[pick].copy(); R = Rc[pick].copy(); I = Ic[pick].copy()
    return T, R, I


def basin_by_seed(kappa, rho, scheme="replace", seeds=10):
    fr, finalR = [], []
    for s in range(seeds):
        big = 0
        for T0 in T0s:
            for R0 in R0s:
                _, R, _ = simulate_kin(T0, R0, kappa, rho, scheme=scheme, seed=s)
                big += R.mean() > RUNAWAY_R; finalR.append(R.mean())
        fr.append(100 * big / (len(T0s) * len(R0s)))
    return np.array(fr), float(np.mean(finalR))


def run_scheme(scheme, seeds):
    print(f"\nScheme: {scheme} ({seeds} seeds)")
    print(f"{'rho':>5} {'kappa':>6} {'% runaway':>10} {'SD':>5} {'mean final R':>13}")
    out = {}
    for rho in RHOS:
        for kappa in KAPPAS:
            if kappa == 0 and rho > 0:
                continue
            fr, mR = basin_by_seed(kappa, rho, scheme, seeds)
            out[(rho, kappa)] = (fr.mean(), fr.std(ddof=1), mR)
            print(f"{rho:5.1f} {kappa:6.1f} {fr.mean():9.1f}% {fr.std(ddof=1):5.1f} {mR:13.2f}")
    return out


def main():
    rep = run_scheme("replace", 10)
    # rho = 0 reproduces public_good_dynamic.py; kappa = 0 reproduces the A = 0 baseline
    assert abs(rep[(0.0, 0.0)][0] - 30.6) < 0.1
    assert abs(rep[(0.0, 2.0)][0] - 20.3) < 0.1 and abs(rep[(0.0, 4.0)][0] - 17.8) < 0.1
    # values quoted in the manuscript
    assert abs(rep[(0.5, 2.0)][0] - 26.4) < 0.1 and abs(rep[(0.5, 4.0)][0] - 23.3) < 0.1
    assert abs(rep[(1.0, 2.0)][0] - 35.6) < 0.1 and abs(rep[(1.0, 4.0)][0] - 31.7) < 0.1
    base = rep[(0.0, 0.0)][0]
    # unrelated care narrows the basin; full-sibling care does not
    assert rep[(0.0, 4.0)][0] < base - 10
    assert rep[(1.0, 2.0)][0] >= base and rep[(1.0, 4.0)][0] >= base - 1
    wf = run_scheme("wf", 20)
    base_wf = wf[(0.0, 0.0)][0]
    assert wf[(0.0, 4.0)][0] < base_wf - 8          # unrelated care narrows the basin
    assert wf[(1.0, 2.0)][0] > base_wf and wf[(1.0, 4.0)][0] > base_wf   # full-sibling care does not
    print("\nUnder both schemes the basin narrows with unrelated helpers, changes little")
    print("with r = 1/4 helpers, and does not narrow when all helpers are full siblings.")


if __name__ == "__main__":
    main()
