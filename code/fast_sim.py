"""Vectorized (generational) version of the P&K dynamics for parameter sweeps.
Generates all n offspring per generation from random parent pairs, applies
survival, and survivors overwrite random slots. Reproduces the same bistable
runaway as the sequential reference in pk_model.py."""
import numpy as np
from pk_model import (p_survive_birth, p_survive_childhood, B, C, V, GAMMA, X99)

def simulate_fast(T0, R0, n=500, generations=80, sigma=1.0, rho=0.9,
                  allocare=0.0, seed=None, record=False):
    rng = np.random.default_rng(seed)
    T = np.full(n, float(T0)); R = np.full(n, float(R0)); I = np.full(n, float(R0))
    L = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    traj = [(T.mean(), R.mean(), I.mean())] if record else None
    for _ in range(generations):
        a = rng.integers(0, n, size=n); b = rng.integers(0, n, size=n)
        Tp = 0.5*(T[a]+T[b]); Rp = 0.5*(R[a]+R[b]); Ip = 0.5*(I[a]+I[b])
        dT = rng.normal(0, sigma, n)
        z = rng.normal(0, 1, size=(2, n)); corr = L @ z
        Tc = np.maximum(Tp + dT, 0.1)
        Rc = np.maximum(Rp + sigma*corr[0], 0.01)
        Ic = np.maximum(Ip + sigma*corr[1], 0.01)
        pb = p_survive_birth(Tc, Rc)
        pc = p_survive_childhood(Tc, Ip, allocare=allocare)
        survive = (rng.random(n) < pb) & (rng.random(n) < pc)
        idx = np.where(survive)[0]
        if idx.size:
            slots = rng.integers(0, n, size=idx.size)
            T[slots] = Tc[idx]; R[slots] = Rc[idx]; I[slots] = Ic[idx]
        if record:
            traj.append((T.mean(), R.mean(), I.mean()))
    return np.array(traj) if record else (T, R, I)

if __name__ == "__main__":
    for (T0,R0) in [(3,5),(5,4),(20,3),(25,2)]:
        tr = simulate_fast(T0,R0,seed=1,record=True)
        print(f"start T={T0:>2} R={R0:>2} -> final T={tr[-1,0]:5.2f} R={tr[-1,1]:5.2f} I={tr[-1,2]:5.2f}")
