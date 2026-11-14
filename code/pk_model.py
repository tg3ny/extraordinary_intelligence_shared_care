"""
Faithful reproduction of the core evolutionary model from
Piantadosi & Kidd (2016) "Extraordinary intelligence and the care of infants",
PNAS 113(25):6874-6879.

All equations and parameter values are taken directly from the paper text
(Eqs. 1-3, Materials and Methods).

    Eq 1 (Gompertz growth of head radius):
        g(x, R) = R * exp(-b * exp(-c * x)),   b = 37.1, c = 0.42
    Eq 2 (birth survival):
        P(survive birth | T, R) = phi(V - g(T, R)),  phi = logistic, V = 5.48
    Eq 3 (childhood survival):
        P(survive to adulthood | M, Ip) = exp(-M * (gamma / Ip)), gamma = 0.4
        with M = time-to-maturity = x99 - T, where g(x99, R) = 0.99 R
        (note: because R is a scale factor, x99 is independent of R)

    Linkage: mutations to R are 90% correlated with mutations to I.
    Fitness = Eq2 * Eq3 (independent events), given T, R, I.
"""

import numpy as np

# ----------------------------------------------------------------------
# Parameters (verbatim from the paper)
# ----------------------------------------------------------------------
B = 37.1        # Gompertz y-displacement
C = 0.42        # Gompertz growth rate
V = 5.48        # head-radius ceiling at birth (cm)
GAMMA = 0.4     # childhood mortality rate parameter (Fig 2 uses gamma = 0.4)

# Age at which Gompertz reaches 99% of R (R cancels out):
#   0.99 = exp(-B exp(-C x))  ->  x99 = -ln( -ln(0.99)/B ) / C
X99 = -np.log(-np.log(0.99) / B) / C


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


def head_radius_at_age(x, R):
    """Eq 1: head radius at age x for adult (limiting) size R."""
    return R * np.exp(-B * np.exp(-C * x))


def p_survive_birth(T, R):
    """Eq 2: probability of surviving childbirth given birth age T, adult size R."""
    return logistic(V - head_radius_at_age(T, R))


def time_to_maturity(T):
    """M = x99 - T (clamped at a small positive floor)."""
    return np.maximum(X99 - T, 1e-6)


def p_survive_childhood(T, Ip, allocare=0.0):
    """Eq 3 (optionally with an additive allocare care-capacity term A).

    Baseline model: care capacity = Ip (parental intelligence only).
    Allocare extension: care capacity = Ip + allocare, so the survival of a
    dependent offspring no longer rests on the parent's individual brain size
    alone but on pooled care from the social group.
    """
    M = time_to_maturity(T)
    care = Ip + allocare
    return np.exp(-M * (GAMMA / care))


def fitness(T, R, Ip=None, allocare=0.0):
    """Product of birth and childhood survival.

    For the landscape (Fig 2A) the paper sets Ip = R (parent intelligence equals
    offspring limiting brain size, valid when generational change is small).
    """
    if Ip is None:
        Ip = R
    return p_survive_birth(T, R) * p_survive_childhood(T, Ip, allocare=allocare)


# ----------------------------------------------------------------------
# Population dynamics (Fig 2B)
# ----------------------------------------------------------------------
def simulate_population(T0, R0, n=1000, generations=100, sigma=1.0,
                        rho=0.9, allocare=0.0, seed=None, record=False):
    """Wright-Fisher-style birth/replace dynamics from Piantadosi & Kidd.

    Each generation: pick two random parents, average their (T, R, I); produce a
    child with Gaussian mutation (SD = sigma); mutations to R and I are correlated
    with coefficient rho (=0.9 in the paper). If the child survives birth (Eq 2,
    on the CHILD's T,R) and childhood (Eq 3, on the PARENTS' averaged I), it
    replaces a random population member.

    Returns final (T, R, I) arrays, or a trajectory of generation-means if record.
    """
    rng = np.random.default_rng(seed)
    T = np.full(n, float(T0))
    R = np.full(n, float(R0))
    I = np.full(n, float(R0))   # intelligence initialised equal to brain size

    # correlated-mutation Cholesky factor for (R, I)
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)

    traj = []
    if record:
        traj.append((T.mean(), R.mean(), I.mean()))

    # one "generation" = n reproduction attempts
    for _ in range(generations):
        for _ in range(n):
            i, j = rng.integers(0, n, size=2)
            Tp = 0.5 * (T[i] + T[j])
            Rp = 0.5 * (R[i] + R[j])
            Ip = 0.5 * (I[i] + I[j])

            dT = rng.normal(0.0, sigma)
            corr = L @ rng.normal(0.0, 1.0, size=2)  # correlated (dR, dI)
            Tc = Tp + dT
            Rc = Rp + sigma * corr[0]
            Ic = Ip + sigma * corr[1]

            # keep variables in sensible physical ranges
            Tc = max(Tc, 0.1)
            Rc = max(Rc, 0.01)
            Ic = max(Ic, 0.01)

            # survival: birth uses child's (T,R); childhood uses parents' Ip
            if rng.random() < p_survive_birth(Tc, Rc):
                if rng.random() < p_survive_childhood(Tc, Ip, allocare=allocare):
                    k = rng.integers(0, n)
                    T[k], R[k], I[k] = Tc, Rc, Ic
        if record:
            traj.append((T.mean(), R.mean(), I.mean()))

    if record:
        return np.array(traj)
    return T, R, I


if __name__ == "__main__":
    print(f"x99 (age at 99% of R) = {X99:.3f} months")
    # quick sanity checks
    print("Birth survival at T=9, R=8.4:", round(float(p_survive_birth(9, 8.4)), 3))
    print("Birth survival at T=3, R=8.4:", round(float(p_survive_birth(3, 8.4)), 3))
    print("Childhood survival at T=9, Ip=8.4:", round(float(p_survive_childhood(9, 8.4)), 3))
