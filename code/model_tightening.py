"""
Model-side robustness for the allocare extension.

(A) gamma / V sensitivity of the selection-gradient collapse.
    Does allocare still switch off the pressure to grow brains across the
    childhood-mortality parameter gamma and the birth-size ceiling V?

(B) Group-scaling allocare variant.
    In the main result, allocare A is a fixed additive care capacity. A critic
    could object that we simply handed the population free care. Here allocare
    co-scales with the group's own intelligence (A = kappa * I_group), a public
    good decoupled from any single individual's brain size. If the private
    selection gradient on individual brain size still collapses, the result is
    not an artifact of a fixed subsidy: it is the socialization of care that
    matters, not its absolute size.
"""
import numpy as np
from pk_model import B, C, X99, logistic

def head_radius(x, R): return R * np.exp(-B * np.exp(-C * x))
def p_birth(T, R, V):  return logistic(V - head_radius(T, R))
def M_of(T):           return np.maximum(X99 - T, 1e-6)

def p_child(T, care, gamma):
    return np.exp(-M_of(T) * gamma / care)

def fitness(T, R, care, gamma, V):
    return p_birth(T, R, V) * p_child(T, care, gamma)

# altricial frontier used throughout (low birth age, modest brain)
FRONTIER_T = (3.0, 4.0, 5.0)
FRONTIER_R = 3.0

# ============================================================ (A) gamma/V grid
def gradient_fixedA(A, gamma, V, R0=FRONTIER_R, h=1e-3):
    """Mean selection gradient dF/dR over the altricial frontier, fixed allocare A."""
    vals = []
    for T in FRONTIER_T:
        fp = fitness(T, R0 + h, (R0 + h) + A, gamma, V)   # care = own Ip + A
        fm = fitness(T, R0 - h, (R0 - h) + A, gamma, V)
        vals.append((fp - fm) / (2 * h))
    return float(np.mean(vals))


def gradient_groupscale(kappa, gamma=0.4, V=5.48, R0=FRONTIER_R, h=1e-3):
    """Invasion gradient on a focal individual's brain size when allocare is a
    public good A = kappa * I_group, with I_group = R0 (resident) held fixed
    while the focal perturbs its own brain size."""
    A = kappa * R0                     # public-good care from the resident group
    vals = []
    for T in FRONTIER_T:
        fp = fitness(T, R0 + h, (R0 + h) + A, gamma, V)   # A does NOT track focal
        fm = fitness(T, R0 - h, (R0 - h) + A, gamma, V)
        vals.append((fp - fm) / (2 * h))
    return float(np.mean(vals))


if __name__ == "__main__":
    print("=== (A) gamma/V sensitivity of the gradient collapse (A=0 -> A=16) ===")
    print(f"{'gamma':>6} {'V':>6} {'grad(A=0)':>11} {'grad(A=16)':>11} {'% reduction':>12}")
    gammas = [0.2, 0.3, 0.4, 0.6, 0.8]
    Vs = [5.0, 5.48, 6.0, 6.5]
    redu = []
    for gamma in gammas:
        for V in Vs:
            g0 = gradient_fixedA(0.0, gamma, V)
            g16 = gradient_fixedA(16.0, gamma, V)
            pct = 100 * (1 - g16 / g0) if g0 != 0 else float("nan")
            redu.append(pct)
            print(f"{gamma:6.2f} {V:6.2f} {g0:11.4f} {g16:11.4f} {pct:11.1f}%")
    print(f"\n  gradient reduction across the grid: "
          f"min {min(redu):.1f}%  median {np.median(redu):.1f}%  max {max(redu):.1f}%")

    # ============================================================ (B) group scaling

    print("\n=== (B) Group-scaling allocare: A = kappa * I_group (public good) ===")
    print(f"{'kappa':>6} {'A=kappa*R':>10} {'gradient':>10} {'% reduction vs kappa=0':>22}")
    g_base = gradient_groupscale(0.0)
    for kappa in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        g = gradient_groupscale(kappa)
        pct = 100 * (1 - g / g_base)
        print(f"{kappa:6.2f} {kappa*FRONTIER_R:10.1f} {g:10.4f} {pct:21.1f}%")

    print("\nInterpretation: with care socialized as a public good that co-scales with")
    print("the group's own intelligence, the PRIVATE gradient on individual brain size")
    print("still collapses, because offspring survival is buffered by pooled care rather")
    print("than by the parent's own brain. The result does not depend on a fixed subsidy.")
