r"""
Analytic treatment of the shared-care extension: closed-form gradients,
dimensionless calibration, component decomposition, and sensitivity to the
functional form of care.

Model (Piantadosi & Kidd 2016, with the additive care extension):
    F(T, R, A) = B(T, R) * C(T, R, A)
    B = logistic(V - g(T, R)),   g(T, R) = R * exp(-b * exp(-c * T))
    C = exp(-M(T) * gamma / K),  K = R + A   [care capacity, with Ip = R]

Two gradients are reported along the linked axis I = R.

  Relative (log-fitness) gradient, the standard measure of selection strength
  on a trait (selection acts on fitness differences among competing types):

      d ln F / dR = B_R / B + M * gamma / (R + A)^2

  The care-mediated term is EXACTLY proportional to 1 / (R + A)^2, so pooled
  care equal to a times the parent's own capacity multiplies it by 1/(1+a)^2:
  a = 1 removes 75%, a = 2 removes 88.9%, a = 5.33 removes 97.5%. The birth
  term B_R / B is about 4e-6 at the frontier and does not affect these figures.

  Absolute gradient, the change in survival probability per unit brain size:

      dF / dR = F * d ln F / dR = C * [ B_R + B * M * gamma / (R + A)^2 ]

  This falls less (29.6%, 55.8%, 85.7% at a = 1, 2, 5.33) because pooled care
  also raises baseline survival C, which multiplies the absolute gradient.
  Under the model's birth-and-replace dynamics, in which survivors replace
  random members, mean survival also sets the turnover rate, so the per-
  generation change in the population mean scales with this absolute gradient
  (Delta z_bar = Cov(z, F)). Both measures are reported in the manuscript; the
  two agree on sign throughout (sign change only at a of about 203).

Functional form. For any care function K(Ip, A),

      d ln C / d Ip = M * gamma * (dK/dIp) / K^2,

  so the relative return to parental intelligence falls with pooled care
  whenever K increases with A faster than dK/dIp does, which holds for
  additive, multiplicative, and interactive specifications alike. The
  dilution follows from diminishing returns to care capacity in the survival
  function (the hazard gamma / K is convex in K), not from additivity.

Relatedness. If a helper is related to the infant by r, the allele for a larger
brain is present in the helper with regression r on the infant's genotype, so part
of the care an infant receives tracks its own lineage's brain genes. With care-
weighted mean relatedness rbar across helpers (the parent-infant regression being
1/2), the relative gradient becomes

      d ln F / dR  =  B_R / B + M * gamma * (1 + 2 * rbar * a) / (R + A)^2,

  a ratio of (1 + 2 rbar a) / (1 + a)^2 to the private-care model. Since no helper
  is more related to an infant than a full sibling (rbar <= 1/2), the ratio is at
  most 1 / (1 + a): kin care returns part of the private payoff but never removes
  the dilution. Reductions at a = 1 (a = 2): unrelated 75% (88.9%); r = 1/4 helpers
  such as grandmothers, aunts and half-siblings 62.5% (77.8%); full siblings only
  50% (66.7%). See kin_structured_care.py for the simulation counterpart.

A genotype-independent cost of helping, exp(-k_c * A) multiplying fitness,
leaves relative fitness and the log-gradient unchanged; it is therefore not a
test of robustness and is no longer reported (an earlier version of this
script printed one).
"""
import numpy as np
from pk_model import (B as GOMP_B, C as GOMP_C, V, GAMMA, X99,
                      logistic, head_radius_at_age, time_to_maturity, fitness)

FRONTIER_T = (3.0, 4.0, 5.0)
FRONTIER_R = 3.0


# ---------------------------------------------------------------- analytic
def dg_dR(T):
    """d(head radius at birth)/dR."""
    return np.exp(-GOMP_B * np.exp(-GOMP_C * T))


def components(T, R, A):
    """Return (birth_term, childhood_term, total) of the ABSOLUTE gradient dF/dR."""
    g = head_radius_at_age(T, R)
    Bv = logistic(V - g)
    M = time_to_maturity(T)
    care = R + A
    Cv = np.exp(-M * GAMMA / care)
    B_R = Bv * (1.0 - Bv) * (-dg_dR(T))          # birth cost (negative)
    C_R = Cv * M * GAMMA / care**2                # childhood benefit (positive)
    birth_term = B_R * Cv
    child_term = Bv * C_R
    return birth_term, child_term, birth_term + child_term


def rel_components(T, R, A):
    """Return (birth_term, childhood_term, total) of the RELATIVE gradient d ln F/dR."""
    g = head_radius_at_age(T, R)
    Bv = logistic(V - g)
    M = time_to_maturity(T)
    birth = (1.0 - Bv) * (-dg_dR(T))              # B_R / B
    child = M * GAMMA / (R + A)**2                # exact inverse square
    return birth, child, birth + child


def gradient_analytic(A, R=FRONTIER_R, Ts=FRONTIER_T):
    """Absolute gradient dF/dR averaged over the frontier."""
    return float(np.mean([components(t, R, A)[2] for t in Ts]))


def rel_gradient_analytic(A, R=FRONTIER_R, Ts=FRONTIER_T):
    """Relative gradient d ln F/dR averaged over the frontier."""
    return float(np.mean([rel_components(t, R, A)[2] for t in Ts]))


def gradient_numeric(A, R=FRONTIER_R, Ts=FRONTIER_T, h=1e-4):
    """Central difference of F along I = R."""
    out = []
    for t in Ts:
        f1 = fitness(t, R + h, Ip=R + h, allocare=A)
        f0 = fitness(t, R - h, Ip=R - h, allocare=A)
        out.append((f1 - f0) / (2 * h))
    return float(np.mean(out))


def rel_gradient_numeric(A, R=FRONTIER_R, Ts=FRONTIER_T, h=1e-4):
    """Central difference of ln F along I = R."""
    out = []
    for t in Ts:
        f1 = np.log(fitness(t, R + h, Ip=R + h, allocare=A))
        f0 = np.log(fitness(t, R - h, Ip=R - h, allocare=A))
        out.append((f1 - f0) / (2 * h))
    return float(np.mean(out))


# ------------------------------------------------- alternative care functions
def rel_care_return(K, dK_dIp, T, Ip=FRONTIER_R):
    """Care-mediated relative return d ln C / d Ip = M gamma K_Ip / K^2."""
    M = time_to_maturity(T)
    return M * GAMMA * dK_dIp / K**2


CARE_FORMS = {
    # name: (K(Ip, a), dK/dIp(Ip, a)), with a = A / Ip the care ratio
    "additive  K = Ip + A":            (lambda Ip, a: Ip + a * Ip,            lambda Ip, a: 1.0),
    "multiplicative  K = Ip (1 + a)":  (lambda Ip, a: Ip * (1 + a),           lambda Ip, a: 1 + a),
    "interactive  K = Ip + A + Ip A":  (lambda Ip, a: Ip + a * Ip + Ip * a * Ip, lambda Ip, a: 1 + a * Ip),
}


def care_form_reductions(a_values=(1.0, 2.0), Ip=FRONTIER_R, Ts=FRONTIER_T):
    """Percent reduction in the care-mediated relative return at each care ratio."""
    out = {}
    for name, (K, dK) in CARE_FORMS.items():
        base = np.mean([rel_care_return(K(Ip, 0.0), dK(Ip, 0.0), t, Ip) for t in Ts])
        out[name] = [100 * (1 - np.mean([rel_care_return(K(Ip, a), dK(Ip, a), t, Ip)
                                         for t in Ts]) / base) for a in a_values]
    return out


def kin_ratio(a, rbar):
    """Relative gradient under kin-structured care as a fraction of the private-care value."""
    return (1 + 2 * rbar * a) / (1 + a) ** 2


def main():
    print("=" * 74)
    print("1. ANALYTIC vs NUMERIC GRADIENTS (validates both closed forms)")
    print("=" * 74)
    print(f"{'A':>6} {'dF/dR analytic':>15} {'numeric':>12} {'dlnF/dR analytic':>17} {'numeric':>12}")
    for A in [0.0, 0.75, 1.5, 3.0, 6.0, 12.0, 16.0]:
        a_, n_ = gradient_analytic(A), gradient_numeric(A)
        ra, rn = rel_gradient_analytic(A), rel_gradient_numeric(A)
        print(f"{A:6.2f} {a_:15.6f} {n_:12.6f} {ra:17.6f} {rn:12.6f}")
        assert abs(a_ - n_) < 1e-9 and abs(ra - rn) < 1e-9, "analytic and numeric disagree beyond 1e-9"
    print("  [both derivations match numerical differentiation to within 1e-9]")

    print()
    print("=" * 74)
    print("2. DIMENSIONLESS CALIBRATION, a = A / Ip at Ip = R = %.1f" % FRONTIER_R)
    print("   relative gradient d ln F/dR (selection strength) and absolute gradient dF/dR")
    print("=" * 74)
    g0, r0 = gradient_analytic(0.0), rel_gradient_analytic(0.0)
    print(f"{'a':>6} {'A':>6} {'rel. grad':>11} {'% red.':>8} {'1/(1+a)^2':>10} "
          f"{'abs. grad':>11} {'% red.':>8}")
    rel_red, abs_red = {}, {}
    for a in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.33]:
        A = a * FRONTIER_R
        g, r = gradient_analytic(A), rel_gradient_analytic(A)
        rel_red[a] = 100 * (1 - r / r0)
        abs_red[a] = 100 * (1 - g / g0)
        print(f"{a:6.2f} {A:6.2f} {r:11.5f} {rel_red[a]:7.1f}% {1/(1+a)**2:10.4f} "
              f"{g:11.6f} {abs_red[a]:7.1f}%")
    # values quoted in the manuscript
    assert abs(rel_red[1.0] - 75.0) < 0.2 and abs(rel_red[2.0] - 88.9) < 0.2 \
        and abs(rel_red[5.33] - 97.5) < 0.2, rel_red
    assert abs(abs_red[1.0] - 29.6) < 0.2 and abs(abs_red[2.0] - 55.8) < 0.2 \
        and abs(abs_red[5.33] - 85.7) < 0.2, abs_red
    # the relative reduction equals the exact inverse-square factor to within the
    # negligible birth term
    for a in (1.0, 2.0, 5.33):
        assert abs(rel_red[a] - 100 * (1 - 1 / (1 + a)**2)) < 0.05
    print("  Interpretation: a = 1 is pooled care equal to one additional caregiver")
    print("  of the parent's own competence. The relative reduction equals the exact")
    print("  inverse-square factor 1 - 1/(1+a)^2; the absolute reduction is smaller")
    print("  because pooled care also raises baseline survival.")

    print()
    print("=" * 74)
    print("3. COMPONENT DECOMPOSITION of dF/dR (birth cost vs care benefit) and sign")
    print("=" * 74)
    print(f"{'a':>6} {'birth term':>13} {'child term':>13} {'total':>13} {'ratio':>9}")
    for a in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 40.0]:
        A = a * FRONTIER_R
        b_, c_, t_ = (np.mean([components(t, FRONTIER_R, A)[k] for t in FRONTIER_T])
                      for k in (0, 1, 2))
        print(f"{a:6.1f} {b_:13.6f} {c_:13.6f} {t_:13.6f} {abs(b_)/c_:9.1e}")
    lo, hi = 0.0, 400.0
    assert gradient_analytic(hi * FRONTIER_R) < 0 < gradient_analytic(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gradient_analytic(mid * FRONTIER_R) > 0:
            lo = mid
        else:
            hi = mid
    a_sign = 0.5 * (lo + hi)
    print(f"\n  Selection on brain size changes sign at a = A/Ip = {a_sign:.1f}")
    assert 195 < a_sign < 210, a_sign
    print("  Below this ratio the total gradient is reduced but positive: shared care")
    print("  weakens selection for larger brains without reversing it.")

    print()
    print("=" * 74)
    print("4. FUNCTIONAL FORM OF CARE: % reduction in the care-mediated relative return")
    print("   d ln C / d Ip = M gamma K_Ip / K^2, at a = 1 and a = 2")
    print("=" * 74)
    forms = care_form_reductions()
    for name, (r1, r2) in forms.items():
        print(f"  {name:36s} a=1: {r1:5.1f}%   a=2: {r2:5.1f}%")
    assert abs(forms["additive  K = Ip + A"][0] - 75.0) < 0.1
    assert abs(forms["multiplicative  K = Ip (1 + a)"][0] - 50.0) < 0.1
    assert abs(forms["multiplicative  K = Ip (1 + a)"][1] - 66.7) < 0.1
    assert all(r > 0 for v in forms.values() for r in v)
    print("  The return falls under every specification in which pooled care raises")
    print("  total capacity; the exact inverse square is specific to additive pooling.")

    print()
    print("=" * 74)
    print("5. RELATEDNESS OF HELPERS: % reduction in the relative return, (1 + 2 rbar a)/(1 + a)^2")
    print("=" * 74)
    print(f"{'a':>5} {'unrelated (rbar=0)':>20} {'rbar = 1/4':>12} {'rbar = 1/2 (full sibs)':>24} {'bound 1/(1+a)':>14}")
    for a in (0.5, 1.0, 2.0, 4.0):
        vals = [100 * (1 - kin_ratio(a, r)) for r in (0.0, 0.25, 0.5)]
        print(f"{a:5.1f} {vals[0]:19.1f}% {vals[1]:11.1f}% {vals[2]:23.1f}% {100*(1-1/(1+a)):13.1f}%")
        assert kin_ratio(a, 0.5) <= 1 / (1 + a) + 1e-12
    assert abs(100 * (1 - kin_ratio(1.0, 0.25)) - 62.5) < 0.05
    assert abs(100 * (1 - kin_ratio(1.0, 0.5)) - 50.0) < 0.05
    assert abs(100 * (1 - kin_ratio(2.0, 0.25)) - 77.8) < 0.05
    assert abs(100 * (1 - kin_ratio(2.0, 0.5)) - 66.7) < 0.05
    print("  Kin care returns part of the private payoff; the dilution is bounded below by")
    print("  1 - 1/(1 + a) because no helper is more related to an infant than a full sibling.")


if __name__ == "__main__":
    main()
