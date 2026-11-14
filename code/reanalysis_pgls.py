"""
Empirical model-tightening for the P&K reanalysis.

Uses the assembled 23-genus matrix and the ultrametric phylogenetic VCV
(phylo_C.npy, in tree_genera.csv order). Adds the ADW-sourced Cercocebus
weaning value, then runs:

  1. Kendall tau (headline) at n=21 and n=22.
  2. PGLS of intelligence ~ weaning, standardized beta, under Brownian
     (lambda=1) and under ML-estimated Pagel's lambda.
  3. Single-brain-control PGLS (intelligence ~ weaning + total brain) under
     ML lambda, n=18.
  4. One-tailed vs two-tailed p for the headline single-control model.

No fabrication: Cercocebus weaning = 270 d (9 mo), consistent with ADW
(C. chrysogaster 8-9 mo, C. torquatus 7-10 mo) and AnAge (C. galeritus 12 mo);
Presbytis weaning left missing (not responsibly sourceable).
"""
import numpy as np, pandas as pd
from scipy.stats import kendalltau, t as tdist
from scipy.optimize import minimize_scalar

import os
_D = os.path.join(os.path.dirname(__file__), "..", "data")
_p = lambda f: os.path.join(_D, f) if os.path.exists(os.path.join(_D, f)) else f
df = pd.read_csv(_p("childhood_first_datamatrix_COMPLETE.csv"))
genera = pd.read_csv(_p("tree_genera.csv"))["genus"].tolist()
C_full = np.load(_p("phylo_C.npy"))                  # tree_genera order
idx = {g: i for i, g in enumerate(genera)}

# --- weaning: use the auditable analysis column if present, else the raw
#     PanTHERIA column with the documented Cercocebus imputation as fallback.
#     Provenance for imputed values is recorded in the dataset columns
#     `weaning_source` and `weaning_selection_rule` (see data dictionary).
if "weaning_d_analysis" in df.columns:
    WEAN = "weaning_d_analysis"
else:
    WEAN = "weaning_d_PanTHERIA_median"
    df.loc[df.genus == "Cercocebus", WEAN] = 270.0  # ADW/AnAge, ~9 months
INTEL = "intelligence_Deaner2006"
BRAIN = "total_brain_vol_mm3_Dunbar1992"

# ---------------------------------------------------------------- GLS helpers
def sub_C(sub_genera):
    ii = [idx[g] for g in sub_genera]
    return C_full[np.ix_(ii, ii)]

def lambda_transform(C, lam):
    d = np.diag(np.diag(C))
    return lam * C + (1 - lam) * d          # Pagel's lambda on off-diagonals

def gls_fit(y, X, C):
    Ci = np.linalg.inv(C)
    XtCiX = X.T @ Ci @ X
    beta = np.linalg.solve(XtCiX, X.T @ Ci @ y)
    resid = y - X @ beta
    n, p = X.shape
    s2 = float(resid.T @ Ci @ resid) / (n - p)
    cov = s2 * np.linalg.inv(XtCiX)
    se = np.sqrt(np.diag(cov))
    return beta, se, n, p

def gls_negloglik(lam, y, X, C):
    Cl = lambda_transform(C, lam)
    Ci = np.linalg.inv(Cl)
    n, p = X.shape
    beta = np.linalg.solve(X.T @ Ci @ X, X.T @ Ci @ y)
    resid = y - X @ beta
    s2 = float(resid.T @ Ci @ resid) / n
    sign, logdet = np.linalg.slogdet(Cl)
    ll = -0.5 * (n * np.log(2 * np.pi * s2) + logdet + n)
    return -ll

def fit_ml_lambda(y, X, C):
    res = minimize_scalar(gls_negloglik, bounds=(0.0, 1.0), method="bounded",
                          args=(y, X, C))
    lam = float(res.x)
    beta, se, n, p = gls_fit(y, X, lambda_transform(C, lam))
    return lam, beta, se, n, p

def zscore(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / a.std(ddof=0)

def report(name, beta, se, n, p, lam=None, one_tailed=False):
    tval = beta[-1] / se[-1]          # last coef = predictor of interest
    dfree = n - p
    if one_tailed:
        pv = tdist.sf(abs(tval), dfree)
    else:
        pv = 2 * tdist.sf(abs(tval), dfree)
    tag = f"lambda={lam:.2f}  " if lam is not None else ""
    tail = "1-tail" if one_tailed else "2-tail"
    print(f"  {name:38s} beta={beta[-1]:+.3f}  t={tval:+.2f}  "
          f"p({tail})={pv:.4f}  n={n}  {tag}")
    return pv


if __name__ == "__main__":
    # ============================================================ 1. Kendall tau
    print("=== 1. Headline Kendall tau (intelligence vs weaning) ===")
    for label, keep in [("n=21 (P&K genera, no ADW fills)", df.genus != "Cercocebus"),
                        ("n=22 (+ Cercocebus ADW weaning)", df.genus == df.genus)]:
        d = df[keep].dropna(subset=[WEAN, INTEL])
        tau, p = kendalltau(d[WEAN], d[INTEL])
        print(f"  {label:34s} tau={tau:.3f}  p={p:.4f}  (n={len(d)})")
    print("  [P&K reported tau = 0.62]")

    # ============================================================ 2. PGLS headline
    print("\n=== 2. PGLS: intelligence ~ weaning (standardized) ===")
    d = df.dropna(subset=[WEAN, INTEL]).copy()
    g = d.genus.tolist()
    C = sub_C(g)
    y = zscore(d[INTEL].values)
    X = np.column_stack([np.ones(len(d)), zscore(d[WEAN].values)])

    b_bm, se_bm, n, p = gls_fit(y, X, C)                       # Brownian lambda=1
    report("Brownian (lambda=1)", b_bm, se_bm, n, p, lam=1.0)
    lam, b_ml, se_ml, n, p = fit_ml_lambda(y, X, C)           # ML lambda
    report("Pagel ML lambda", b_ml, se_ml, n, p, lam=lam)
    print("  [P&K reported standardized beta = 0.78, t = 3.32]")

    # ============================================================ 3. Brain control
    print("\n=== 3. Single-brain-control PGLS: intelligence ~ weaning + total_brain ===")
    d2 = df.dropna(subset=[WEAN, INTEL, BRAIN]).copy()
    g2 = d2.genus.tolist()
    C2 = sub_C(g2)
    y2 = zscore(d2[INTEL].values)
    X2 = np.column_stack([np.ones(len(d2)),
                          zscore(d2[WEAN].values),
                          zscore(np.log(d2[BRAIN].values))])
    lam2, b2, se2, n2, p2 = fit_ml_lambda(y2, X2, C2)
    # weaning coef is column 1; report both weaning and brain
    tw = b2[1]/se2[1]; tb = b2[2]/se2[2]; dfree = n2-p2
    print(f"  weaning     beta={b2[1]:+.3f} t={tw:+.2f} "
          f"p2={2*tdist.sf(abs(tw),dfree):.4f}  (lambda={lam2:.2f}, n={n2})")
    print(f"  log(brain)  beta={b2[2]:+.3f} t={tb:+.2f} "
          f"p2={2*tdist.sf(abs(tb),dfree):.4f}")

    # ============================================================ 4. one vs two tail
    print("\n=== 4. One-tailed vs two-tailed (headline single-control weaning) ===")
    p_one = tdist.sf(abs(tw), dfree); p_two = 2*p_one
    print(f"  weaning under brain control: p(1-tail)={p_one:.4f}  p(2-tail)={p_two:.4f}")
    print("  (P&K report one-tailed throughout.)")

    # ============================================================ 5. Multi-covariate collinearity / VIF
    # This section reproduces the collinearity result reported in Table 1 of the
    # manuscript (predictor correlation r and variance inflation factors) for the
    # Table S1 style multi-covariate design (weaning + log brain + log neonate mass).
    # Brain and neonatal mass are log-transformed before correlation/VIF, matching
    # the single-brain-control model above.
    NEO = "neonate_body_mass_g_median"

    print("\n=== 5. Multi-covariate design: collinearity and VIF (Table 1 bottom row) ===")
    d3 = df.dropna(subset=[WEAN, INTEL, BRAIN, NEO]).copy()
    n3 = len(d3)
    P = np.column_stack([
        zscore(d3[WEAN].values),
        zscore(np.log(d3[BRAIN].values)),
        zscore(np.log(d3[NEO].values)),
    ])
    pred_names = ["weaning", "log_brain", "log_neonate_mass"]

    # pairwise correlations among predictors
    corr = np.corrcoef(P, rowvar=False)
    r_brain_neo = corr[1, 2]
    print(f"  n = {n3}")
    print(f"  corr(log_brain, log_neonate_mass) = {r_brain_neo:.3f}")
    print(f"  corr(weaning, log_brain)          = {corr[0,1]:.3f}")
    print(f"  corr(weaning, log_neonate_mass)   = {corr[0,2]:.3f}")

    # variance inflation factors: VIF_j = 1 / (1 - R^2_j) from regressing predictor j on the others
    def vif(mat, j):
        y_j = mat[:, j]
        X_j = np.column_stack([np.ones(len(mat))] + [mat[:, k] for k in range(mat.shape[1]) if k != j])
        beta_j, *_ = np.linalg.lstsq(X_j, y_j, rcond=None)
        resid = y_j - X_j @ beta_j
        ss_res = float(resid @ resid)
        ss_tot = float(((y_j - y_j.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot
        return 1.0 / (1.0 - r2)

    vifs = {nm: vif(P, j) for j, nm in enumerate(pred_names)}
    for nm in pred_names:
        print(f"  VIF[{nm:16s}] = {vifs[nm]:6.1f}")
    max_vif = max(vifs.values())
    print(f"  max VIF = {max_vif:.1f}")

    # assertions against the values reported in the manuscript / Table 1
    assert n3 == 18, f"expected n=18, got {n3}"
    assert round(r_brain_neo, 2) == 0.99, f"expected r=.99, got {r_brain_neo:.3f}"
    assert max_vif > 45, f"expected max VIF > 45, got {max_vif:.1f}"
    print("  [assertions passed: n = 18, r = .99, VIF > 45 -> matches Table 1]")
