# Extraordinary intelligence and the shared care of infants

Reproducibility package for the manuscript *Extraordinary intelligence and the shared care of
infants* (T. Grossmann, 2026). It revisits the Piantadosi and Kidd (2016)
runaway selection model of human intelligence and extends it with a single term for shared care.

Release: v1.1.0 (14 November 2026). A versioned archive with a Zenodo DOI will be deposited upon acceptance of the manuscript; until then this repository is the citable record.

## What is here

| Path | Contents |
|---|---|
| `code/` | All analysis and figure scripts (Python 3.11+). Key scripts: `pk_model.py` (reference model), `fast_sim.py` (vectorized simulation), `analytic_care.py` (closed-form relative and absolute gradients, dimensionless calibration, component decomposition, functional form of care), `sim_benchmark.py` (validation of the vectorized simulation against the sequential reference: endpoint agreement, final trait values, basin fractions by seed), `model_tightening.py` (robustness across gamma, V, and the static group-scaling public-good variant), `public_good_dynamic.py` (dynamic public good in which care tracks the group's evolving mean intelligence), `basin_sensitivity.py` (seed-level uncertainty of the runaway basin and sensitivity to threshold, generations, population size, and grid), `kin_structured_care.py` (care from relatives: helper intelligence correlated with the infant's lineage, under the published replacement scheme and a standard Wright-Fisher scheme), `small_brain_early_birth.py` (survival at early birth for small versus large brains under private and shared care, the values quoted in the fossil section), `reanalysis_pgls.py` (comparative reanalysis: Kendall's tau, PGLS under Brownian and Pagel's lambda, brain control, collinearity and VIF), `plot_figures.py`, `plot_figure3_robustness.py`, `plot_figure4_bodybrain.py`, `figstyle.py`. |
| `data/` | Assembled comparative dataset with auditable provenance columns, the phylogenetic variance-covariance matrix and genus order, third-party fossil datasets (attributed in `LICENSE-DATA`), `DATA_DICTIONARY.md`, and `PROVENANCE.md` (sources, processing, imputation rules, and the 10kTrees pruning). |
| `figures/` | The four manuscript figures as 400-dpi PNG, with vector PDF versions in `figures/vector/`. |
| `reproduce.sh` | One command that runs every analysis and regenerates every figure. |
| `requirements.txt`, `requirements-lock.txt` | Dependencies; the lock file pins the exact versions used. |

## Reproduce everything

```bash
pip install -r requirements-lock.txt
bash reproduce.sh
```

Regenerated figures are written to `figures/regenerated/`. The analysis scripts contain embedded
assertions that reproduce the values reported in the manuscript (for example Kendall's tau = 0.635
and 0.633, the predictor correlation r = 0.99 with VIF > 45, and the agreement between the analytic
and numerical gradients to within 1e-9). A failed assertion signals a reproduction mismatch.

## Key results the scripts reproduce

- Closed form: the relative (log-fitness) return on parental intelligence scales exactly as
  1 / (Ip + A)^2; the absolute gradient dF/dR falls less because pooled care raises baseline survival.
- Dimensionless calibration (a = A / Ip): relative return reduced by 75% at a = 1, 88.9% at a = 2,
  97.5% at a = 5.3; absolute gradient reduced by 29.6%, 55.8%, 85.7%.
- Functional form: the return falls under additive, multiplicative (50%, 67% at a = 1, 2), and
  interactive (84%, 91%) pooling.
- Runaway basin (6 x 6 grid, 10 seeds): 30.6% (SD 2.6 across seeds) at A = 0 to 19.4% (SD 3.5) at A = 8;
  the 11 to 13 point contraction is stable across threshold, generations, population size, and grid.
- Dynamic public good (A = kappa x mean group intelligence each generation): basin 30.6% at kappa = 0,
  24.4%, 20.3%, 17.8% at kappa = 1, 2, 4.
- Relatedness of helpers: the relative return is (1 + 2 rbar a)/(1 + a)^2 of the private-care value,
  at most 1/(1 + a); reductions at a = 1 (2) are 75% (89%) unrelated, 62.5% (78%) for r = 1/4,
  50% (67%) for full siblings. In simulation the basin narrows with unrelated helpers, changes little
  with r = 1/4 helpers (26.4%, 23.3% at kappa = 2, 4), and does not narrow with full siblings only
  (35.6%, 31.7%); the same holds under a Wright-Fisher scheme.
- Simulation validation: 59 of 60 paired runs agree in endpoint classification; final brain size and
  birth age correlate at r = .97 and .99 between implementations.
- Early birth (T = 4) at brain size 2 versus 5: survival 0.04 versus 0.29 under private care (A = 0),
  0.53 versus 0.62 under shared care (A = 8).
- Comparative reanalysis: tau = 0.63; PGLS beta = 1.00 (Brownian) and 0.87 (lambda = 0); brain control
  beta = 0.64 [0.19, 1.09], P = 0.014; collinearity r = 0.99, VIF 54 and 47.

## Licenses

Code is released under the MIT License (`LICENSE`). Data, figures, and documentation are released
under CC BY 4.0 (`LICENSE-DATA`), with third-party data redistributed under their original terms.

## Citation

See `CITATION.cff`. Please cite the article and the archived software DOI.
