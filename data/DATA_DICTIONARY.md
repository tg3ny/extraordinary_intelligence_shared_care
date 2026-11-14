# Data dictionary

Sources, processing steps, and imputation rules for every file are documented in PROVENANCE.md.

## childhood_first_datamatrix_COMPLETE.csv (one row per primate genus, n = 23)

| Column | Definition | Units | Missing-value rule |
|---|---|---|---|
| genus | Primate genus | - | - |
| weaning_d_PanTHERIA_median | Raw genus-median weaning age from PanTHERIA (Jones et al. 2009) | days | blank where PanTHERIA has no value (Cercocebus, Presbytis) |
| weaning_d_analysis | Weaning value actually used in analysis | days | blank only where no defensible value exists (Presbytis) |
| weaning_source | Source of the analysis weaning value | - | blank where no value |
| weaning_selection_rule | Rule used to derive/select an imputed value | - | blank for direct PanTHERIA values |
| adult_body_mass_g_median | Adult body mass (PanTHERIA median) | grams | - |
| neonate_body_mass_g_median | Neonatal body mass (PanTHERIA median) | grams | - |
| gestation_d_median | Gestation length (PanTHERIA median) | days | - |
| intelligence_Deaner2006 | Composite domain-general cognition score | dimensionless (= -theta) | - |
| intelligence_Deaner2006_theta | Underlying theta parameter (higher = less able) | dimensionless | - |
| neocortex_Dunbar1992 | Neocortex measure (Dunbar 1992) | ratio | blank where unavailable |
| total_brain_vol_mm3_Dunbar1992 | Total brain volume (Dunbar 1992) | mm^3 | blank where unavailable (5 genera) |
| source_note | Free-text provenance note | - | - |

Notes.
- Intelligence values were recovered from the published figure in Deaner et al. (2006) by
  digitization; this introduces minor measurement noise but does not affect the rank-based
  headline statistic.
- Cercocebus weaning (270 d) is imputed from Animal Diversity Web / AnAge (~9 months, midpoint of a
  7-12 month range); Presbytis has no defensible weaning value and is omitted from weaning analyses.
- Brain-based models (single-brain control, collinearity check) use the 18 genera with brain values.

## Other data files
- tree_genera.csv : genus order used to index the phylogenetic matrix.
- phylo_C.npy     : ultrametric phylogenetic variance-covariance matrix (23 x 23), computed from the
                    10kTrees Version 3 consensus tree pruned to one species per genus, in
                    tree_genera.csv order. Reruns PGLS but is not a substitute for the source tree;
                    see PROVENANCE.md for the source and pruning procedure.
- puschel2024_specimen_data_sd04.xlsx : reused third-party data, Puschel et al. (2024) Dataset S4
                    (open access, CC BY). Retains original attribution.
- gardner_taxon_bodymass.csv : taxon-level body-mass means derived from Gardner et al. (2026).
- weaning_by_genus.csv, pantheria_primates.tsv : intermediate/source extracts.
- figure4_taxon_summary.csv : convenience summary of Figure 4 taxon medians (not required to run any analysis).
