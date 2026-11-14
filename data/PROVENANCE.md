# Data provenance

This file documents how each data file was assembled, so that the comparative
reanalysis can be audited and regenerated from primary sources. The comparative
dataset is a reconstruction from the sources named in Piantadosi and Kidd (2016),
not a copy of their working file.

## childhood_first_datamatrix_COMPLETE.csv (23 primate genera)

Genera: the 23 genera analyzed by Piantadosi and Kidd (2016), as listed in their
supplementary material.

Weaning age (days). PanTHERIA v1.0 (WR05_Aug2008; Jones et al. 2009, Ecology 90,
2648), filtered to Primates (376 species, 69 genera), missing code -999 treated as
missing, genus-level medians computed across species (the original study's stated
procedure). The full primate extract is `pantheria_primates.tsv`; per-genus
coverage is in `weaning_by_genus.csv`. Two genera have no PanTHERIA weaning
value, Cercocebus and Presbytis, which are the two genera the original study
reports filling from Animal Diversity Web. For Cercocebus a genus-level value of
270 days (9 months) is used in `weaning_d_analysis`, the midpoint of the 7 to 12
month range reported across species in Animal Diversity Web (C. chrysogaster 8 to
9 months, C. torquatus 7 to 10 months) and AnAge (C. galeritus 12 months); the
source and rule are recorded in `weaning_source` and `weaning_selection_rule`. No
defensible genus-level value was found for Presbytis, which is left missing and
omitted from weaning analyses. The raw PanTHERIA medians are retained in
`weaning_d_PanTHERIA_median`.

Adult body mass, neonatal body mass, gestation length. PanTHERIA genus medians,
computed as above.

Intelligence composite. The global cognitive variable (theta) of Deaner, van Schaik
and Johnson (2006, Evolutionary Psychology 4, 149-196), stored as
`intelligence_Deaner2006_theta`; `intelligence_Deaner2006` = -theta so that higher
values mean greater ability. Genus values were digitized from the published figure
because the article does not tabulate them. Digitization introduces small
measurement noise (on the order of 0.01 to 0.02 units) that does not affect the
rank-based headline statistic; the PGLS slopes should be read with this in mind.

Neocortex and total brain volume. Dunbar (1992, Journal of Human Evolution 22,
469-493), transcribed from the published table, with the correction noted by the
original study (Miopithecus total brain volume 37,760 mm^3). Values are available
for 18 of the 23 genera.

## Phylogeny: tree_genera.csv and phylo_C.npy

Source tree: 10kTrees Primates, Version 3, consensus tree (Arnold, Matthews and
Nunn 2010, Evolutionary Anthropology 19, 114-118; https://10ktrees.nunn-lab.org).
The consensus tree was pruned to one species per genus for the 23 genera, and the
ultrametric phylogenetic variance-covariance matrix (shared branch length from the
root) was computed from the pruned tree and saved as `phylo_C.npy`, in the genus
order given by `tree_genera.csv`. The Newick/NEXUS source file
(TreeBlock_10kTrees_Primates_Version3.nex) is distributed by 10kTrees under its
own terms and is not redistributed here; the matrix is sufficient to rerun every
PGLS model in the manuscript. Differences between the reanalysis slopes and the
original study's (for example, beta = 1.00 versus 0.78 under Brownian motion)
are consistent with the use of a single consensus tree, 21 to 22 rather than 23
genera, and digitized intelligence values.

## Fossil data

`puschel2024_specimen_data_sd04.xlsx`: Dataset S4 of Puschel et al. (2024, PNAS
121, e2409542121), open access (CC BY), reused unchanged with original
attribution; each specimen is placed at the midpoint of its estimated age range
by `code/plot_figure4_bodybrain.py`.

`gardner_taxon_bodymass.csv`: taxon-level body-mass means read from Gardner et al.
(2026, PNAS 123, e2521732123). The August 2026 correction to that article concerns
the conversion of the fitted log-scale trend to kg per million years and does not
affect the taxon means used here.

`figure4_taxon_summary.csv`: convenience output of the Figure 4 script (taxon
medians), not an input to any analysis.
