"""
Figure 4 - "Bodies and brains both intensify late."

Two stacked panels sharing a time axis:
  A) body mass (kg) through time
  B) cranial capacity (cm3) through time

Primary data: Puschel et al. 2024 (PNAS) Dataset S4, specimen-level cranial
capacities and body masses, ages in ka converted to Ma (specimens placed at the
midpoint of their dated range). Body panel also overlays Gardner et al. 2026
taxon means as an independent cross-check.

The figure shows absolute trajectories. Puschel's headline claims (that the
increase is anagenetic / within-species and accelerating) come from their
Bayesian mixed models, not from this scatter; the scatter shows the grade-level
pattern only. Annotations are kept to that honest scope.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs

fs.apply()
INK, ACC, BLUE, GREY, SAND = fs.INK, fs.ACC, fs.BLUE, fs.GREY, fs.SAND

import os
# Resolve data from the package's ../data folder when present, else the local
# working directory. Keeps the script runnable both in development and from the
# distributed reproducibility package.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "data")

def _find(*names):
    for n in names:
        for cand in (os.path.join(_DATA, n), os.path.join(_HERE, n), n):
            if os.path.exists(cand):
                return cand
    raise FileNotFoundError(f"could not locate any of {names}")

S4   = _find("puschel2024_specimen_data_sd04.xlsx", "pnas_2409542121_sd04.xlsx",
             "/mnt/user-data/uploads/pnas_2409542121_sd04.xlsx")
GARD = _find("gardner_taxon_bodymass.csv")

CC = "Cranial capacity [cm3]"; BM = "Body mass [kg]"

# ---------------------------------------------------------------- load Puschel
df = pd.read_excel(S4)
df.columns = [c.strip() for c in df.columns]
for c in [CC, BM, "MinAge", "MaxAge"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["age"] = (df["MinAge"] + df["MaxAge"]) / 2 / 1000.0          # ka -> Ma
df["taxon"] = df["Taxonomic assignment"].str.split(";").str[0].str.strip()

# grade flags
NONHOMO = df["taxon"].str.startswith(("Australopithecus", "Paranthropus",
             "Ardipithecus", "Sahelanthropus", "Kenyanthropus"))
DWARF = df["taxon"].isin(["Homo naledi", "Homo floresiensis"])
df["grade"] = np.where(DWARF, "dwarf", np.where(NONHOMO, "early", "homo"))

def tax_summary(col):
    g = (df.dropna(subset=[col])
           .groupby("taxon")
           .agg(age=("age", "median"), val=(col, "median"),
                n=(col, "count"), grade=("grade", "first"))
           .sort_values("age", ascending=False))
    return g

bm_t = tax_summary(BM)
cc_t = tax_summary(CC)

# ---------------------------------------------------------------- Gardner overlay
gard = pd.read_csv(GARD)                    # Taxon, n, mass, mass_sd, age (Ma)

TRANS = 2.0    # Ma, approximate onset of the Homo body/brain expansion
GRADE_LO, GRADE_HI = 33, 46   # australopith-grade body-mass band (kg)

col = {"early": BLUE, "homo": ACC, "dwarf": "#177e6f"}

fig, (axA, axB) = plt.subplots(2, 1, figsize=(9.8, 9.2), sharex=True)

# ============================== PANEL A : body mass =========================
axA.axhspan(GRADE_LO, GRADE_HI, color=SAND, zorder=0)
axA.text(6.7, (GRADE_LO+GRADE_HI)/2, "australopith grade\n(~35-45 kg)",
         va="center", ha="left", fontsize=fs.ANNOT, color="#7a5c2e")
axA.axvline(TRANS, color=GREY, lw=1.0, ls="--", zorder=1)

# specimen cloud
for grd, sub in df.dropna(subset=[BM]).groupby("grade"):
    axA.scatter(sub["age"], sub[BM], s=12, color=col[grd], alpha=0.20,
                edgecolors="none", zorder=2)
# Gardner taxon means (independent cross-check)
axA.scatter(gard["age"], gard["mass"], marker="s", s=26, facecolors="none",
            edgecolors=INK, linewidths=0.9, alpha=0.75, zorder=3)
# Puschel taxon medians
for grd, sub in bm_t.groupby("grade"):
    axA.scatter(sub["age"], sub["val"], s=46, color=col[grd],
                edgecolors="white", linewidths=0.8, zorder=4)

axA.annotate("Homo ergaster / erectus:\nbody mass climbs to ~60-70 kg",
             xy=(1.44, 59.6), xytext=(3.3, 83), fontsize=fs.ANNOT, color=ACC,
             arrowprops=dict(arrowstyle="->", color=ACC, lw=1.8))
for name, x, y, dx, dy, ha in [("H. floresiensis", 0.08, 34.1, -0.05, -6.0, "right"),
                               ("H. naledi", 0.29, 35.9, 0.55, 6.5, "left")]:
    axA.annotate(name, xy=(x, y), xytext=(x+dx, y+dy), fontsize=fs.ANNOT,
                 color=col["dwarf"], style="italic", ha=ha, va="center",
                 arrowprops=dict(arrowstyle="->", color=col["dwarf"], lw=1.5,
                                 shrinkA=2, shrinkB=3))
axA.set_ylabel("Body mass  (kg)")
axA.set_ylim(25, 92)
axA.set_title("Body mass holds at australopith grade, then expands in Homo",
              loc="left")
fs.panel_letter(axA, "A", x=-0.085, y=1.02)

# ============================== PANEL B : cranial capacity ===================
axB.axvline(TRANS, color=GREY, lw=1.0, ls="--", zorder=1)
for grd, sub in df.dropna(subset=[CC]).groupby("grade"):
    axB.scatter(sub["age"], sub[CC], s=12, color=col[grd], alpha=0.20,
                edgecolors="none", zorder=2)
for grd, sub in cc_t.groupby("grade"):
    axB.scatter(sub["age"], sub["val"], s=46, color=col[grd],
                edgecolors="white", linewidths=0.8, zorder=4)

axB.annotate("acceleration through late Homo\n(heidelbergensis, neanderthals, sapiens)",
             xy=(0.14, 1347), xytext=(2.6, 1520), fontsize=fs.ANNOT, color=ACC,
             arrowprops=dict(arrowstyle="->", color=ACC, lw=1.8))
for name, x, y, dx, dy, ha in [("H. floresiensis", 0.08, 426, -0.05, -95, "right"),
                               ("H. naledi", 0.29, 560, 0.60, 175, "left")]:
    axB.annotate(name, xy=(x, y), xytext=(x+dx, y+dy), fontsize=fs.ANNOT,
                 color=col["dwarf"], style="italic", ha=ha, va="center",
                 arrowprops=dict(arrowstyle="->", color=col["dwarf"], lw=1.5,
                                 shrinkA=2, shrinkB=3))
_t = axB.text(TRANS - 0.06, 300, "~2 Ma", fontsize=fs.ANNOT, color=fs.DGREY,
              ha="right", va="bottom")
fs.boxed(_t)
axB.set_ylabel("Cranial capacity  (cm$^3$)")
axB.set_xlabel("Age  (Ma, past  \u2192  present)")
axB.set_ylim(200, 1750)
axB.set_title("Brain size is flat early, then rises and accelerates in Homo",
              loc="left")
fs.panel_letter(axB, "B", x=-0.085, y=1.02)

axB.set_xlim(7.6, -0.3)   # inverted: past on the left, present on the right

# ---- legend
handles = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=BLUE, markersize=9,
           label="pre-Homo"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=ACC, markersize=9,
           label="Homo"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=col["dwarf"],
           markersize=9, label="small-bodied Homo"),
    Line2D([0],[0], marker="s", color="w", markeredgecolor=INK, markersize=9,
           label="Gardner et al. (2026)"),
]
axA.legend(handles=handles, fontsize=fs.LEGEND, loc="upper left", frameon=True,
           framealpha=0.9, edgecolor="none", handletextpad=0.5, labelspacing=0.35,
           borderaxespad=0.7)

fig.suptitle("Large body mass and large brain size emerge late in hominin evolution",
             y=0.995, fontsize=fs.SUPTITLE, weight="bold")
# Source and method details live in the manuscript caption, not the plot.
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig("figure4_bodybrain_trajectory.png", bbox_inches="tight")
fig.savefig("figure4_bodybrain_trajectory.pdf", bbox_inches="tight")
print("saved figure4_bodybrain_trajectory.png")

# quick console readout of the grade shift
print("\nBody mass (kg): early grade median =",
      round(bm_t[bm_t.grade=='early'].val.median(),1),
      "| Homo (excl dwarf) median =",
      round(bm_t[bm_t.grade=='homo'].val.median(),1))
print("Cranial cap (cm3): early grade median =",
      round(cc_t[cc_t.grade=='early'].val.median(),0),
      "| Homo (excl dwarf) median =",
      round(cc_t[cc_t.grade=='homo'].val.median(),0))
