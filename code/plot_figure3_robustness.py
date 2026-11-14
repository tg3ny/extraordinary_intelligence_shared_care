"""Figure 3 (robustness): reduction in the private selection gradient across gamma and V,
under a static group-scaling (public-good) formulation of shared care, the runaway
basin under a dynamic public good in which care tracks the group's evolving intelligence,
and the same basin when helpers are kin of the infant (helper intelligence correlated
with the infant's lineage).
Panels A and B report the conservative absolute gradient dF/dR; the relative gradient's
reduction is exactly (R/(R+A))^2 and does not vary with gamma or V.

The explanation of why the four V curves coincide lives in the manuscript caption,
not in the plot area. Typography follows figstyle.py (>= 9.5 pt everywhere).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs
from model_tightening import gradient_fixedA, gradient_groupscale
from public_good_dynamic import basin_by_seed, KAPPAS
from kin_structured_care import basin_by_seed as kin_basin

fs.apply()
ACC, BLUE, GREY, DGREY = fs.ACC, fs.BLUE, fs.GREY, fs.DGREY

fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(10.8, 9.0))

# --- A: reduction vs gamma. Four V curves coincide; draw them faint beneath a bold curve
#        so a reader can see none are hidden. Explanation goes in the caption. ---
gammas = np.linspace(0.15, 0.85, 25)
Vs = [5.0, 5.48, 6.0, 6.5]
curves = {V: [100 * (1 - gradient_fixedA(16.0, gm, V) / gradient_fixedA(0.0, gm, V))
              for gm in gammas] for V in Vs}
for V in Vs:
    axA.plot(gammas, curves[V], lw=4.0, color="#cccccc", alpha=0.95, zorder=1)
axA.plot(gammas, curves[5.48], lw=2.2, color=ACC, zorder=2)

axA.axvline(0.4, color=GREY, ls="--", lw=1.1, zorder=0)
t = axA.annotate("γ = 0.4\n(published value)", xy=(0.4, 62), xytext=(0.52, 78),
                 fontsize=fs.ANNOT, color=DGREY, va="center", ha="left", zorder=4,
                 arrowprops=dict(arrowstyle="->", color=DGREY, lw=1.2))
fs.boxed(t)

axA.legend(handles=[
    Line2D([0], [0], color=ACC, lw=2.4, label="V = 5.0 to 6.5"),
    Line2D([0], [0], color="#cccccc", lw=4.0, label="individual V curves"),
], loc="lower left", fontsize=fs.LEGEND, frameon=True, framealpha=0.95,
   edgecolor="none", handletextpad=0.6, borderaxespad=0.8)

axA.set_xlabel("childhood-mortality parameter  γ")
axA.set_ylabel("% reduction in private\ngradient dF/dR")
axA.set_title("Strong across γ; invariant to V", fontsize=fs.TITLE)
axA.set_ylim(0, 100)
fs.panel_letter(axA, "A", x=-0.20)

# --- B: group-scaling public good ---
kappas = np.linspace(0, 8, 30)
gb = gradient_groupscale(0.0)
red = [100 * (1 - gradient_groupscale(k) / gb) for k in kappas]
axB.plot(kappas, red, color=ACC, lw=2.4)
axB.fill_between(kappas, red, color=ACC, alpha=0.12)
axB.set_xlabel("shared-care scaling  κ")
axB.set_ylabel("% reduction in private\ngradient dF/dR")
axB.set_title("Static public good", fontsize=fs.TITLE)
axB.set_ylim(0, 100)
fs.panel_letter(axB, "B", x=-0.20)

# --- C: dynamic public good, A_t = kappa * mean(I_t) each generation ---
means, sds = [], []
for k in KAPPAS:
    fr, _ = basin_by_seed(k)
    means.append(fr.mean()); sds.append(fr.std(ddof=1))
axC.errorbar(KAPPAS, means, yerr=sds, fmt="o-", color=fs.BLUE, lw=2.2, ms=7,
             capsize=4, elinewidth=1.4)
axC.set_xlabel("shared-care scaling  κ")
axC.set_ylabel("% of starts\nthat run away")
axC.set_title("Dynamic public good", fontsize=fs.TITLE)
axC.set_ylim(0, max(means) * 1.35)
fs.panel_letter(axC, "C", x=-0.24)

# --- D: kin-structured care, A_i = kappa * [rho * Ip_i + (1 - rho) * mean(I_t)] ---
kin_k = (0.0, 1.0, 2.0, 4.0)
styles = [(0.0, fs.BLUE, "o-", "unrelated, r = 0"),
          (0.5, fs.DGREY, "s--", "r = 1/4"),
          (1.0, fs.ACC, "^:", "full siblings, r = 1/2")]
for rho, col, fmt, lab in styles:
    m, sd = [], []
    for k in kin_k:
        fr, _ = kin_basin(k, rho)
        m.append(fr.mean()); sd.append(fr.std(ddof=1))
    axD.errorbar(kin_k, m, yerr=sd, fmt=fmt, color=col, lw=2.0, ms=6.5, capsize=3.5,
                 elinewidth=1.2, label=lab)
axD.axhline(m[0], color=GREY, lw=0.8, ls=":")
axD.set_xlabel("shared-care scaling  κ")
axD.set_ylabel("% of starts\nthat run away")
axD.set_title("Relatedness of helpers", fontsize=fs.TITLE)
axD.set_ylim(0, 62)
axD.legend(loc="upper right", fontsize=fs.LEGEND, frameon=True, framealpha=0.95,
           edgecolor="none", handletextpad=0.6, borderaxespad=0.6)
fs.panel_letter(axD, "D", x=-0.24)

fig.suptitle("Robustness of the shared-care result", y=1.01,
             fontsize=fs.SUPTITLE, weight="bold")
fig.tight_layout()
fig.savefig("figure3_robustness.png")
fig.savefig("figure3_robustness.pdf")
print("saved figure3_robustness.png (+pdf)")
