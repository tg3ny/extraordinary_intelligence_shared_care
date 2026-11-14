"""Generate the two model figures:
  Figure 1  - reproduction of Piantadosi & Kidd (2016)
  Figure 2  - cooperative care attenuates the childcare-driven runaway

Typography is governed by figstyle.py, which enforces a print-safe minimum
(>= 9.5 pt) for all in-plot text so the figures survive reduction to
single-column width. Methodological detail that would clutter the plot area
lives in the manuscript figure captions instead.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figstyle as fs
from pk_model import fitness
from fast_sim import simulate_fast

fs.apply()
INK, ACC, BLUE, GREY, SAND = fs.INK, fs.ACC, fs.BLUE, fs.GREY, fs.SAND


def landscape(A=0.0, Tmax=30, Rmax=10, ngrid=240):
    T = np.linspace(0.1, Tmax, ngrid)
    R = np.linspace(0.1, Rmax, ngrid)
    TT, RR = np.meshgrid(T, R)
    F = fitness(TT, RR, Ip=RR, allocare=A)
    return T, R, F


# ======================================================================
# FIGURE 1 : reproduction
# ======================================================================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

T, R, F = landscape(0.0)
im = axL.pcolormesh(T, R, F, cmap=fs.SEQ, shading="auto", vmin=0, vmax=1)
axL.set_xlabel("Birth age  T  (months)")
axL.set_ylabel("Brain size  R  (radius, cm)")
axL.set_title("Fitness landscape")
cb = fig.colorbar(im, ax=axL, fraction=0.046, pad=0.03)
cb.set_label("P(survive to adulthood)", fontsize=fs.BASE)
cb.ax.tick_params(labelsize=fs.TICK)
axL.annotate("big-brain ridge\n(runaway)", xy=(4.2, 8.4), xytext=(9.0, 6.6),
             color="white", fontsize=fs.ANNOT, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
fs.panel_letter(axL, "A")

T2, R2, F2 = landscape(0.0)
axR.contourf(T2, R2, (F2 > 0.5).astype(float), levels=[0.5, 1.5],
             colors=[SAND], alpha=0.9)
axR.contour(T2, R2, F2, levels=[0.5], colors=[GREY], linewidths=0.9)
starts = [(t, r) for t in (4, 7, 11, 16, 21) for r in (2, 4, 6)]
cmap = plt.get_cmap("viridis", len(starts))
for k, (T0, R0) in enumerate(starts):
    tr = simulate_fast(T0, R0, n=1000, generations=100, seed=3, record=True)
    axR.plot(tr[:, 0], tr[:, 1], color=cmap(k), lw=1.3, alpha=0.9)
    axR.plot(T0, R0, "o", color=cmap(k), ms=5, mec="k", mew=0.5)
    axR.plot(tr[-1, 0], tr[-1, 1], "*", color=cmap(k), ms=11, mec="k", mew=0.5)
axR.set_xlim(0, 30); axR.set_ylim(0, 10)
axR.set_xlabel("Birth age  T  (months)")
axR.set_ylabel("Brain size  R  (radius, cm)")
axR.set_title("Population dynamics")
fs.panel_letter(axR, "B")
axR.legend(handles=[
    Line2D([0], [0], marker="o", color="w", markerfacecolor=GREY, markeredgecolor="k",
           markersize=7, label="start"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor=GREY, markeredgecolor="k",
           markersize=12, label="end"),
], loc="upper right", fontsize=fs.LEGEND, frameon=True, framealpha=0.9,
   edgecolor="none", handletextpad=0.5)

fig.suptitle("Reproduction of the Piantadosi and Kidd (2016) model",
             y=1.02, fontsize=fs.SUPTITLE, weight="bold")
fig.tight_layout()
fig.savefig("figure1_reproduction.png")
fig.savefig("figure1_reproduction.pdf")
print("saved figure1_reproduction.png (+pdf)")


# ======================================================================
# FIGURE 2 : cooperative-care extension
# ======================================================================
fig2 = plt.figure(figsize=(11.5, 9.6))
gs = fig2.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.78, wspace=0.62)

axes_top = []
for col, A in enumerate([0.0, 4.0, 16.0]):
    ax = fig2.add_subplot(gs[0, col])
    T, R, F = landscape(A)
    im = ax.pcolormesh(T, R, F, cmap=fs.SEQ, shading="auto", vmin=0, vmax=1)
    ax.set_title(f"shared care  A = {A:.0f}", fontsize=fs.TITLE)
    ax.set_xlabel("Birth age  T")
    if col == 0:
        ax.set_ylabel("Brain size  R")
    ax.contour(T, R, F, levels=[0.5], colors=["white"], linewidths=0.9)
    axes_top.append(ax)
fs.panel_letter(axes_top[0], "A", x=-0.24, y=1.18)

axg = fig2.add_subplot(gs[1, 0])

def dF_dR(T, R, A, h=1e-3):
    return (fitness(T, R + h, Ip=R + h, allocare=A)
            - fitness(T, R - h, Ip=R - h, allocare=A)) / (2 * h)

def dlnF_dR(T, R, A, h=1e-3):
    return (np.log(fitness(T, R + h, Ip=R + h, allocare=A))
            - np.log(fitness(T, R - h, Ip=R - h, allocare=A))) / (2 * h)

As = np.linspace(0, 16, 33)
grad = np.array([np.mean([dF_dR(t, 3.0, A) for t in (3, 4, 5)]) for A in As])
rgrad = np.array([np.mean([dlnF_dR(t, 3.0, A) for t in (3, 4, 5)]) for A in As])
axg.plot(As, rgrad / rgrad[0], color=ACC, lw=2.4, label="relative")
axg.plot(As, grad / grad[0], color=BLUE, lw=2.4, ls="--", label="absolute")
axg.fill_between(As, rgrad / rgrad[0], color=ACC, alpha=0.10)
axg.set_xlabel("shared care  A")
axg.set_ylabel("private return on R\n(fraction of A = 0)")
axg.set_title("Return declines with care", fontsize=fs.TITLE)
axg.set_ylim(0, 1.05)
axg.axhline(0, color=GREY, lw=0.8, ls=":")
axg.legend(loc="center right", fontsize=fs.LEGEND, frameon=True, framealpha=0.95,
           edgecolor="none", handletextpad=0.6, borderaxespad=0.4, bbox_to_anchor=(1.04, 0.62))
fs.panel_letter(axg, "B", x=-0.34, y=1.14)

axf = fig2.add_subplot(gs[1, 1])
T0s = np.linspace(3, 22, 6); R0s = np.linspace(1, 6, 6)
Asw = [0, 1, 2, 4, 8, 12]; frac = []
for A in Asw:
    big = tot = 0
    for T0 in T0s:
        for R0 in R0s:
            for seed in range(10):
                Tf, Rf, If = simulate_fast(T0, R0, n=300, generations=50,
                                           allocare=A, seed=seed)
                big += (np.mean(Rf) > 4); tot += 1
    frac.append(100 * big / tot)
axf.plot(Asw, frac, "o-", color=BLUE, lw=2.2, ms=7)
axf.set_xlabel("shared care  A")
axf.set_ylabel("% of starts\nthat run away")
axf.set_title("Runaway basin narrows", fontsize=fs.TITLE)
axf.set_ylim(0, max(frac) * 1.18)
fs.panel_letter(axf, "C", x=-0.30, y=1.14)

axt = fig2.add_subplot(gs[1, 2])
ex_starts = [(4, 3), (6, 4), (8, 5)]
for (T0, R0) in ex_starts:
    tr0 = simulate_fast(T0, R0, n=400, generations=80, allocare=0.0, seed=2, record=True)
    tr8 = simulate_fast(T0, R0, n=400, generations=80, allocare=8.0, seed=2, record=True)
    axt.plot(tr0[:, 0], tr0[:, 1], color=ACC, lw=1.8, alpha=0.9)
    axt.plot(tr8[:, 0], tr8[:, 1], color=BLUE, lw=1.8, alpha=0.9)
    axt.plot(T0, R0, "ko", ms=5)
axt.legend(handles=[
    Line2D([0], [0], color=ACC, lw=2.4, label="A = 0"),
    Line2D([0], [0], color=BLUE, lw=2.4, label="A = 8"),
], loc="lower right", fontsize=fs.LEGEND, frameon=True, framealpha=0.95,
   edgecolor="none", handletextpad=0.6, borderaxespad=0.8)
axt.set_xlabel("Birth age  T")
axt.set_ylabel("Brain size R")
axt.set_title("With vs without care", fontsize=fs.TITLE)
axt.set_xlim(2, 12); axt.set_ylim(2, 8)
fs.panel_letter(axt, "D", x=-0.30, y=1.14)

fig2.suptitle("Cooperative care attenuates the childcare-driven runaway",
              y=1.01, fontsize=fs.SUPTITLE, weight="bold")
fig2.savefig("figure2_allocare.png")
fig2.savefig("figure2_allocare.pdf")
print("saved figure2_allocare.png (+pdf)")
print("absolute gradient A=0 -> A=16:", round(grad[0], 4), "->", round(grad[-1], 4),
      "| relative gradient:", round(rgrad[0], 4), "->", round(rgrad[-1], 4))
print("runaway fraction A=0 -> A=12:", round(frac[0], 1), "->", round(frac[-1], 1))
