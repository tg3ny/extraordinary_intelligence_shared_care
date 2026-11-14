"""
figstyle.py - shared typography and palette for all manuscript figures.

Enforces a print-safe minimum text size. Authoring sizes are set larger than the
9.5 pt floor because the figures are placed in the manuscript at roughly 0.55x of their
authoring canvas width; at that reduction these sizes render at about 9-10 pt. Journal figures are reduced to
single-column (~8.7 cm) or 1.5-column widths; anything set below ~9 pt at
authoring size becomes unreadable after scaling. Every figure script imports
this so that axis labels, tick labels, panel letters, legends, and in-plot
annotations are consistent in size and weight across the manuscript.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- typography floor (points) ---
BASE      = 17   # axis labels, tick labels
TICK      = 15.5  # tick labels
LEGEND    = 15.5  # legend text
ANNOT     = 15.5  # in-plot annotations (minimum)
PANEL     = 19   # panel letters A, B, C, D
TITLE     = 18   # panel titles
SUPTITLE  = 21 # figure suptitle

# --- palette (perceptually uniform, colorblind-safe) ---
INK   = "#1a1a2e"
ACC   = "#c1440e"   # burnt orange
BLUE  = "#20486b"   # deep blue
TEAL  = "#177e6f"
GREY  = "#8a8a8a"
DGREY = "#4a4a4a"   # darker grey for readable annotation text
SAND  = "#f2e9dc"
SEQ   = "magma"     # sequential, perceptually uniform


def apply():
    """Apply the shared rcParams. Call once at the top of each figure script."""
    mpl.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         BASE,
        "axes.titlesize":    TITLE,
        "axes.labelsize":    BASE,
        "xtick.labelsize":   TICK,
        "ytick.labelsize":   TICK,
        "legend.fontsize":   LEGEND,
        "figure.titlesize":  SUPTITLE,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelpad":     6.0,    # padding between axis label and ticks
        "axes.titlepad":     8.0,
        "legend.frameon":    False,
        "legend.handlelength": 1.6,
        "axes.linewidth":    1.2,
        "lines.linewidth":   2.6,
        "figure.dpi":        130,
        "savefig.dpi":       400,    # print-quality raster
        "savefig.bbox":      "tight",
        "pdf.fonttype":      42,     # embed TrueType (editable vector text)
        "ps.fonttype":       42,
    })


def panel_letter(ax, letter, x=-0.14, y=1.06):
    """Consistent bold panel designator in axes coordinates."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL,
            fontweight="bold", va="bottom", ha="left", color=INK)


def boxed(text_obj):
    """Give an annotation a white bounding box so it does not collide with lines."""
    text_obj.set_bbox(dict(facecolor="white", edgecolor="none",
                           alpha=0.85, boxstyle="round,pad=0.25"))
    return text_obj
