"""Final MSSP Figure 5: frozen unit-disjoint Paderborn evaluation.

All numerical values are taken from the registered Cell 12 and Cell 12B
outputs. The script changes presentation only and does not alter outcomes.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# -----------------------------------------------------------------------------
# Output path
# -----------------------------------------------------------------------------
try:
    OUT = Path(MSSP_DIRS["figures_main"])  # noqa: F821 - notebook variable
except NameError:
    OUT = Path.cwd()
OUT.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Frozen Paderborn unit-disjoint results
# -----------------------------------------------------------------------------
PB_FWER = pd.DataFrame(
    {
        "scope": [
            "Overall",
            "N09_M07_F10",
            "N15_M01_F10",
            "N15_M07_F04",
            "N15_M07_F10",
        ],
        "rate": [0.0708, 0.0583, 0.0250, 0.1750, 0.0250],
        "ci_low": [0.0104, 0.0083, 0.0000, 0.0000, 0.0000],
        "ci_high": [0.1708, 0.1417, 0.0583, 0.5083, 0.0750],
    }
)

PB_PERFORMANCE = pd.DataFrame(
    [
        ["IR", "Detection", 0.4977, 0.3284, 0.6852],
        ["IR", "Localization", 0.8886, 0.8080, 0.9545],
        ["IR", "Correct diagnosis", 0.4477, 0.2614, 0.6511],
        ["OR", "Detection", 0.7153, 0.5115, 0.8917],
        ["OR", "Localization", 0.9416, 0.8822, 0.9885],
        ["OR", "Correct diagnosis", 0.7153, 0.5164, 0.8906],
    ],
    columns=["fault", "metric", "rate", "ci_low", "ci_high"],
)

CONDITIONS = ["N09_M07_F10", "N15_M01_F10", "N15_M07_F04", "N15_M07_F10"]
UNITS = ["K001", "K002", "K003", "K004", "K005", "K006"]
PB_UNIT_FWER = pd.DataFrame(
    [
        [0.05, 0.00, 0.00, 0.00],
        [0.00, 0.05, 0.05, 0.00],
        [0.00, 0.00, 0.00, 0.00],
        [0.05, 0.00, 0.00, 0.00],
        [0.25, 0.00, 0.00, 0.00],
        [0.00, 0.10, 1.00, 0.15],
    ],
    index=UNITS,
    columns=CONDITIONS,
)

assert PB_FWER["rate"].between(0, 1).all()
assert (PB_FWER["ci_low"] <= PB_FWER["rate"]).all()
assert (PB_FWER["rate"] <= PB_FWER["ci_high"]).all()
assert PB_PERFORMANCE["rate"].between(0, 1).all()
assert (PB_PERFORMANCE["ci_low"] <= PB_PERFORMANCE["rate"]).all()
assert (PB_PERFORMANCE["rate"] <= PB_PERFORMANCE["ci_high"]).all()


# -----------------------------------------------------------------------------
# Style matched to final Fig. 6
# -----------------------------------------------------------------------------
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "font.size": 10.5,
        "axes.labelsize": 12,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
        "savefig.dpi": 600,
    }
)

BLUE = "#4C78A8"
GREEN = "#59A14F"
GOLD = "#F2C14E"
RED = "#E45756"


def finish_axes(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")


def panel_letter(ax, label, y=-0.245):
    ax.text(
        0.5,
        y,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        clip_on=False,
    )


fig = plt.figure(figsize=(13.4, 9.7))
gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[1.0, 1.18],
    left=0.075,
    right=0.965,
    top=0.915,
    bottom=0.105,
    wspace=0.28,
    hspace=0.73,
)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, :])


# -----------------------------------------------------------------------------
# (a) Overall and condition-specific healthy FWER with cluster-aware intervals
# -----------------------------------------------------------------------------
x = np.arange(len(PB_FWER))
rate = PB_FWER["rate"].to_numpy()
yerr = np.vstack(
    [
        rate - PB_FWER["ci_low"].to_numpy(),
        PB_FWER["ci_high"].to_numpy() - rate,
    ]
)
colors = [BLUE, BLUE, BLUE, RED, BLUE]

ax_a.errorbar(
    x,
    rate,
    yerr=yerr,
    fmt="none",
    ecolor="black",
    elinewidth=1.6,
    capsize=5,
    capthick=1.6,
    zorder=2,
)
ax_a.scatter(
    x,
    rate,
    s=125,
    c=colors,
    edgecolors="black",
    linewidths=1.2,
    zorder=3,
)
ax_a.axvline(0.5, color="0.70", linewidth=1.3, zorder=1)
ax_a.hlines(0.075, -0.42, 0.42, color="black", linestyle="--", linewidth=1.7)
ax_a.hlines(0.10, 0.58, 4.42, color=RED, linestyle=":", linewidth=2.0)

ax_a.set_xticks(x)
ax_a.set_xticklabels(
    ["Overall", "N09\nM07 F10", "N15\nM01 F10", "N15\nM07 F04", "N15\nM07 F10"]
)
ax_a.set_ylabel("Healthy family-wise error rate")
ax_a.set_xlabel("Evaluation scope")
ax_a.set_xlim(-0.52, 4.52)
ax_a.set_ylim(0, 0.55)
ax_a.set_yticks(np.arange(0, 0.551, 0.10))
ax_a.legend(
    handles=[
        Line2D([], [], color="black", linestyle="--", linewidth=1.7,
               label="Overall limit 0.075"),
        Line2D([], [], color=RED, linestyle=":", linewidth=2.0,
               label="Conditional limit 0.10"),
    ],
    loc="lower center",
    bbox_to_anchor=(0.5, 1.045),
    ncol=2,
    columnspacing=1.0,
)
finish_axes(ax_a)
panel_letter(ax_a, "(a)")


# -----------------------------------------------------------------------------
# (b) Detection, localization, and correct diagnosis by physical fault class
# -----------------------------------------------------------------------------
metrics = ["Detection", "Localization", "Correct diagnosis"]
metric_colors = [BLUE, GREEN, GOLD]
centers = np.arange(2)
width = 0.23

for j, (metric, color) in enumerate(zip(metrics, metric_colors)):
    d = (
        PB_PERFORMANCE[PB_PERFORMANCE["metric"] == metric]
        .set_index("fault")
        .reindex(["IR", "OR"])
        .reset_index()
    )
    xpos = centers + (j - 1) * width
    ax_b.bar(
        xpos,
        d["rate"],
        width,
        color=color,
        edgecolor="black",
        linewidth=1.1,
        yerr=np.vstack(
            [
                d["rate"] - d["ci_low"],
                d["ci_high"] - d["rate"],
            ]
        ),
        capsize=4,
        error_kw={"elinewidth": 1.4, "ecolor": "black", "capthick": 1.4},
        label=metric,
        zorder=3,
    )

ax_b.axhline(0.80, color="black", linestyle="--", linewidth=1.5)
ax_b.axhline(0.75, color="0.40", linestyle="-.", linewidth=1.4)
ax_b.axhline(0.70, color="0.40", linestyle=":", linewidth=1.6)
ax_b.set_xticks(centers)
ax_b.set_xticklabels(["Inner race", "Outer race"])
ax_b.set_ylabel("Probability")
ax_b.set_xlabel("Fault class")
ax_b.set_ylim(0, 1.05)
ax_b.set_yticks(np.arange(0, 1.01, 0.2))
ax_b.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.045),
    ncol=3,
    columnspacing=1.0,
    handlelength=1.6,
)
finish_axes(ax_b)
panel_letter(ax_b, "(b)")


# -----------------------------------------------------------------------------
# (c) Held-out healthy unit-condition false-alarm map
# -----------------------------------------------------------------------------
m = PB_UNIT_FWER.to_numpy()
im = ax_c.imshow(m, cmap="Reds", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
ax_c.set_xticks(np.arange(len(CONDITIONS)))
ax_c.set_xticklabels([c.replace("_", " ") for c in CONDITIONS])
ax_c.set_yticks(np.arange(len(UNITS)))
ax_c.set_yticklabels(UNITS)
ax_c.set_xlabel("Operating condition")
ax_c.set_ylabel("Held-out healthy bearing")

for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        ax_c.text(
            j,
            i,
            f"{m[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if m[i, j] >= 0.50 else "black",
            fontsize=11,
            fontweight="bold",
        )

cb = fig.colorbar(im, ax=ax_c, fraction=0.021, pad=0.015)
cb.set_label("False-alarm fraction", fontweight="bold")
for tick in cb.ax.get_yticklabels():
    tick.set_fontweight("bold")
finish_axes(ax_c)
panel_letter(ax_c, "(c)", y=-0.205)


FIGURE_PATH = OUT / "Fig05_paderborn_unit_disjoint_final.png"
fig.savefig(
    FIGURE_PATH,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
    pad_inches=0.12,
)
plt.close(fig)

print("Saved:", FIGURE_PATH)
print("Pixel-ready PNG; all plotted values are frozen Cell 12/12B results.")

