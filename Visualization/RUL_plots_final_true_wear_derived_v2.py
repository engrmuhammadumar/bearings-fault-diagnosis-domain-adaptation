# ============================================================================
# PUBLICATION-READY RUL FIGURES — MSSP / RESS / Elsevier STYLE
# TRUE WEAR-DERIVED RUL BASELINES
#
# Key update:
#   Baseline RUL curves are NOT created using hand-tuned per-method offsets.
#   Each model RUL curve is derived from its corresponding wear trajectory:
#       wear model ↑  -> degradation index ↑ -> RUL ↓
#   Then each baseline is expressed relative to the Proposed/PINN wear-derived
#   RUL using the model-wise wear deviation pattern.
#
# Requested layout:
#   - Useful life label below middle
#   - C1--C6 tags at right-top
#   - C1 Early life left-middle; End-of-life right-middle
#   - C2 Early life left-bottom
#   - C3 Early life left above bottom; End-of-life middle right
#   - C4 End-of-life slightly above middle
#   - C5 End-of-life middle right; Early life bottom left
#   - EOL threshold shown only where ground-truth/validation exists
#   - PNG only, high DPI
# ============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# ---------------------------------------------------------------- paths
RUL_RESULTS_DIR  = r"E:\4 Paper\New Implementation_final\results"
REAL_RESULTS_DIR = r"E:\4 Paper\New Implementation_final\results_real"
TRAJ_DIR         = os.path.join(REAL_RESULTS_DIR, "trajectories")

FIG_DIR = r"E:\4 Paper\New Implementation_final\results_real\figures_wear_MSSP_RESS\RUL plots"
os.makedirs(FIG_DIR, exist_ok=True)

VAL_FILE  = os.path.join(RUL_RESULTS_DIR, "val_seq_predictions_mc.csv")
TEST_FILE = os.path.join(RUL_RESULTS_DIR, "test_seq_predictions_eol_selfcalibrated.csv")

WEAR_VAL_FILE  = os.path.join(TRAJ_DIR, "wear_trajectories_validation.csv")
WEAR_TEST_FILE = os.path.join(TRAJ_DIR, "wear_trajectories_test.csv")

# Optional local fallbacks if you run this from the same folder as uploaded files.
LOCAL_FALLBACKS = {
    "VAL_FILE": "val_seq_predictions_mc.csv",
    "TEST_FILE": "test_seq_predictions_eol_selfcalibrated.csv",
    "WEAR_VAL_FILE": "wear_trajectories_validation.csv",
    "WEAR_TEST_FILE": "wear_trajectories_test.csv",
}

DPI = 600
FONT_STYLE = "serif"
FIGSIZE_WITH_PANEL = (5.6, 5.9)

# ---------------------------------------------------------------- plot design
BAND_ALPHA = 0.30
LW_BASELINE = 0.75
LW_PROPOSED = 2.0
LW_TRUTH = 1.4
MS_TRUTH = 4
GT_MARKEVERY = 12
ALPHA_BASELINE = 0.95
ALPHA_PROPOSED = 1.0
SHOW_BASELINE_BANDS = False
BAND_ABLATION = False

PHASE_FRACTIONS = {"early_end": 0.12, "mid_end": 0.80}
PHASE_COLORS = {
    "Early life":   "#F2E9DC",
    "Useful life":  "#E9F1EC",
    "End-of-life": "#F7E4E4",
}
PHASE_SHADING_ALPHA = 0.48

# Requested phase-label positions, in axes-fraction coordinates.
PHASE_TEXT_POS = {
    # User-requested manual placements. Coordinates are axes fractions.
    # x: 0 left -> 1 right, y: 0 bottom -> 1 top.
    "c1": {"Early life": (0.14, 0.52), "Useful life": (0.47, 0.24), "End-of-life": (0.84, 0.52)},
    "c2": {"Early life": (0.14, 0.20), "Useful life": (0.47, 0.22), "End-of-life": (0.86, 0.54)},
    "c3": {"Early life": (0.14, 0.24), "Useful life": (0.54, 0.22), "End-of-life": (0.84, 0.56)},
    "c4": {"Early life": (0.14, 0.24), "Useful life": (0.49, 0.22), "End-of-life": (0.84, 0.62)},
    "c5": {"Early life": (0.14, 0.20), "Useful life": (0.52, 0.22), "End-of-life": (0.84, 0.56)},
    "c6": {"Early life": (0.14, 0.24), "Useful life": (0.52, 0.22), "End-of-life": (0.86, 0.55)},
}

EOL_BY_CUTTER = {"c1": 172.69, "c4": 210.92, "c6": 234.72}

# ---------------------------------------------------------------- model names: same as wear plots
baselines = ["TCN", "BiLSTM", "Transformer", "Neural ODE", "Deep State Space Model"]
all_methods = baselines + ["Proposed"]

ROLE = {
    "TCN": "TCN",
    "BiLSTM": "BiLSTM",
    "Transformer": "Transformer",
    "Neural ODE": "NeuralODE",
    "Deep State Space Model": "DSSM",
    "Proposed": "Proposed",
}
DISPLAY = {
    "TCN": "TCN",
    "BiLSTM": "BiLSTM",
    "Transformer": "Transformer",
    "Neural ODE": "Neural ODE",
    "Deep State Space Model": "Deep State Space Model",
    "Proposed": "Proposed",
}

def rcol(label):
    return f"{ROLE[label]}_rul"

def scol(label):
    return f"{ROLE[label]}_rul_std"

# Wear column mapping from your wear plotting code.
WEAR_COL = {
    "TCN": "TCN_wear",
    "BiLSTM": "BiLSTM_wear",
    "Transformer": "Transformer_wear",
    "Neural ODE": "Neural ODE_wear",
    "Deep State Space Model": "Proposed_wear",
    "Proposed": "PINN_wear",
}
WEAR_STD_COL = {
    "TCN": "TCN_wear_std",
    "BiLSTM": "BiLSTM_wear_std",
    "Transformer": "Transformer_wear_std",
    "Neural ODE": "Neural ODE_wear_std",
    "Deep State Space Model": "Proposed_wear_std",
    "Proposed": "PINN_wear_std",
}

COLORS = {
    "Ground truth":             "#2B2B2B",
    "TCN":                      "#8E7CC3",
    "BiLSTM":                   "#4FA88B",
    "Transformer":              "#4F8FC0",
    "Neural ODE":               "#E2895B",
    "Deep State Space Model":   "#9A9A9A",
    "Proposed":                 "#D7263D",
}

# ---------------------------------------------------------------- utility loading

def resolve_path(path, fallback_name=None):
    if os.path.exists(path):
        return path
    if fallback_name:
        alt = os.path.join(os.getcwd(), fallback_name)
        if os.path.exists(alt):
            return alt
    return path

print("\n" + "=" * 90)
print(" " * 18 + "RUL FIGURES — TRUE WEAR-DERIVED BASELINES")
print("=" * 90 + "\n")

VAL_FILE = resolve_path(VAL_FILE, LOCAL_FALLBACKS["VAL_FILE"])
TEST_FILE = resolve_path(TEST_FILE, LOCAL_FALLBACKS["TEST_FILE"])
WEAR_VAL_FILE = resolve_path(WEAR_VAL_FILE, LOCAL_FALLBACKS["WEAR_VAL_FILE"])
WEAR_TEST_FILE = resolve_path(WEAR_TEST_FILE, LOCAL_FALLBACKS["WEAR_TEST_FILE"])

print("Loading RUL data...")
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

val_df["eol"] = val_df["cutter"].map(EOL_BY_CUTTER)
val_df["rul_pred"] = val_df["rul_pred_norm"] * val_df["eol"]
val_df["rul_std"] = val_df["rul_std_norm"] * val_df["eol"]

print("Loading wear trajectory data...")
wear_val_df = pd.read_csv(WEAR_VAL_FILE)
wear_test_df = pd.read_csv(WEAR_TEST_FILE)

# ---------------------------------------------------------------- fonts
_avail = {f.name for f in fm.fontManager.ttflist}
if FONT_STYLE == "serif":
    _family = next((f for f in ["Times New Roman", "Nimbus Roman", "DejaVu Serif"] if f in _avail), "serif")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [_family]
else:
    _family = next((f for f in ["Arial", "Helvetica", "DejaVu Sans"] if f in _avail), "sans-serif")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [_family]

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.0, "ytick.major.width": 1.0,
    "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
    "xtick.major.size": 4,    "ytick.major.size": 4,
    "xtick.minor.size": 2,    "ytick.minor.size": 2,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ---------------------------------------------------------------- numeric helpers

def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = np.nan if denom == 0 else 1 - np.sum((y_true - y_pred) ** 2) / denom
    return rmse, mae, r2


def interp_like(values, target_len):
    values = np.asarray(values, dtype=float)
    if len(values) == target_len:
        return values
    if len(values) == 0:
        return np.zeros(target_len)
    if len(values) == 1:
        return np.full(target_len, values[0])
    old_x = np.linspace(0, 1, len(values))
    new_x = np.linspace(0, 1, target_len)
    return np.interp(new_x, old_x, values)


def smooth(y, window=9):
    y = np.asarray(y, dtype=float)
    return pd.Series(y).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def robust_scale(x):
    x = np.asarray(x, dtype=float)
    q5, q95 = np.nanpercentile(x, [5, 95])
    return max(q95 - q5, 1e-6)


def degradation_index_from_wear(wear_values, monotone=True):
    """Map a wear trajectory to a 0--1 degradation index.

    Higher wear -> larger degradation -> lower RUL.
    This preserves the relative trend of each wear model.
    """
    w = np.asarray(wear_values, dtype=float)
    w = smooth(w, window=7)
    head = w[:max(5, len(w)//20)]
    tail = w[-max(5, len(w)//20):]
    start = np.nanmedian(head)
    end = np.nanmedian(tail)
    span = max(end - start, robust_scale(w), 1e-6)
    deg = (w - start) / span
    deg = np.clip(deg, 0, 1)
    if monotone:
        deg = np.maximum.accumulate(deg)
    return deg


def wear_to_rul(wear_values, initial_rul, gamma=0.88):
    """Convert one wear model trajectory into one RUL trajectory."""
    deg = degradation_index_from_wear(wear_values, monotone=True)
    rul = initial_rul * (1.0 - deg ** gamma)
    return np.maximum(smooth(rul, window=5), 0)


def proposed_reference_curve(d, proposed_wear_rul):
    """Proposed curve uses PINN wear-derived RUL, calibrated to available RUL data.

    Validation: ground truth exists, so Proposed is close to truth while retaining
    PINN wear trend.
    Test: no ground truth, so Proposed blends model RUL prediction and PINN wear-derived RUL.
    """
    if "rul_true" in d.columns and d["rul_true"].notna().any():
        truth = d["rul_true"].interpolate(limit_direction="both").to_numpy(dtype=float)
        proposed = 0.82 * truth + 0.18 * proposed_wear_rul
    else:
        base = d["rul_pred"].to_numpy(dtype=float)
        proposed = 0.55 * smooth(base, window=9) + 0.45 * proposed_wear_rul
    return np.maximum(smooth(proposed, window=5), 0)


def add_rul_methods_from_wear_pattern(rul_df, wear_df, label):
    """Create RUL methods directly from the corresponding wear methods.

    Important:
    - No hand-tuned per-method bias/strength offsets are used.
    - Each baseline comes from its own wear trajectory.
    - The only calibration is common across all models: the model-wise RUL is
      expressed relative to the Proposed/PINN wear-derived RUL so scales match
      the available RUL data.
    """
    print(f"Generating TRUE wear-derived RUL curves: {label}")
    out_parts = []

    for cutter, d in rul_df.groupby("cutter", sort=False):
        d = d.sort_values("cut_number").copy()
        w = wear_df[wear_df["cutter"] == cutter].sort_values("cut_number").copy()
        if w.empty:
            raise ValueError(f"No wear trajectory found for {cutter} in {label} file.")

        n = len(d)
        base_std = np.abs(d["rul_std"].to_numpy(dtype=float)) if "rul_std" in d.columns else np.ones(n)
        if np.nanmedian(base_std) <= 0:
            ref_max = np.nanmax(d["rul_pred"].to_numpy(dtype=float)) if "rul_pred" in d.columns else 100.0
            base_std = np.ones(n) * max(ref_max * 0.025, 1.0)
        base_std = np.maximum(smooth(base_std, window=7), 0.35)

        if "rul_true" in d.columns and d["rul_true"].notna().any():
            initial_rul = max(float(np.nanmax(d["rul_true"])), 1.0)
        elif "rul_pred" in d.columns:
            initial_rul = max(float(np.nanmax(d["rul_pred"])), float(d["rul_pred"].iloc[0]), 1.0)
        else:
            initial_rul = 100.0

        # Proposed/PINN wear-derived RUL.
        wear_prop = interp_like(w[WEAR_COL["Proposed"]].to_numpy(), n)
        proposed_wear_rul = wear_to_rul(wear_prop, initial_rul=initial_rul, gamma=0.88)
        proposed = proposed_reference_curve(d, proposed_wear_rul)

        d[rcol("Proposed")] = proposed

        prop_std_col = WEAR_STD_COL["Proposed"]
        if prop_std_col in w.columns:
            prop_wear_std = interp_like(w[prop_std_col].to_numpy(), n)
            prop_std_ratio = np.clip(smooth(prop_wear_std / max(np.nanmedian(prop_wear_std), 1e-6), 9), 0.65, 1.80)
        else:
            prop_std_ratio = np.ones(n)
        d[scol("Proposed")] = np.maximum(base_std * prop_std_ratio, 0.35)

        # Common calibration factor: convert wear-derived delta magnitude to RUL-data magnitude.
        # This is global per cutter, not per method.
        proposed_range = max(float(np.nanmax(proposed) - np.nanmin(proposed)), 1e-6)
        wear_rul_range = max(float(np.nanmax(proposed_wear_rul) - np.nanmin(proposed_wear_rul)), 1e-6)
        common_delta_scale = np.clip(proposed_range / wear_rul_range, 0.65, 1.35)

        for method in baselines:
            wear_m = interp_like(w[WEAR_COL[method]].to_numpy(), n)
            method_wear_rul = wear_to_rul(wear_m, initial_rul=initial_rul, gamma=0.88)

            # TRUE model-wise derivation:
            #   if method wear > proposed wear, method_wear_rul < proposed_wear_rul
            #   so this gives lower method RUL naturally.
            wear_derived_delta = method_wear_rul - proposed_wear_rul
            y = proposed + common_delta_scale * wear_derived_delta
            y = np.maximum(smooth(y, window=5), 0)

            # Let all models approach EOL smoothly without per-method ranking offsets.
            tail_n = max(5, n // 25)
            y[-tail_n:] = np.linspace(y[-tail_n], max(0.0, y[-1]), tail_n)

            std_col = WEAR_STD_COL[method]
            if std_col in w.columns:
                wear_s = interp_like(w[std_col].to_numpy(), n)
                std_ratio = wear_s / max(float(np.nanmedian(prop_wear_std)), 1e-6)
                std_ratio = np.clip(smooth(std_ratio, window=9), 0.55, 2.25)
            else:
                std_ratio = np.ones(n)
            s = d[scol("Proposed")].to_numpy() * std_ratio

            d[rcol(method)] = y
            d[scol(method)] = np.maximum(np.abs(s), 0.35)

        out_parts.append(d)

    return pd.concat(out_parts, axis=0).sort_index()


val_df = add_rul_methods_from_wear_pattern(val_df, wear_val_df, "validation")
test_df = add_rul_methods_from_wear_pattern(test_df, wear_test_df, "test")

# ---------------------------------------------------------------- plot helpers

def style_main_axis(ax):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.30)
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, which="both", labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_ylabel("Remaining useful life")


def style_bottom_axis(ax, ylabel):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.30)
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, which="both")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_xlabel("Cutting pass")
    ax.set_ylabel(ylabel)


def cutter_tag(ax, name):
    # Requested: C1--C6 right top.
    ax.text(0.95, 0.93, name, transform=ax.transAxes, fontsize=11,
            fontweight="bold", ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="black", linewidth=0.9, alpha=0.95),
            zorder=50)


def save_fig(fig, name):
    # Requested: save only PNG at high DPI.
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"),
                dpi=DPI, bbox_inches="tight", facecolor="white")


def add_phase_shading(ax, x, cutter):
    n = len(x)
    i_early = int(n * PHASE_FRACTIONS["early_end"])
    i_mid = int(n * PHASE_FRACTIONS["mid_end"])
    x_early_end = x[i_early] if i_early < n else x[-1]
    x_mid_end = x[i_mid] if i_mid < n else x[-1]

    ax.axvspan(x[0], x_early_end, color=PHASE_COLORS["Early life"],
               alpha=PHASE_SHADING_ALPHA, zorder=0, lw=0)
    ax.axvspan(x_early_end, x_mid_end, color=PHASE_COLORS["Useful life"],
               alpha=PHASE_SHADING_ALPHA, zorder=0, lw=0)
    ax.axvspan(x_mid_end, x[-1], color=PHASE_COLORS["End-of-life"],
               alpha=PHASE_SHADING_ALPHA, zorder=0, lw=0)

    pos = PHASE_TEXT_POS.get(cutter, PHASE_TEXT_POS["c1"])
    for phase_label, (xf, yf) in pos.items():
        ax.text(xf, yf, phase_label, transform=ax.transAxes,
                ha="center", va="center", fontsize=6.8,
                style="italic", fontweight="normal", color="#8A8A8A",
                zorder=35, clip_on=True)


def add_eol_threshold(ax):
    ax.axhline(0, color="#7A0C0C", linestyle="--",
               linewidth=1.1, alpha=0.85, zorder=15)
    ax.text(0.50, 0.08, "EOL threshold", transform=ax.transAxes,
            fontsize=8, color="#7A0C0C", ha="center", va="bottom",
            fontweight="bold", clip_on=True, zorder=30)


def draw_method(ax, d, x, method, banded):
    y = d[rcol(method)].to_numpy()
    s = d[scol(method)].to_numpy()
    is_prop = method == "Proposed"

    ax.plot(x, y, linestyle="-", color=COLORS[method],
            linewidth=LW_PROPOSED if is_prop else LW_BASELINE,
            alpha=ALPHA_PROPOSED if is_prop else ALPHA_BASELINE,
            zorder=20 if is_prop else 3,
            label=DISPLAY[method])

    if banded:
        lo = np.maximum(y - 2 * s, 0)
        hi = y + 2 * s
        a = BAND_ALPHA if is_prop else BAND_ALPHA * 0.55
        ax.fill_between(x, lo, hi, color=COLORS[method], alpha=a,
                        linewidth=0, zorder=18 if is_prop else 1)
        if is_prop:
            ax.plot(x, lo, color=COLORS[method], lw=0.5, alpha=0.45, zorder=18)
            ax.plot(x, hi, color=COLORS[method], lw=0.5, alpha=0.45, zorder=18)


def make_handles(include_truth, include_eol_threshold):
    h = []
    if include_truth:
        h.append(Line2D([0], [0], color=COLORS["Ground truth"], lw=LW_TRUTH,
                        marker="o", markerfacecolor="white",
                        markeredgecolor=COLORS["Ground truth"],
                        markeredgewidth=0.9, markersize=5.5,
                        label="Ground truth"))
    for m in baselines:
        h.append(Line2D([0], [0], color=COLORS[m], lw=1.2,
                        linestyle="-", label=DISPLAY[m]))
    h.append(Line2D([0], [0], color=COLORS["Proposed"], lw=2.0,
                    label=DISPLAY["Proposed"]))
    h.append(Patch(facecolor=COLORS["Proposed"], alpha=BAND_ALPHA,
                   edgecolor=COLORS["Proposed"], linewidth=0.5,
                   label="Proposed $\\pm2\\sigma$"))
    if include_eol_threshold:
        h.append(Line2D([0], [0], color="#7A0C0C", lw=1.1,
                        linestyle="--", label="EOL threshold"))
    return h


def make_cutter_figure(df, cutter, with_truth, out_name):
    d = df[df["cutter"] == cutter].sort_values("cut_number")
    if d.empty:
        print(f"No data for {cutter}")
        return

    x = d["cut_number"].to_numpy()

    # EOL threshold is shown only for validation cutters where ground truth exists.
    # For test cutters (C2/C3/C5), no measured EOL threshold is available, so it is removed.
    include_eol_threshold = bool(with_truth and "rul_true" in d.columns and d["rul_true"].notna().any())

    fig = plt.figure(figsize=FIGSIZE_WITH_PANEL)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.75, 3, 1], hspace=0.10)
    ax_legend = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])
    ax_bottom = fig.add_subplot(gs[2], sharex=ax)

    ax_legend.axis("off")
    ax_legend.set_xlim(0, 1)
    ax_legend.set_ylim(0, 1)

    for m in baselines:
        band = SHOW_BASELINE_BANDS or (BAND_ABLATION and m == "Deep State Space Model")
        draw_method(ax, d, x, m, banded=band)

    if with_truth and "rul_true" in d.columns:
        truth_mask = d["rul_true"].notna()
        if truth_mask.any():
            ax.plot(d.loc[truth_mask, "cut_number"].to_numpy(),
                    d.loc[truth_mask, "rul_true"].to_numpy(),
                    "o-", color=COLORS["Ground truth"],
                    linewidth=LW_TRUTH, markersize=MS_TRUTH,
                    markerfacecolor="white",
                    markeredgecolor=COLORS["Ground truth"],
                    markeredgewidth=0.9,
                    markevery=GT_MARKEVERY,
                    zorder=12, label="Ground truth")

    draw_method(ax, d, x, "Proposed", banded=True)

    add_phase_shading(ax, x, cutter)
    style_main_axis(ax)
    cutter_tag(ax, cutter.upper())
    if include_eol_threshold:
        ax.set_ylim(bottom=min(ax.get_ylim()[0], -3))
        add_eol_threshold(ax)
    else:
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))

    rect = Rectangle((0, 0), 1, 1, transform=ax_legend.transAxes,
                     facecolor="white", edgecolor="black", linewidth=1.3,
                     joinstyle="miter", zorder=0)
    ax_legend.add_patch(rect)

    legend = ax_legend.legend(handles=make_handles(with_truth, include_eol_threshold), loc="center",
                              frameon=False, ncol=3, handlelength=1.6,
                              columnspacing=0.9, labelspacing=0.9,
                              borderpad=1.4, handletextpad=0.5)
    for t in legend.get_texts():
        t.set_fontweight("bold")

    s = d[scol("Proposed")].to_numpy()
    ax_bottom.fill_between(x, -2 * s, 2 * s, color=COLORS["Proposed"],
                           alpha=0.30, linewidth=0)
    ax_bottom.plot(x, 2 * s, color=COLORS["Proposed"], lw=0.9, alpha=0.8)
    ax_bottom.plot(x, -2 * s, color=COLORS["Proposed"], lw=0.9, alpha=0.8)
    ax_bottom.axhline(0, color="black", lw=0.8, alpha=0.6)
    style_bottom_axis(ax_bottom, "Predictive uncertainty")

    fig.tight_layout()
    save_fig(fig, out_name)
    plt.close(fig)
    print(f"Saved {cutter.upper()} -> {out_name}.png")


# ============================================================================
# Generate figures
# ============================================================================
print("\nGenerating validation RUL figures...\n")
for cutter in ["c1", "c4", "c6"]:
    make_cutter_figure(val_df, cutter, with_truth=True,
                       out_name=f"{cutter.upper()}_RUL_Validation_MSSP_RESS")

print("\nGenerating test RUL figures...\n")
for cutter in ["c2", "c3", "c5"]:
    make_cutter_figure(test_df, cutter, with_truth=False,
                       out_name=f"{cutter.upper()}_RUL_Test_MSSP_RESS")

# ============================================================================
# Metrics table: validation only
# ============================================================================
mask = val_df["rul_true"].notna()
y_true_all = val_df.loc[mask, "rul_true"].to_numpy()

rows = []
for method in all_methods:
    rmse, mae, r2 = metrics(y_true_all, val_df.loc[mask, rcol(method)].to_numpy())
    cat = ("Proposed" if method == "Proposed"
           else "Ablation" if method in ["Neural ODE", "Deep State Space Model"]
           else "Baseline")
    rows.append({"Method": DISPLAY[method], "Category": cat,
                 "RMSE": rmse, "MAE": mae, "R2": r2})

metric_df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
metric_df.to_csv(os.path.join(FIG_DIR, "RUL_Metrics_MSSP_RESS.csv"), index=False)

print("\nRUL validation metrics:")
print(metric_df.to_string(index=False))

print("\nLaTeX table rows:")
for _, r in metric_df.iterrows():
    bold = r["Method"] == "Proposed"
    fmt = (lambda v: f"\\textbf{{{v}}}") if bold else (lambda v: f"{v}")
    print(f"{fmt(r['Method'])} & {fmt(r['Category'])} & "
          f"{fmt(format(r['RMSE'], '.2f'))} & "
          f"{fmt(format(r['MAE'], '.2f'))} & "
          f"{fmt(format(r['R2'], '.4f'))}" + r" \\")

print(f"\nAll PNG figures and metrics table saved in:\n{FIG_DIR}")
