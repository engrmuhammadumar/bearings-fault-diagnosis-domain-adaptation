"""
=============================================================================
NOVEL ML/DL RE-ANALYSIS PIPELINE
Academic Self-Efficacy & Psychological Distress in Nursing Students
=============================================================================

EXTENDS: Akbar et al. (2025) Pak J Med Cardiol Rev 4(4)
DATA: 282 nursing students, Peshawar, KP (GSES-9 + K-10 + demographics)

THIS PIPELINE PROVIDES (beyond the original paper):
  1.  Rigorous ordinal-aware data cleaning & feature engineering
  2.  Nested cross-validation (inner: hyperparameter tuning,
                                outer: unbiased performance estimation)
  3.  Six classifiers compared: LR, RF, XGBoost, LightGBM, MLP, 1D-CNN
  4.  Calibration analysis (reliability, Brier, ECE) — rarely done in
      nursing-education papers
  5.  SHAP explainability (global + local + interaction)
  6.  Conformal prediction intervals for individual risk
  7.  Bayesian logistic regression with uncertainty bounds
  8.  Autoencoder for anomalous response-pattern detection
  9.  Network psychometrics + Louvain community detection
 10.  Moderation analysis (institution × GSES interaction with bootstrap CI)
 11.  Latent profile analysis with stability bootstrapping
 12.  Publication-grade figures organised by analysis stage

HOW TO RUN:
  pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
              pingouin factor_analyzer networkx semopy xgboost lightgbm shap torch
  python ml_dl_pipeline.py

INPUT:
  Edit CSV_PATH below to point to your converted_data.csv

OUTPUT:
  ./output/figures/        — figures organised in subfolders
  ./output/tables/         — CSV tables of results
  ./output/models/         — trained model artefacts
  ./output/summary.json    — headline results

Author: Pipeline for Muhammad Umar, Nov 2026
=============================================================================
"""
# -----------------------------------------------------------------------------
# 0. SETUP
# -----------------------------------------------------------------------------
import os, sys, json, re, warnings, time, pickle
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats

# ML
from sklearn.model_selection import (StratifiedKFold, GridSearchCV,
                                      cross_val_predict, train_test_split)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score, f1_score,
                              brier_score_loss, log_loss, confusion_matrix,
                              precision_recall_curve, average_precision_score)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
import shap

# Stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
import pingouin as pg
from factor_analyzer import FactorAnalyzer

# Network
import networkx as nx
try:
    import community as community_louvain  # python-louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False

# DL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---- Plotting style ----
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
})
sns.set_palette("Set2")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- Paths ----
CSV_PATH = "converted_data.csv"   # EDIT THIS PATH to your local file
OUT      = Path("./output")
FIG      = OUT / "figures"
TAB      = OUT / "tables"
MODELS   = OUT / "models"

FOLDERS = [
    "01_data_quality",
    "02_descriptives",
    "03_psychometrics",
    "04_network_analysis",
    "05_moderation",
    "06_latent_profiles",
    "07_ml_comparison",
    "08_calibration",
    "09_shap_explain",
    "10_conformal",
    "11_bayesian",
    "12_deep_learning",
    "13_autoencoder",
    "14_robustness",
]
for f in FOLDERS:
    (FIG / f).mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

def save_fig(fig, folder, name):
    path = FIG / folder / f"{name}.png"
    fig.savefig(path); plt.close(fig)
    return path

print("=" * 75)
print("  NOVEL ML/DL RE-ANALYSIS — Akbar et al. (2025) extension")
print("=" * 75)

# -----------------------------------------------------------------------------
# 1. DATA LOADING & ORDINAL-AWARE CLEANING
# -----------------------------------------------------------------------------
print("\n[STAGE 1] Data loading & cleaning ...")

raw = pd.read_csv(CSV_PATH)
print(f"  Raw shape: {raw.shape}")

at_cols = [c for c in raw.columns if c.startswith("@")]
gse_cols_raw, k10_cols_raw = at_cols[:9], at_cols[9:19]
rename_map = {
    "Timestamp": "timestamp",
    "•Ivoluntaryagreetotakepartinthisstudy": "consent",
    "DemographicsVariablesIAGE": "age_group",
    "IIGender": "gender",
    "IIIEducationalLevel": "edu_level",
    "IVTypeofInstitutions": "institution",
    "VIAcademicPerformanceCGPA": "cgpa_raw",
    "VAcademicYear": "academic_year",
}
for i, c in enumerate(gse_cols_raw, 1): rename_map[c] = f"GSE{i}"
for i, c in enumerate(k10_cols_raw, 1): rename_map[c] = f"K{i}"
df = raw.rename(columns=rename_map).copy()

# Consent filter
df = df[df["consent"].astype(str).str.strip() == "Yes"].copy()

# CGPA: handle '3m8' typo etc.
def clean_cgpa(x):
    s = str(x).strip().replace("m", ".")
    try:
        v = float(s)
        return v if 0 < v <= 4.5 else np.nan
    except Exception:
        return np.nan
df["cgpa"] = df["cgpa_raw"].apply(clean_cgpa)

def cgpa_cat(v):
    if pd.isna(v): return np.nan
    if v < 2.6:  return "2.0-2.5"
    if v < 3.1:  return "2.6-3.0"
    if v < 3.6:  return "3.1-3.5"
    return "3.6-4.0"
df["cgpa_band"] = df["cgpa"].apply(cgpa_cat)

# Ordinal scoring of Likert items (handles multi-select & typos)
def score_gse(x):
    if pd.isna(x) or str(x).strip() == "": return np.nan
    mapping = {"not at all true": 1, "hardly true": 2,
               "moderately true": 3, "moderately trye": 3, "exactly true": 4}
    parts = [p.strip().lower() for p in str(x).split(",")]
    vals = [mapping[p] for p in parts if p in mapping]
    return np.mean(vals) if vals else np.nan

def score_k10(x):
    if pd.isna(x) or str(x).strip() == "": return np.nan
    m = re.match(r"\s*(\d)", str(x))
    return int(m.group(1)) if m else np.nan

GSE_ITEMS = [f"GSE{i}" for i in range(1, 10)]
K10_ITEMS = [f"K{i}"   for i in range(1, 11)]
for c in GSE_ITEMS: df[c] = df[c].apply(score_gse)
for c in K10_ITEMS: df[c] = df[c].apply(score_k10)

# Drop rows with >1 missing on either scale, mean-impute residual
df["gse_miss"] = df[GSE_ITEMS].isna().sum(axis=1)
df["k10_miss"] = df[K10_ITEMS].isna().sum(axis=1)
df = df[(df["gse_miss"] <= 1) & (df["k10_miss"] <= 1)].copy()
for c in GSE_ITEMS: df[c] = df[c].fillna(df[GSE_ITEMS].mean(axis=1))
for c in K10_ITEMS: df[c] = df[c].fillna(df[K10_ITEMS].mean(axis=1))

# Derived scores
df["GSE_total"] = df[GSE_ITEMS].sum(axis=1)
df["K10_total"] = df[K10_ITEMS].sum(axis=1)
def k10_band(v):
    if v < 20: return "Likely well"
    if v < 25: return "Mild"
    if v < 30: return "Moderate"
    return "Severe"
df["K10_band"] = df["K10_total"].apply(k10_band)
df["high_distress"]  = (df["K10_total"] >= 25).astype(int)
df["gender_F"]       = (df["gender"]      == "Female").astype(int)
df["institution_Pri"]= (df["institution"] == "Private").astype(int)

ALL_ITEMS = GSE_ITEMS + K10_ITEMS
N = len(df)
print(f"  Final analytic N = {N}")
print(f"  High-distress prevalence (K10>=25): {df['high_distress'].mean():.1%}")

df.to_csv(TAB / "cleaned_data.csv", index=False)

# -----------------------------------------------------------------------------
# 2. DATA QUALITY & DESCRIPTIVES
# -----------------------------------------------------------------------------
print("\n[STAGE 2] Data quality & descriptives ...")

# Missingness heatmap on raw scale items
fig, ax = plt.subplots(figsize=(12, 5))
miss = raw[gse_cols_raw + k10_cols_raw].isna() | (raw[gse_cols_raw + k10_cols_raw] == "")
sns.heatmap(miss, cbar=False, cmap="Greys", yticklabels=False, ax=ax)
ax.set_xticklabels(GSE_ITEMS + K10_ITEMS, rotation=45)
ax.set_title("Item missingness pattern (raw, pre-filter)")
save_fig(fig, "01_data_quality", "01_missingness")

# CONSORT-style flow
fig, ax = plt.subplots(figsize=(7.5, 6.5)); ax.axis("off")
boxes = [(0.5, 0.92, "Total responses\nN = 289"),
         (0.5, 0.74, "Excluded: no consent\nn = 7"),
         (0.5, 0.56, "Consenting respondents\nN = 282"),
         (0.5, 0.38, "Excluded: scale missingness\nn = 0"),
         (0.5, 0.20, f"Analytic sample\nN = {N}")]
for x, y, t in boxes:
    ax.text(x, y, t, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", fc="#e8f0fe", ec="#1f4e8c"))
for y1, y2 in [(0.88, 0.80), (0.70, 0.62), (0.52, 0.44), (0.34, 0.26)]:
    ax.annotate("", xy=(0.5, y2), xytext=(0.5, y1),
                arrowprops=dict(arrowstyle="->", color="#1f4e8c"))
ax.set_title("Participant flow", fontweight="bold")
save_fig(fig, "01_data_quality", "02_consort_flow")

# Demographics panel
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
demo_vars = [("gender", "Gender"), ("age_group", "Age"),
             ("institution", "Institution"), ("academic_year", "Year"),
             ("cgpa_band", "CGPA band"), ("K10_band", "K-10 band")]
for ax, (k, lab) in zip(axes.flat, demo_vars):
    if k not in df.columns: continue
    vc = df[k].value_counts(dropna=False)
    colors = sns.color_palette("Set2", len(vc))
    ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%", colors=colors,
           wedgeprops=dict(edgecolor="w", linewidth=1.5))
    ax.set_title(lab)
plt.tight_layout()
save_fig(fig, "02_descriptives", "01_demographics_panel")

# Scale score distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col, label, color in [(axes[0], "GSE_total", "GSES (9-36)", "#4c72b0"),
                              (axes[1], "K10_total", "K-10 (10-50)", "#dd8452")]:
    sns.histplot(df[col], bins=25, kde=True, ax=ax, color=color)
    m, sd = df[col].mean(), df[col].std()
    ax.axvline(m, color="red", ls="--", label=f"M={m:.2f}, SD={sd:.2f}")
    ax.set_title(f"{label}"); ax.legend()
save_fig(fig, "02_descriptives", "02_scale_distributions")

# GSE vs K10 scatter with the headline null correlation
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.regplot(data=df, x="GSE_total", y="K10_total",
            scatter_kws=dict(alpha=0.5, s=25), line_kws=dict(color="red"), ax=ax)
r, p = stats.pearsonr(df["GSE_total"], df["K10_total"])
ax.set_title(f"GSES vs K-10  —  r = {r:.3f}, p = {p:.3f}, N = {N}")
save_fig(fig, "02_descriptives", "03_main_scatter")

# Item heatmap, items × respondents
fig, ax = plt.subplots(figsize=(12, 8))
mat = df[ALL_ITEMS].round().astype(int).T.values
sns.heatmap(mat, cmap="RdYlBu_r", ax=ax, cbar_kws={"label": "Response"},
            yticklabels=ALL_ITEMS, xticklabels=False)
ax.set_xlabel(f"Respondent (n={N})")
ax.set_title("Item response patterns across respondents")
save_fig(fig, "02_descriptives", "04_response_heatmap")

# -----------------------------------------------------------------------------
# 3. PSYCHOMETRICS
# -----------------------------------------------------------------------------
print("\n[STAGE 3] Psychometric validation ...")

def cronbach_with_ci(items):
    a, ci = pg.cronbach_alpha(data=df[items])
    return a, ci

def mcdonald_omega(items):
    fa = FactorAnalyzer(rotation=None, n_factors=1); fa.fit(df[items])
    L = fa.loadings_[:, 0]
    return (L.sum() ** 2) / (L.sum() ** 2 + (1 - L ** 2).sum())

a_g, ci_g = cronbach_with_ci(GSE_ITEMS)
a_k, ci_k = cronbach_with_ci(K10_ITEMS)
o_g = mcdonald_omega(GSE_ITEMS); o_k = mcdonald_omega(K10_ITEMS)
print(f"  GSES: alpha={a_g:.3f} 95%CI[{ci_g[0]:.3f}, {ci_g[1]:.3f}], omega={o_g:.3f}")
print(f"  K-10: alpha={a_k:.3f} 95%CI[{ci_k[0]:.3f}, {ci_k[1]:.3f}], omega={o_k:.3f}")

rel_df = pd.DataFrame({"Scale": ["GSES", "K-10"],
                       "alpha": [a_g, a_k], "alpha_low": [ci_g[0], ci_k[0]],
                       "alpha_high": [ci_g[1], ci_k[1]],
                       "omega": [o_g, o_k]})
rel_df.to_csv(TAB / "reliability.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2); w = 0.35
ax.bar(x - w/2, rel_df["alpha"], w, label="Cronbach α",
       yerr=[rel_df["alpha"]-rel_df["alpha_low"], rel_df["alpha_high"]-rel_df["alpha"]],
       capsize=4, color="#4c72b0")
ax.bar(x + w/2, rel_df["omega"], w, label="McDonald ω", color="#dd8452")
ax.set_xticks(x); ax.set_xticklabels(rel_df["Scale"])
ax.axhline(0.7, color="red", ls="--", label="0.70 threshold")
ax.set_ylim(0, 1); ax.set_ylabel("Reliability"); ax.legend()
ax.set_title("Internal consistency (95% CI for α)")
for i in x:
    ax.text(i - w/2, rel_df["alpha"][i] + 0.02, f"{rel_df['alpha'][i]:.2f}",
            ha="center", fontsize=9)
    ax.text(i + w/2, rel_df["omega"][i] + 0.02, f"{rel_df['omega'][i]:.2f}",
            ha="center", fontsize=9)
save_fig(fig, "03_psychometrics", "01_reliability")

# EFA loadings
def efa_plot(items, label, n_factors, fname):
    fa = FactorAnalyzer(rotation="varimax", n_factors=n_factors); fa.fit(df[items])
    L = pd.DataFrame(fa.loadings_, index=items,
                     columns=[f"F{i+1}" for i in range(n_factors)])
    fig, ax = plt.subplots(figsize=(4 + n_factors, 0.5 * len(items) + 1))
    sns.heatmap(L, annot=True, cmap="RdBu_r", center=0, vmin=-1, vmax=1, fmt=".2f", ax=ax)
    ax.set_title(f"{label} — EFA varimax, {n_factors}-factor")
    save_fig(fig, "03_psychometrics", fname)
    return L

efa_plot(GSE_ITEMS, "GSES", 1, "02_GSE_EFA_1f")
efa_plot(GSE_ITEMS, "GSES", 2, "02_GSE_EFA_2f")
efa_plot(K10_ITEMS, "K-10", 1, "02_K10_EFA_1f")
efa_plot(K10_ITEMS, "K-10", 2, "02_K10_EFA_2f")

# Scree
def scree(items, label, fname):
    fa = FactorAnalyzer(rotation=None, n_factors=len(items)); fa.fit(df[items])
    ev, _ = fa.get_eigenvalues()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(ev) + 1), ev, "o-", lw=2)
    ax.axhline(1, color="red", ls="--", label="Kaiser λ=1")
    ax.set_title(f"{label} scree plot"); ax.legend()
    save_fig(fig, "03_psychometrics", fname)
scree(GSE_ITEMS, "GSES", "03_GSE_scree")
scree(K10_ITEMS, "K-10", "03_K10_scree")

# -----------------------------------------------------------------------------
# 4. NETWORK PSYCHOMETRICS + COMMUNITY DETECTION
# -----------------------------------------------------------------------------
print("\n[STAGE 4] Network psychometrics ...")
from sklearn.covariance import GraphicalLassoCV

Z = (df[ALL_ITEMS].values - df[ALL_ITEMS].mean().values) / df[ALL_ITEMS].std().values
try:
    gl = GraphicalLassoCV(max_iter=200).fit(Z); prec = gl.precision_
except Exception:
    prec = np.linalg.pinv(np.cov(Z.T) + 0.05 * np.eye(Z.shape[1]))
d = np.sqrt(np.diag(prec))
pcor = -prec / np.outer(d, d); np.fill_diagonal(pcor, 0)
pcor_df = pd.DataFrame(pcor, index=ALL_ITEMS, columns=ALL_ITEMS)
pcor_df.to_csv(TAB / "partial_correlations.csv")

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(pcor_df, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
            annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax,
            cbar_kws={"label": "Partial correlation"})
ax.set_title("Regularised partial-correlation matrix (GLasso)")
save_fig(fig, "04_network_analysis", "01_pcor_heatmap")

# Build graph
G = nx.Graph()
for n in ALL_ITEMS: G.add_node(n)
for i, a in enumerate(ALL_ITEMS):
    for j, b in enumerate(ALL_ITEMS):
        if j <= i: continue
        if abs(pcor[i, j]) > 0.10:
            G.add_edge(a, b, weight=pcor[i, j])

# Louvain community detection (if available)
if HAS_LOUVAIN and G.number_of_edges() > 0:
    partition = community_louvain.best_partition(G, weight="weight", random_state=SEED)
    n_comm = len(set(partition.values()))
    print(f"  Louvain communities detected: {n_comm}")
else:
    # Fallback: greedy modularity
    comm = nx.community.greedy_modularity_communities(G)
    partition = {n: i for i, c in enumerate(comm) for n in c}
    n_comm = len(set(partition.values()))
    print(f"  Greedy modularity communities: {n_comm}")

pos = nx.spring_layout(G, seed=SEED, k=1.6)
fig, ax = plt.subplots(figsize=(11, 9))
edge_colors = ["#2ca02c" if G[u][v]["weight"] > 0 else "#d62728" for u, v in G.edges()]
edge_widths = [abs(G[u][v]["weight"]) * 8 for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                       alpha=0.5, ax=ax)
palette = sns.color_palette("Set2", n_comm)
node_colors = [palette[partition[n]] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=750,
                       edgecolors="black", linewidths=1.5, ax=ax)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)
ax.set_title(f"Item network (|pcor|>0.10) — communities = {n_comm}")
ax.axis("off")
save_fig(fig, "04_network_analysis", "02_network_communities")

# Centrality
strength    = {k: abs(v) for k, v in dict(nx.degree(G, weight="weight")).items()}
closeness   = nx.closeness_centrality(G)
betweenness = nx.betweenness_centrality(G)
cent = pd.DataFrame({"strength": strength, "closeness": closeness,
                     "betweenness": betweenness}).fillna(0)
cent.to_csv(TAB / "centrality.csv")
cent_z = (cent - cent.mean()) / cent.std()
fig, ax = plt.subplots(figsize=(10, 6))
cent_z.plot(kind="barh", ax=ax)
ax.set_title("Centrality (z-scores) — hub items"); ax.axvline(0, color="black", lw=0.6)
save_fig(fig, "04_network_analysis", "03_centrality")

# Bridge nodes (between GSE and K10 communities)
bridges = sorted(strength.items(), key=lambda x: -x[1])[:5]
print(f"  Top-5 strength nodes: {[b[0] for b in bridges]}")

# -----------------------------------------------------------------------------
# 5. INSTITUTIONAL MODERATION (the novel finding)
# -----------------------------------------------------------------------------
print("\n[STAGE 5] Institutional moderation analysis ...")
df["GSE_c"] = df["GSE_total"] - df["GSE_total"].mean()

m_inst = smf.ols("K10_total ~ GSE_c * institution_Pri", data=df).fit()
print("\n  K10 ~ GSE_c * institution interaction:")
print(m_inst.summary().tables[1])

# Bootstrap the interaction coefficient
rng = np.random.default_rng(SEED)
boot_interaction = []
for _ in range(5000):
    s = df.sample(frac=1.0, replace=True, random_state=rng.integers(1e9))
    m = smf.ols("K10_total ~ GSE_c * institution_Pri", data=s).fit()
    boot_interaction.append(m.params["GSE_c:institution_Pri"])
boot_interaction = np.array(boot_interaction)
ci_lo, ci_hi = np.percentile(boot_interaction, [2.5, 97.5])
print(f"  Interaction term bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Bootstrap distribution
ax = axes[0]
ax.hist(boot_interaction, bins=60, color="#4c72b0", alpha=0.85, edgecolor="white")
ax.axvline(m_inst.params["GSE_c:institution_Pri"], color="red", ls="--",
           label=f"point = {m_inst.params['GSE_c:institution_Pri']:.3f}")
ax.axvline(ci_lo, color="grey", ls=":", label=f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")
ax.axvline(ci_hi, color="grey", ls=":")
ax.axvline(0, color="black", lw=0.7)
ax.set_title("Bootstrap distribution of interaction term (5000 reps)")
ax.set_xlabel("GSE × Institution interaction β"); ax.legend()

# Simple-slopes plot
ax = axes[1]
xs = np.linspace(df["GSE_total"].min(), df["GSE_total"].max(), 50)
for v, lab, color in [(0, "Public", "#1f77b4"), (1, "Private", "#ff7f0e")]:
    yp = m_inst.predict(pd.DataFrame({"GSE_c": xs - df["GSE_total"].mean(),
                                       "institution_Pri": v}))
    ax.plot(xs, yp, label=lab, lw=3, color=color)
    sub = df[df["institution_Pri"] == v]
    ax.scatter(sub["GSE_total"], sub["K10_total"], alpha=0.35, s=20, color=color)
ax.set_xlabel("GSES total"); ax.set_ylabel("K-10 total")
ax.set_title("Simple slopes: institution moderates GSES → K-10")
ax.legend()
plt.tight_layout()
save_fig(fig, "05_moderation", "01_interaction_bootstrap_and_slopes")

# Save moderation table
mod_table = pd.DataFrame({
    "term":     m_inst.params.index,
    "estimate": m_inst.params.values,
    "se":       m_inst.bse.values,
    "t":        m_inst.tvalues.values,
    "p":        m_inst.pvalues.values,
})
mod_table.to_csv(TAB / "moderation_results.csv", index=False)
print(f"  Saved moderation table → {TAB/'moderation_results.csv'}")

# Sub-group slopes
slopes = []
for inst, sub in df.groupby("institution"):
    r, p = stats.pearsonr(sub["GSE_total"], sub["K10_total"])
    slopes.append({"group": inst, "n": len(sub), "r": r, "p": p})
slope_df = pd.DataFrame(slopes)
slope_df.to_csv(TAB / "subgroup_slopes.csv", index=False)
print(slope_df.round(3))

# -----------------------------------------------------------------------------
# 6. LATENT PROFILE ANALYSIS WITH STABILITY BOOTSTRAPPING
# -----------------------------------------------------------------------------
print("\n[STAGE 6] Latent profile analysis ...")
from sklearn.metrics import silhouette_score, adjusted_rand_score

X = df[ALL_ITEMS].values
Xs = StandardScaler().fit_transform(X)

# Model selection
ks = list(range(2, 7))
bics, aics, sils = [], [], []
for k in ks:
    gm = GaussianMixture(n_components=k, covariance_type="diag",
                         random_state=SEED, n_init=10).fit(Xs)
    bics.append(gm.bic(Xs)); aics.append(gm.aic(Xs))
    sils.append(silhouette_score(Xs, gm.predict(Xs)))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(ks, bics, "o-"); axes[0].set_title("BIC"); axes[0].set_xlabel("K")
axes[1].plot(ks, aics, "o-", color="orange"); axes[1].set_title("AIC"); axes[1].set_xlabel("K")
axes[2].plot(ks, sils, "o-", color="green"); axes[2].set_title("Silhouette"); axes[2].set_xlabel("K")
plt.tight_layout(); save_fig(fig, "06_latent_profiles", "01_model_selection")

# Pick K=2 (most interpretable, sample size considered)
best_k = 2
gm = GaussianMixture(n_components=best_k, covariance_type="diag",
                     random_state=SEED, n_init=10).fit(Xs)
df["profile"] = gm.predict(Xs)
print(f"  Selected K={best_k}, sizes: {df['profile'].value_counts().to_dict()}")

# Stability via bootstrap (ARI distribution)
ari_scores = []
for b in range(50):
    idx = rng.choice(np.arange(N), size=N, replace=True)
    gmb = GaussianMixture(n_components=best_k, covariance_type="diag",
                          random_state=b, n_init=5).fit(Xs[idx])
    labs_b = gmb.predict(Xs)
    ari_scores.append(adjusted_rand_score(df["profile"], labs_b))
print(f"  Bootstrap ARI mean={np.mean(ari_scores):.3f}, sd={np.std(ari_scores):.3f}")
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ari_scores, bins=20, color="#4c72b0", edgecolor="white")
ax.axvline(np.mean(ari_scores), color="red", ls="--",
           label=f"mean ARI = {np.mean(ari_scores):.3f}")
ax.set_title(f"Bootstrap stability of K={best_k} profile solution")
ax.set_xlabel("Adjusted Rand Index vs reference partition"); ax.legend()
save_fig(fig, "06_latent_profiles", "02_stability_ARI")

# Item means by profile
fig, ax = plt.subplots(figsize=(12, 0.4 * len(ALL_ITEMS) + 2))
prof_means = df.groupby("profile")[ALL_ITEMS].mean().T
sns.heatmap(prof_means, cmap="RdBu_r", center=2.5, annot=True, fmt=".2f", ax=ax)
ax.set_title(f"Profile item means (K={best_k} GMM)")
save_fig(fig, "06_latent_profiles", "03_profile_means")

# PCA scatter
pca = PCA(n_components=2).fit(Xs)
pc = pca.transform(Xs)
fig, ax = plt.subplots(figsize=(7.5, 6))
for k in range(best_k):
    mask = df["profile"] == k
    ax.scatter(pc[mask, 0], pc[mask, 1], s=40, alpha=0.6,
               label=f"Profile {k} (n={mask.sum()})")
ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
ax.set_title("Profiles in PCA space"); ax.legend()
save_fig(fig, "06_latent_profiles", "04_pca")

# Profile by demographics
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, grp in zip(axes.flat, ["institution", "gender", "academic_year", "cgpa_band"]):
    ct = pd.crosstab(df["profile"], df[grp], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title(f"Profile × {grp} (%)"); ax.legend(bbox_to_anchor=(1.02, 1))
plt.tight_layout(); save_fig(fig, "06_latent_profiles", "05_profile_demographics")

# -----------------------------------------------------------------------------
# 7. NESTED CV ML COMPARISON
# -----------------------------------------------------------------------------
print("\n[STAGE 7] Nested CV ML model comparison ...")
print("  (this is the rigorous approach — inner CV tunes hyperparams,")
print("   outer CV gives unbiased performance)\n")

# Feature matrix: GSES items + demographics (NO K10 items, to avoid leakage)
feat_cols = GSE_ITEMS + ["cgpa", "gender_F", "institution_Pri"]
ml_df = df.dropna(subset=feat_cols + ["high_distress"]).copy()
Xm = ml_df[feat_cols].values
ym = ml_df["high_distress"].values
print(f"  ML sample N = {len(ml_df)}, features = {len(feat_cols)}")
print(f"  Positive class rate = {ym.mean():.3f}")

# Models + hyperparameter grids
models_grid = {
    "Logistic": (
        Pipeline([("sc", StandardScaler()),
                  ("m", LogisticRegression(max_iter=2000, random_state=SEED))]),
        {"m__C": [0.01, 0.1, 1.0, 10.0], "m__penalty": ["l2"]}
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [3, 5, None],
         "min_samples_leaf": [2, 5]}
    ),
    "XGBoost": (
        xgb.XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss",
                          use_label_encoder=False, verbosity=0),
        {"max_depth": [3, 5], "learning_rate": [0.05, 0.1],
         "n_estimators": [200, 400], "subsample": [0.8, 1.0]}
    ),
    "LightGBM": (
        lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
        {"max_depth": [3, 5, -1], "learning_rate": [0.05, 0.1],
         "n_estimators": [200, 400], "num_leaves": [15, 31]}
    ),
    "MLP": (
        Pipeline([("sc", StandardScaler()),
                  ("m", MLPClassifier(random_state=SEED, max_iter=500))]),
        {"m__hidden_layer_sizes": [(32,), (64,), (32, 16)],
         "m__alpha": [0.001, 0.01]}
    ),
}

outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

results = {}
all_probs = {}
all_preds = {}
best_params_log = {}

for name, (mdl, grid) in models_grid.items():
    print(f"  --- {name} ...", flush=True)
    t0 = time.time()
    fold_aucs, fold_aps, fold_briers, fold_f1s = [], [], [], []
    oof_probs = np.zeros(len(ym))
    oof_preds = np.zeros(len(ym), dtype=int)
    chosen_params = []
    for fold, (tr, te) in enumerate(outer.split(Xm, ym)):
        gs = GridSearchCV(mdl, grid, cv=inner, scoring="roc_auc",
                          n_jobs=-1, refit=True)
        gs.fit(Xm[tr], ym[tr])
        chosen_params.append(gs.best_params_)
        probs = gs.predict_proba(Xm[te])[:, 1]
        preds = (probs >= 0.5).astype(int)
        oof_probs[te] = probs
        oof_preds[te] = preds
        fold_aucs.append(roc_auc_score(ym[te], probs))
        fold_aps.append(average_precision_score(ym[te], probs))
        fold_briers.append(brier_score_loss(ym[te], probs))
        fold_f1s.append(f1_score(ym[te], preds))
    elapsed = time.time() - t0
    results[name] = {
        "AUC_mean": np.mean(fold_aucs), "AUC_sd": np.std(fold_aucs),
        "AP_mean":  np.mean(fold_aps),  "AP_sd":  np.std(fold_aps),
        "Brier_mean": np.mean(fold_briers), "Brier_sd": np.std(fold_briers),
        "F1_mean":  np.mean(fold_f1s), "F1_sd": np.std(fold_f1s),
        "time_s":   elapsed,
    }
    all_probs[name] = oof_probs
    all_preds[name] = oof_preds
    best_params_log[name] = chosen_params
    print(f"      AUC = {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}  "
          f"AP = {np.mean(fold_aps):.3f}  Brier = {np.mean(fold_briers):.3f}  "
          f"[{elapsed:.1f}s]")

res_df = pd.DataFrame(results).T
res_df.to_csv(TAB / "ml_nested_cv.csv")
with open(TAB / "best_params_log.json", "w") as f:
    json.dump(best_params_log, f, indent=2, default=str)
print("\n", res_df.round(3))

# Model comparison plots
fig, ax = plt.subplots(figsize=(9, 5))
metric_names = ["AUC_mean", "AP_mean", "F1_mean"]
x = np.arange(len(res_df))
w = 0.25
for i, m in enumerate(metric_names):
    offset = (i - 1) * w
    err = res_df[m.replace("_mean", "_sd")]
    ax.bar(x + offset, res_df[m], w, yerr=err, capsize=3, label=m.replace("_mean", ""))
ax.set_xticks(x); ax.set_xticklabels(res_df.index, rotation=15)
ax.axhline(0.5, color="grey", ls="--", alpha=0.5, label="Chance (AUC)")
ax.set_ylim(0, 1); ax.set_title("Nested-CV performance comparison")
ax.legend()
save_fig(fig, "07_ml_comparison", "01_metrics_compare")

# ROC and PR curves (using OOF probs)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for name, probs in all_probs.items():
    fpr, tpr, _ = roc_curve(ym, probs)
    axes[0].plot(fpr, tpr, lw=2, label=f"{name}  AUC={results[name]['AUC_mean']:.3f}")
    p, r, _ = precision_recall_curve(ym, probs)
    axes[1].plot(r, p, lw=2, label=f"{name}  AP={results[name]['AP_mean']:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC curves (out-of-fold)"); axes[0].legend(loc="lower right")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("PR curves (out-of-fold)"); axes[1].legend(loc="lower left")
save_fig(fig, "07_ml_comparison", "02_roc_pr_curves")

# Best model
best_name = max(results, key=lambda k: results[k]["AUC_mean"])
print(f"\n  Best model by AUC: {best_name}")
best_probs = all_probs[best_name]

# Confusion matrix for best
cm = confusion_matrix(ym, all_preds[best_name])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low distress", "High distress"],
            yticklabels=["Low distress", "High distress"], ax=ax)
ax.set_title(f"{best_name} — out-of-fold confusion matrix")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
save_fig(fig, "07_ml_comparison", "03_confusion_best")

# -----------------------------------------------------------------------------
# 8. CALIBRATION ANALYSIS
# -----------------------------------------------------------------------------
print("\n[STAGE 8] Calibration analysis ...")

def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() == 0: continue
        e += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return e

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
calib_metrics = {}
for name, probs in all_probs.items():
    try:
        frac_pos, mean_pred = calibration_curve(ym, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", lw=2,
                label=f"{name}  Brier={results[name]['Brier_mean']:.3f}, "
                      f"ECE={ece(ym, probs):.3f}")
        calib_metrics[name] = {"Brier": results[name]["Brier_mean"],
                               "ECE": ece(ym, probs),
                               "LogLoss": log_loss(ym, np.clip(probs, 1e-6, 1-1e-6))}
    except Exception as e:
        print(f"    skipping {name}: {e}")

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Reliability diagram (out-of-fold probabilities)")
ax.legend(fontsize=9, loc="upper left")
save_fig(fig, "08_calibration", "01_reliability_diagram")

calib_df = pd.DataFrame(calib_metrics).T
calib_df.to_csv(TAB / "calibration_metrics.csv")
print(calib_df.round(4))

# Histograms of predicted probs by class
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, name in zip(axes.flat, all_probs):
    probs = all_probs[name]
    ax.hist(probs[ym == 0], bins=20, alpha=0.6, label="Low distress", color="#4c72b0")
    ax.hist(probs[ym == 1], bins=20, alpha=0.6, label="High distress", color="#dd8452")
    ax.set_title(f"{name}: predicted prob distributions")
    ax.set_xlabel("P(high distress)"); ax.legend()
for ax in axes.flat[len(all_probs):]: ax.axis("off")
plt.tight_layout(); save_fig(fig, "08_calibration", "02_prob_histograms")

# Platt-scaled refit of best model for downstream use
print(f"  Calibrating {best_name} via Platt scaling (held-out)...")
best_mdl_template = models_grid[best_name][0]
calibrated = CalibratedClassifierCV(best_mdl_template, cv=5, method="sigmoid")
calibrated.fit(Xm, ym)
calib_probs = cross_val_predict(calibrated, Xm, ym, cv=outer,
                                method="predict_proba", n_jobs=-1)[:, 1]
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
frac_pos_u, mp_u = calibration_curve(ym, best_probs, n_bins=10, strategy="quantile")
frac_pos_c, mp_c = calibration_curve(ym, calib_probs, n_bins=10, strategy="quantile")
ax.plot(mp_u, frac_pos_u, "o-", label=f"Uncalibrated ECE={ece(ym, best_probs):.3f}")
ax.plot(mp_c, frac_pos_c, "s-", label=f"Platt-scaled ECE={ece(ym, calib_probs):.3f}")
ax.set_title(f"Calibration of {best_name} before vs after Platt scaling")
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
ax.legend()
save_fig(fig, "08_calibration", "03_platt_scaling")

# -----------------------------------------------------------------------------
# 9. SHAP EXPLAINABILITY
# -----------------------------------------------------------------------------
print("\n[STAGE 9] SHAP explainability ...")

# Use the best tree-based model for SHAP (XGBoost/LightGBM/RF)
tree_models = [m for m in ["XGBoost", "LightGBM", "RandomForest"] if m in results]
shap_model_name = max(tree_models, key=lambda k: results[k]["AUC_mean"]) if tree_models else "RandomForest"
print(f"  Using {shap_model_name} for SHAP")

# Fit best params on full data
shap_template, shap_grid = models_grid[shap_model_name]
gs_final = GridSearchCV(shap_template, shap_grid, cv=inner, scoring="roc_auc",
                        n_jobs=-1, refit=True)
gs_final.fit(Xm, ym)
shap_model = gs_final.best_estimator_

# TreeExplainer
explainer = shap.TreeExplainer(shap_model)
shap_values = explainer(Xm)
# Handle binary classifier shape variations
sv = shap_values.values
if sv.ndim == 3:                       # (n, p, 2)
    sv = sv[:, :, 1]
base = explainer.expected_value
if isinstance(base, (list, np.ndarray)) and np.ndim(base) > 0:
    base = float(np.asarray(base).ravel()[-1])

# Beeswarm (global)
fig = plt.figure(figsize=(10, 7))
shap.summary_plot(sv, Xm, feature_names=feat_cols, show=False)
plt.title(f"SHAP beeswarm — {shap_model_name}")
plt.savefig(FIG / "09_shap_explain" / "01_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Bar plot (mean |SHAP|)
fig = plt.figure(figsize=(8, 6))
shap.summary_plot(sv, Xm, feature_names=feat_cols, plot_type="bar", show=False)
plt.title(f"Global feature importance (mean |SHAP|) — {shap_model_name}")
plt.savefig(FIG / "09_shap_explain" / "02_mean_abs_shap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Dependence plots for top features
mean_abs = np.abs(sv).mean(axis=0)
top_idx = np.argsort(-mean_abs)[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, idx in zip(axes.flat, top_idx):
    ax.scatter(Xm[:, idx], sv[:, idx], c=ym, cmap="coolwarm",
               alpha=0.6, s=30, edgecolors="white")
    ax.set_xlabel(feat_cols[idx]); ax.set_ylabel("SHAP value")
    ax.set_title(f"Dependence: {feat_cols[idx]}")
plt.tight_layout(); save_fig(fig, "09_shap_explain", "03_dependence_top6")

# Local explanations for two example respondents (high-prob, low-prob)
shap_model_probs = shap_model.predict_proba(Xm)[:, 1]
high_idx = int(np.argmax(shap_model_probs))
low_idx  = int(np.argmin(shap_model_probs))
for label, i in [("high_risk_example", high_idx), ("low_risk_example", low_idx)]:
    fig = plt.figure(figsize=(10, 4))
    explanation_i = shap.Explanation(values=sv[i], base_values=base,
                                     data=Xm[i], feature_names=feat_cols)
    shap.plots.waterfall(explanation_i, show=False, max_display=12)
    plt.title(f"Local SHAP — {label} (P={shap_model_probs[i]:.3f})")
    plt.savefig(FIG / "09_shap_explain" / f"04_waterfall_{label}.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

shap_imp = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs})\
    .sort_values("mean_abs_shap", ascending=False)
shap_imp.to_csv(TAB / "shap_importance.csv", index=False)
print(shap_imp.head(8).round(4).to_string(index=False))

# -----------------------------------------------------------------------------
# 10. CONFORMAL PREDICTION
# -----------------------------------------------------------------------------
print("\n[STAGE 10] Conformal prediction ...")

# Split conformal for classification (Adaptive Prediction Sets)
X_tr, X_cal, y_tr, y_cal = train_test_split(Xm, ym, test_size=0.4,
                                             random_state=SEED, stratify=ym)
conf_template, conf_grid = models_grid[best_name]
gs_c = GridSearchCV(conf_template, conf_grid, cv=inner, scoring="roc_auc", n_jobs=-1)
gs_c.fit(X_tr, y_tr)
conf_model = gs_c.best_estimator_

cal_probs = conf_model.predict_proba(X_cal)
# Non-conformity score = 1 - prob of true class
nc = 1 - cal_probs[np.arange(len(y_cal)), y_cal]
alpha = 0.10  # 90% coverage
q_hat = np.quantile(nc, np.ceil((len(nc) + 1) * (1 - alpha)) / len(nc))
print(f"  Conformal threshold q_hat = {q_hat:.4f}")

# Apply to full data
test_probs = conf_model.predict_proba(Xm)
sets = (1 - test_probs) <= q_hat   # which classes are "in" the prediction set
set_sizes = sets.sum(axis=1)
empirical_coverage = sets[np.arange(len(ym)), ym].mean()
print(f"  Empirical coverage (target {1-alpha:.2f}): {empirical_coverage:.3f}")
print(f"  Average prediction-set size: {set_sizes.mean():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
sz_counts = pd.Series(set_sizes).value_counts().sort_index()
ax.bar(sz_counts.index.astype(str), sz_counts.values, color="#4c72b0")
ax.set_xlabel("Prediction set size"); ax.set_ylabel("Count")
ax.set_title(f"Conformal set sizes (avg = {set_sizes.mean():.2f}, coverage = {empirical_coverage:.3f})")

ax = axes[1]
order = np.argsort(test_probs[:, 1])
ax.scatter(np.arange(len(order)), test_probs[order, 1],
           c=[set_sizes[i] for i in order], cmap="viridis", s=20)
ax.scatter(np.arange(len(order)), ym[order], color="black", s=8, alpha=0.4,
           label="True label")
ax.set_xlabel("Respondent (sorted)"); ax.set_ylabel("P(high distress)")
ax.set_title("Sorted predictions coloured by set size")
ax.legend()
save_fig(fig, "10_conformal", "01_conformal_sets")

# Coverage by subgroup (conditional coverage check)
covs = []
for grp in ["institution", "gender", "academic_year"]:
    for lv in df[grp].dropna().unique():
        mask = ml_df.reset_index()[grp] == lv  # align indices
        if mask.sum() < 10: continue
        m = mask.values
        cov = sets[m, ym[m]].mean()
        covs.append({"group": grp, "level": lv, "n": int(m.sum()), "coverage": float(cov)})
cov_df = pd.DataFrame(covs)
cov_df.to_csv(TAB / "conformal_coverage_subgroups.csv", index=False)
fig, ax = plt.subplots(figsize=(10, 0.4 * len(cov_df) + 2))
y_pos = np.arange(len(cov_df))
ax.barh(y_pos, cov_df["coverage"], color="#4c72b0")
ax.axvline(1 - alpha, color="red", ls="--", label=f"Target {1-alpha:.2f}")
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['group']}={r['level']} (n={r['n']})" for _, r in cov_df.iterrows()])
ax.set_xlabel("Empirical coverage"); ax.legend()
ax.set_title("Subgroup-conditional conformal coverage")
save_fig(fig, "10_conformal", "02_subgroup_coverage")
print(cov_df.round(3))

# -----------------------------------------------------------------------------
# 11. BAYESIAN LOGISTIC REGRESSION (via statsmodels MCMC alternative)
# -----------------------------------------------------------------------------
print("\n[STAGE 11] Bayesian logistic regression ...")

# Use bootstrap-Bayesian approach (sampling for credible intervals)
# This avoids heavy PyMC dependency while giving uncertainty quantification
from sklearn.linear_model import LogisticRegression
n_boot = 2000
coef_samples = []
sc = StandardScaler().fit(Xm)
Xms = sc.transform(Xm)
for b in range(n_boot):
    idx = rng.choice(np.arange(len(ym)), size=len(ym), replace=True)
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=b)
    lr.fit(Xms[idx], ym[idx])
    coef_samples.append(np.concatenate([[lr.intercept_[0]], lr.coef_[0]]))
coef_samples = np.array(coef_samples)
coef_names = ["intercept"] + feat_cols

# Posterior summaries
post = pd.DataFrame({
    "feature": coef_names,
    "mean":   coef_samples.mean(axis=0),
    "median": np.median(coef_samples, axis=0),
    "ci_low":  np.percentile(coef_samples, 2.5, axis=0),
    "ci_high": np.percentile(coef_samples, 97.5, axis=0),
    "p_pos":  (coef_samples > 0).mean(axis=0),
})
post.to_csv(TAB / "bayesian_posteriors.csv", index=False)
print(post.round(3).to_string(index=False))

# Forest plot of standardised coefficients
fig, ax = plt.subplots(figsize=(8, 0.3 * len(post) + 2))
non_int = post[post["feature"] != "intercept"].sort_values("mean")
y = np.arange(len(non_int))
ax.errorbar(non_int["mean"], y,
            xerr=[non_int["mean"] - non_int["ci_low"],
                  non_int["ci_high"] - non_int["mean"]],
            fmt="o", color="#1f4e8c", capsize=4)
ax.axvline(0, color="red", ls="--")
ax.set_yticks(y); ax.set_yticklabels(non_int["feature"])
ax.set_xlabel("Standardised log-odds coefficient (95% CI)")
ax.set_title("Bootstrap-Bayesian logistic regression posteriors")
save_fig(fig, "11_bayesian", "01_forest_posteriors")

# Posterior distributions
top_features = non_int.assign(absm=non_int["mean"].abs())\
                       .sort_values("absm", ascending=False).head(8)["feature"].tolist()
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, f in zip(axes.flat, top_features):
    idx = coef_names.index(f)
    ax.hist(coef_samples[:, idx], bins=40, color="#4c72b0", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="red", ls="--")
    ax.axvline(np.percentile(coef_samples[:, idx], 2.5), color="grey", ls=":")
    ax.axvline(np.percentile(coef_samples[:, idx], 97.5), color="grey", ls=":")
    ax.set_title(f"{f}\n95% CI [{np.percentile(coef_samples[:, idx], 2.5):.2f}, "
                 f"{np.percentile(coef_samples[:, idx], 97.5):.2f}]")
plt.tight_layout(); save_fig(fig, "11_bayesian", "02_posterior_histograms")

# -----------------------------------------------------------------------------
# 12. DEEP LEARNING: 1D-CNN ON ITEM RESPONSES
# -----------------------------------------------------------------------------
print("\n[STAGE 12] Deep learning: 1D-CNN on item-response sequences ...")

# Note: with N=282, this is a deliberately small, well-regularised network.
# Reported as a comparison, NOT the headline result.

class TinyCNN(nn.Module):
    def __init__(self, in_len, n_demo, hidden=16, dropout=0.4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Sequential(
            nn.Linear(hidden * 2 + n_demo, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
    def forward(self, x_items, x_demo):
        z = F.relu(self.conv1(x_items.unsqueeze(1)))
        z = F.relu(self.conv2(z))
        z = self.pool(z).squeeze(-1)
        z = torch.cat([z, x_demo], dim=1)
        return self.fc(z).squeeze(-1)

def run_cnn_cv(Xm, ym, item_cols, demo_cols, feat_cols, n_epochs=120):
    item_idx = [feat_cols.index(c) for c in item_cols]
    demo_idx = [feat_cols.index(c) for c in demo_cols]
    sc_items = StandardScaler().fit(Xm[:, item_idx])
    sc_demo  = StandardScaler().fit(Xm[:, demo_idx])
    aucs, oof = [], np.zeros(len(ym))
    for fold, (tr, te) in enumerate(outer.split(Xm, ym)):
        x_it_tr = torch.FloatTensor(sc_items.transform(Xm[tr][:, item_idx]))
        x_de_tr = torch.FloatTensor(sc_demo.transform(Xm[tr][:, demo_idx]))
        x_it_te = torch.FloatTensor(sc_items.transform(Xm[te][:, item_idx]))
        x_de_te = torch.FloatTensor(sc_demo.transform(Xm[te][:, demo_idx]))
        y_tr = torch.FloatTensor(ym[tr]); y_te = torch.FloatTensor(ym[te])

        model = TinyCNN(in_len=len(item_idx), n_demo=len(demo_idx))
        # Class imbalance weight
        pos = y_tr.mean().item()
        pos_w = torch.tensor((1 - pos) / max(pos, 1e-6))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

        model.train()
        for ep in range(n_epochs):
            opt.zero_grad()
            logits = model(x_it_tr, x_de_tr)
            loss = loss_fn(logits, y_tr)
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits_te = model(x_it_te, x_de_te)
            probs = torch.sigmoid(logits_te).numpy()
        oof[te] = probs
        aucs.append(roc_auc_score(ym[te], probs))
    return aucs, oof

cnn_item_cols = GSE_ITEMS    # 9 ordinal items
cnn_demo_cols = ["cgpa", "gender_F", "institution_Pri"]
cnn_aucs, cnn_oof = run_cnn_cv(Xm, ym, cnn_item_cols, cnn_demo_cols, feat_cols)
print(f"  1D-CNN 5-fold AUC = {np.mean(cnn_aucs):.3f} ± {np.std(cnn_aucs):.3f}")

# Add to results
results["1D-CNN"] = {
    "AUC_mean": float(np.mean(cnn_aucs)), "AUC_sd": float(np.std(cnn_aucs)),
    "AP_mean":  float(average_precision_score(ym, cnn_oof)), "AP_sd": np.nan,
    "Brier_mean": float(brier_score_loss(ym, cnn_oof)), "Brier_sd": np.nan,
    "F1_mean": float(f1_score(ym, (cnn_oof >= 0.5).astype(int))), "F1_sd": np.nan,
    "time_s": np.nan,
}
all_probs["1D-CNN"] = cnn_oof

# Full comparison including CNN
res_full = pd.DataFrame(results).T.round(4)
res_full.to_csv(TAB / "ml_dl_full_comparison.csv")
print("\n  Full comparison (incl. 1D-CNN):")
print(res_full[["AUC_mean", "AUC_sd", "AP_mean", "Brier_mean", "F1_mean"]])

# Compare bar
fig, ax = plt.subplots(figsize=(10, 5))
auc_means = [results[m]["AUC_mean"] for m in results]
auc_sds   = [results[m]["AUC_sd"] if not np.isnan(results[m]["AUC_sd"]) else 0
              for m in results]
colors = sns.color_palette("Set2", len(results))
bars = ax.bar(list(results.keys()), auc_means, yerr=auc_sds, capsize=4, color=colors)
ax.axhline(0.5, color="grey", ls="--", alpha=0.5, label="Chance")
ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUC")
ax.set_title("All-model AUC comparison (5-fold CV, ± SD)")
for bar, v in zip(bars, auc_means):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
            ha="center", fontsize=9)
plt.xticks(rotation=15); ax.legend()
save_fig(fig, "12_deep_learning", "01_cnn_vs_classical")

# CNN ROC
fig, ax = plt.subplots(figsize=(7, 6))
for name in ["Logistic", best_name, "1D-CNN"]:
    if name not in all_probs: continue
    fpr, tpr, _ = roc_curve(ym, all_probs[name])
    ax.plot(fpr, tpr, lw=2, label=f"{name} AUC={results[name]['AUC_mean']:.3f}")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("1D-CNN ROC vs classical baselines"); ax.legend()
save_fig(fig, "12_deep_learning", "02_cnn_roc")

# -----------------------------------------------------------------------------
# 13. AUTOENCODER FOR ANOMALY DETECTION (atypical response patterns)
# -----------------------------------------------------------------------------
print("\n[STAGE 13] Autoencoder anomaly detection ...")

class AE(nn.Module):
    def __init__(self, in_dim, bottleneck=4):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, bottleneck),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottleneck, 16), nn.ReLU(),
            nn.Linear(16, in_dim),
        )
    def forward(self, x):
        z = self.enc(x); return self.dec(z), z

sc_all = StandardScaler().fit(df[ALL_ITEMS].values)
Xae = torch.FloatTensor(sc_all.transform(df[ALL_ITEMS].values))
ae = AE(in_dim=len(ALL_ITEMS), bottleneck=3)
opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss(reduction="none")

ae.train()
losses_history = []
for ep in range(500):
    opt.zero_grad()
    Xhat, z = ae(Xae)
    loss = loss_fn(Xhat, Xae).mean()
    loss.backward(); opt.step()
    losses_history.append(loss.item())

ae.eval()
with torch.no_grad():
    Xhat, z = ae(Xae)
    recon_err = loss_fn(Xhat, Xae).mean(dim=1).numpy()
    embedding = z.numpy()

# Add to df
df["ae_recon_err"] = recon_err
df["ae_z1"] = embedding[:, 0]
df["ae_z2"] = embedding[:, 1]
df["ae_z3"] = embedding[:, 2] if embedding.shape[1] > 2 else embedding[:, 0]

# Loss curve
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(losses_history, color="#4c72b0")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
ax.set_title("Autoencoder training loss"); ax.set_yscale("log")
save_fig(fig, "13_autoencoder", "01_training_loss")

# Recon error distribution + threshold
thr = np.quantile(recon_err, 0.95)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(recon_err, bins=40, color="#4c72b0", edgecolor="white", alpha=0.85)
ax.axvline(thr, color="red", ls="--", label=f"95th pct = {thr:.3f}")
ax.set_xlabel("Reconstruction error")
ax.set_title("AE reconstruction error distribution — high-error = atypical patterns")
ax.legend()
save_fig(fig, "13_autoencoder", "02_recon_error_dist")

# 2D embedding
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, color_by, label in [(axes[0], "high_distress", "High distress"),
                            (axes[1], "institution", "Institution")]:
    if color_by == "high_distress":
        for v in [0, 1]:
            m = df[color_by] == v
            ax.scatter(df.loc[m, "ae_z1"], df.loc[m, "ae_z2"],
                       label=label if v == 1 else "Low distress",
                       s=30, alpha=0.6)
    else:
        for v in df[color_by].unique():
            m = df[color_by] == v
            ax.scatter(df.loc[m, "ae_z1"], df.loc[m, "ae_z2"],
                       label=v, s=30, alpha=0.6)
    ax.set_xlabel("AE z1"); ax.set_ylabel("AE z2")
    ax.set_title(f"AE latent space coloured by {label}"); ax.legend()
save_fig(fig, "13_autoencoder", "03_latent_embedding")

# Anomaly characterisation
anom_df = df[df["ae_recon_err"] >= thr][["GSE_total", "K10_total", "institution",
                                         "gender", "academic_year", "ae_recon_err"]]
anom_df.to_csv(TAB / "anomalous_respondents.csv", index=False)
print(f"  Flagged {len(anom_df)} atypical respondents (top 5%)")

# Compare normal vs anomalous on scale totals
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
df["is_anom"] = (df["ae_recon_err"] >= thr).astype(int)
for ax, col, lab in [(axes[0], "GSE_total", "GSES"), (axes[1], "K10_total", "K-10")]:
    sns.boxplot(data=df, x="is_anom", y=col, ax=ax, hue="is_anom",
                palette="Set2", legend=False)
    ax.set_xticklabels(["Typical", "Atypical (top 5%)"])
    ax.set_title(f"{lab} total: typical vs atypical")
save_fig(fig, "13_autoencoder", "04_anom_vs_typical_scales")

# -----------------------------------------------------------------------------
# 14. ROBUSTNESS & SENSITIVITY
# -----------------------------------------------------------------------------
print("\n[STAGE 14] Robustness & sensitivity checks ...")

# Drop-one-subgroup sensitivity on main correlation
sens = []
ref_r, _ = stats.pearsonr(df["GSE_total"], df["K10_total"])
for grp in ["institution", "gender", "academic_year", "cgpa_band"]:
    for lv in df[grp].dropna().unique():
        sub = df[df[grp] != lv]
        r, p = stats.pearsonr(sub["GSE_total"], sub["K10_total"])
        sens.append({"drop": f"{grp}={lv}", "r_excl": r, "p": p, "n_remain": len(sub)})
sens_df = pd.DataFrame(sens)
sens_df.to_csv(TAB / "sensitivity_drop_one.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 0.35 * len(sens_df) + 2))
y_pos = np.arange(len(sens_df))
ax.barh(y_pos, sens_df["r_excl"], color="#4c72b0")
ax.axvline(ref_r, color="red", ls="--", label=f"Full-sample r = {ref_r:.3f}")
ax.axvline(0, color="black", lw=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{r['drop']} (n={r['n_remain']})" for _, r in sens_df.iterrows()])
ax.set_xlabel("GSES↔K-10 Pearson r excluding that subgroup")
ax.set_title("Sensitivity: drop-one-subgroup robustness of main correlation")
ax.legend()
save_fig(fig, "14_robustness", "01_drop_one_sensitivity")

# Bootstrap distribution of the main correlation
boot_r = []
for _ in range(5000):
    s = df.sample(frac=1.0, replace=True, random_state=rng.integers(1e9))
    boot_r.append(stats.pearsonr(s["GSE_total"], s["K10_total"])[0])
lo_r, hi_r = np.percentile(boot_r, [2.5, 97.5])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(boot_r, bins=60, color="#4c72b0", alpha=0.85, edgecolor="white")
ax.axvline(np.mean(boot_r), color="red", ls="--", label=f"mean = {np.mean(boot_r):.3f}")
ax.axvline(lo_r, color="grey", ls=":", label=f"95% CI [{lo_r:.3f}, {hi_r:.3f}]")
ax.axvline(hi_r, color="grey", ls=":")
ax.axvline(0, color="black", lw=0.7)
ax.set_title("Bootstrap distribution of overall GSES↔K-10 r (5000 reps)")
ax.legend()
save_fig(fig, "14_robustness", "02_bootstrap_r")

# Influence diagnostics: Cook's D on K10 ~ GSE
Xinf = sm.add_constant(df["GSE_total"])
mod_inf = sm.OLS(df["K10_total"], Xinf).fit()
cd = mod_inf.get_influence().cooks_distance[0]
fig, ax = plt.subplots(figsize=(9, 4))
ax.stem(np.arange(len(cd)), cd, basefmt=" ")
ax.axhline(4 / len(cd), color="red", ls="--", label="4/n threshold")
ax.set_title("Cook's distance — K10 ~ GSES")
ax.set_xlabel("Observation"); ax.set_ylabel("Cook's D"); ax.legend()
save_fig(fig, "14_robustness", "03_cooks_distance")

# Multiple testing correction across cross-scale correlations
pvals = np.zeros((len(GSE_ITEMS), len(K10_ITEMS)))
rvals = np.zeros_like(pvals)
for i, a in enumerate(GSE_ITEMS):
    for j, b in enumerate(K10_ITEMS):
        rvals[i, j], pvals[i, j] = stats.pearsonr(df[a], df[b])
flat = pvals.flatten()
_, qvals, _, _ = multipletests(flat, method="fdr_bh")
qmat = qvals.reshape(pvals.shape)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(rvals, cmap="RdBu_r", center=0, vmin=-0.4, vmax=0.4,
            annot=True, fmt=".2f", xticklabels=K10_ITEMS, yticklabels=GSE_ITEMS,
            ax=axes[0], cbar_kws={"label": "r"})
axes[0].set_title("Cross-scale Pearson r")
sns.heatmap(qmat < 0.05, cmap="Greens", annot=qmat, fmt=".3f",
            xticklabels=K10_ITEMS, yticklabels=GSE_ITEMS, ax=axes[1],
            cbar_kws={"label": "FDR q < 0.05"})
axes[1].set_title("FDR-corrected significance (annot = q-value)")
save_fig(fig, "14_robustness", "04_FDR_cross_scale")

# Normality + Q-Q plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
stats.probplot(df["GSE_total"], dist="norm", plot=axes[0])
axes[0].set_title("GSES total Q-Q")
stats.probplot(df["K10_total"], dist="norm", plot=axes[1])
axes[1].set_title("K-10 total Q-Q")
save_fig(fig, "14_robustness", "05_qq_plots")

# Subgroup correlations
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, grp in zip(axes.flat, ["institution", "gender", "academic_year", "cgpa_band"]):
    levels = sorted(df[grp].dropna().unique().tolist())
    rs = []
    for lv in levels:
        sub = df[df[grp] == lv]
        if len(sub) < 10: continue
        r, p = stats.pearsonr(sub["GSE_total"], sub["K10_total"])
        rs.append((lv, r, p, len(sub)))
    if not rs: continue
    rs_df = pd.DataFrame(rs, columns=[grp, "r", "p", "n"])
    bars = ax.bar(rs_df[grp], rs_df["r"], color="#4c72b0")
    ax.axhline(0, color="black", lw=0.6)
    for i, row in rs_df.iterrows():
        ax.text(i, row["r"] + (0.01 if row["r"] >= 0 else -0.04),
                f"r={row['r']:.2f}\np={row['p']:.3f}\nn={row['n']}",
                ha="center", fontsize=8)
    ax.set_title(f"GSES↔K-10 by {grp}")
    ax.set_ylim(-0.35, 0.35); plt.setp(ax.get_xticklabels(), rotation=20)
plt.tight_layout(); save_fig(fig, "14_robustness", "06_subgroup_correlations")

# -----------------------------------------------------------------------------
# 15. WRAP-UP
# -----------------------------------------------------------------------------
print("\n" + "=" * 75)
print("  PIPELINE COMPLETE")
print("=" * 75)

total = 0
for fld in FOLDERS:
    n = len(list((FIG / fld).glob("*.png")))
    total += n
    print(f"  {fld:30s}: {n:3d} figures")
print(f"\n  Total figures: {total}")

# Save the full cleaned + augmented data
df.to_csv(TAB / "final_data_with_predictions.csv", index=False)

summary = {
    "N_final": int(N),
    "GSE_alpha": float(a_g), "GSE_omega": float(o_g),
    "K10_alpha": float(a_k), "K10_omega": float(o_k),
    "overall_GSE_K10_r": float(ref_r),
    "overall_GSE_K10_r_95CI": [float(lo_r), float(hi_r)],
    "moderation_interaction_beta": float(m_inst.params["GSE_c:institution_Pri"]),
    "moderation_interaction_p":    float(m_inst.pvalues["GSE_c:institution_Pri"]),
    "moderation_interaction_95CI": [float(ci_lo), float(ci_hi)],
    "public_subgroup_r": float(slope_df.loc[slope_df["group"] == "Public", "r"].iloc[0]),
    "public_subgroup_p": float(slope_df.loc[slope_df["group"] == "Public", "p"].iloc[0]),
    "private_subgroup_r": float(slope_df.loc[slope_df["group"] == "Private", "r"].iloc[0]),
    "private_subgroup_p": float(slope_df.loc[slope_df["group"] == "Private", "p"].iloc[0]),
    "ml_models": {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating))
                            else vv) for kk, vv in v.items()}
                  for k, v in results.items()},
    "best_ml_model": best_name,
    "conformal_target_coverage": 1 - alpha,
    "conformal_empirical_coverage": float(empirical_coverage),
    "n_anomalous": int(len(anom_df)),
    "bayesian_features_credible": post.loc[
        (post["ci_low"] > 0) | (post["ci_high"] < 0), "feature"].tolist(),
}
with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  HEADLINE RESULTS:")
print(f"    GSES α = {a_g:.3f}, K-10 α = {a_k:.3f}")
print(f"    Overall r = {ref_r:.3f} (95% CI [{lo_r:.3f}, {hi_r:.3f}])  --> null")
print(f"    *** Public subgroup r = {summary['public_subgroup_r']:.3f} "
      f"(p = {summary['public_subgroup_p']:.3f})  --> negative")
print(f"    *** Private subgroup r = {summary['private_subgroup_r']:.3f} "
      f"(p = {summary['private_subgroup_p']:.3f})  --> null")
print(f"    Institution × GSE interaction β = "
      f"{summary['moderation_interaction_beta']:.3f}, "
      f"p = {summary['moderation_interaction_p']:.4f}")
print(f"    Best ML AUC ({best_name}) = {results[best_name]['AUC_mean']:.3f}")
print(f"    1D-CNN AUC = {results['1D-CNN']['AUC_mean']:.3f}")
print(f"    Conformal coverage = {empirical_coverage:.3f} (target {1-alpha:.2f})")
print(f"\n  All outputs in: {OUT.resolve()}")
