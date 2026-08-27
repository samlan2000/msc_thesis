"""
processing_shl2.py
───────────────────
SHL2 station validation: satellite (S3A/S3B OLCI) WASI inversion vs in-situ
Chl-a and phytoplankton community (abundance/biovolume) match-ups.

Adapted from pixel_processor_shl2_tx_V3.py to:
  - use the current PixelProcessor / MiniWASIsafe model, which only
    supports phytoplankton groups C_0-C_5 (the old C_6/C_7/C_8/C_mie
    groups no longer exist in the model, so all cyanobacteria compare
    against C_2 alone instead of a C_2+C_7+C_8 sum).
  - read both Chl-a and phytoplankton abundance/biovolume in-situ data
    from the single combined matchups csv (it already contains both).
  - replace the Chl-a time series and 1:1 scatter with an exact copy of
    the corresponding plots from pixel_processor_shl2_allvars_V2.py,
    and drop the per-group (C_0-C_5) time series plot entirely.
  - get the Chl-a "max depth" series for the depth-vs-Secchi panel from
    the depth-tagged chla_* columns already in the matchups csv (per
    date: the depth whose chla_* column has the highest value), instead
    of a separate long-format profile file. Uses the FULL depth profile
    (chla_0 ... chla_30, see CHLA_DEPTH_COLS) rather than just the
    near-surface columns used for chla_mean, since the chlorophyll
    maximum can sit below 7.5m.
  - save every figure as a .png under PLOTS_DIR (relative to this
    script's location) in addition to showing it interactively.

All paths below are overridable via environment variables so that
main_shl2.py can control them centrally. Defaults are only used when
this script is run standalone.
"""

import os
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from statsmodels.stats.multitest import multipletests
from PixelProcessor import SinglePixelProcessor

# ─────────────────────────────────────────────
# Font — change this to switch the font used everywhere in this script
# ─────────────────────────────────────────────
FONT_FAMILY = "Times New Roman"
plt.rcParams["font.family"] = FONT_FAMILY
# Bump the default font size +1.5pt (TNR renders slightly smaller than
# matplotlib's default DejaVu Sans at the same point size) so every text
# element that does NOT set an explicit fontsize (e.g. some tick labels)
# grows in step with the ones below that do. Uses an ABSOLUTE value (not
# +=) so re-running this script in the same session/kernel never compounds
# the font size on repeated runs.
plt.rcParams["font.size"] = 10 + 1.5  # matplotlib default (10) + bump

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

IMG_DIR     = os.environ.get("SHL2_BSQ_DIR", r"C:\MSc_thesis_data\satellite\shl2\bsq")
INSITU_PATH = os.environ.get(
    "SHL2_MATCHUPS_CSV",
    r"C:\MSc_thesis_data\insitu\shl2\matchups\matchups_shl2_v3_full_with_phyto.csv",
)
SECCHI_PATH = os.environ.get(
    "SHL2_SECCHI_CSV",
    r"C:\MSc_thesis_data\insitu\shl2\secchi\France_Geneva_secchi_postprocessed.csv",
)

# outputs_L3/plots_shl2, reconstructed relative to this script's location
PLOTS_DIR = Path(os.environ.get("SHL2_PLOTS_DIR", str(BASE_DIR / "outputs_L3" / "plots_shl2")))
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Depth [m] represented by each chla_* column in the matchups csv.
# CHLA_COLS (near-surface only, 0-7.5m) is used for the satellite-comparable
# surface mean (chla_mean). CHLA_DEPTH_COLS is the FULL depth profile
# available in the file and is used when searching for the depth of the
# chlorophyll maximum, since the max can occur below 7.5m (deep chl max).
CHLA_COL_DEPTHS = {
    "chla_0": 0, "chla_1": 1, "chla_2p5": 2.5,
    "chla_3p5": 3.5, "chla_5": 5, "chla_7p5": 7.5,
}
CHLA_COLS = list(CHLA_COL_DEPTHS)

CHLA_DEPTH_COLS = {
    **CHLA_COL_DEPTHS,
    "chla_10": 10, "chla_15": 15, "chla_20": 20, "chla_30": 30,
}

C_COMPONENTS     = ["C_0", "C_1", "C_2", "C_3", "C_4", "C_5"]
C_PHY_COMPONENTS = C_COMPONENTS

# C_1-C_5: taxa groups used throughout (stacked-bar community plot,
# per-group scatter). This is the full set the current model supports.
C_TAXA = ["C_1", "C_2", "C_3", "C_4", "C_5"]

# Maps each model group to its in-situ abundance column.
# C_0 has no direct in-situ counterpart (bulk Chl-a).
C_TO_ABUNDANCE = {
    "C_1": "abundance_cryptophyta",
    "C_2": "abundance_cyanobacteria",
    "C_3": "abundance_diatoms",
    "C_4": "abundance_dinoflagellates",
    "C_5": "abundance_green algae",
}

C_TO_BIOVOLUME = {
    c: col.replace("abundance_", "biovolume_")
    for c, col in C_TO_ABUNDANCE.items()
}

# Biovolume-only in-situ groups with no model (C_x) or abundance counterpart
# — no satellite retrieval or abundance equivalent exists, so these are only
# ever added to the in-situ biovolume stacking, never to the abundance or
# satellite-retrieved proportions.
BIOVOL_EXTRA_TO_COL = {
    "Chrysophyceae": "biovolume_chrysophyceae",
}
BIOVOL_TAXA = C_TAXA + list(BIOVOL_EXTRA_TO_COL.keys())

GROUP_LABELS = {
    "C_0": "C$_0$ (bulk Chl-a)",
    "C_1": "C$_1$ – Cryptophyta",
    "C_2": "C$_2$ – Cyanobacteria",
    "C_3": "C$_3$ – Diatoms",
    "C_4": "C$_4$ – Dinoflagellates",
    "C_5": "C$_5$ – Green algae",
    "Chrysophyceae": "Chrysophyceae",
}

# Colors for C_1-C_5 (used in the stacked-bar plot)
TAXA_COLORS = {
    "C_1": "#916d00",
    "C_2": "#910065",
    "C_3": "#aaf7ff",
    "C_4": "#672f00",
    "C_5": "#5ae400",
    "Chrysophyceae": "#ff8c00",
}

# Colors for C_0 … C_5 (used in the per-group scatter plot)
GROUP_COLORS = [
    "#916d00",
    "#910065",
    "#aaf7ff",
    "#672f00",
    "#5ae400",
    "#1f77b4",
]

# Current MiniWASIsafe model only knows C_0-C_5, C_x, C_y — vary/init are
# restricted to those keys (no C_6/C_7/C_8/C_mie).
PROCESSOR_KWARGS = dict(
    a_norm_y_from_file=False,
    station_name="shl2",
    weights=[0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    i_offset=0,
    j_offset=0,
    valid_pixel_min=2000,
    vary={
        "C_0": False, "C_1": True, "C_2": True, "C_3": True,
        "C_4": True,  "C_5": True, "C_x": True,  "C_y": True,
    },
    init={
        "C_0": 0, "C_1": 1, "C_2": 1, "C_3": 1, "C_4": 1,
        "C_5": 1, "C_x": 1, "C_y": 0.1,
    },
)

# ─────────────────────────────────────────────
# Load in-situ data (Chl-a + phytoplankton abundance/biovolume, one file)
# ─────────────────────────────────────────────
insitu_df = pd.read_csv(INSITU_PATH, sep=";", encoding="utf-8")
print(insitu_df)
insitu_df["chla_mean"] = insitu_df[CHLA_COLS].mean(axis=1)
insitu_lookup = insitu_df.groupby("insitu_date")["chla_mean"].first()

phyto_abund_cols = [c for c in insitu_df.columns if c.startswith("abundance_")]
phyto_biovol_cols = [c for c in insitu_df.columns if c.startswith("biovolume_")]
phyto_lookup = insitu_df.groupby("insitu_date")[phyto_abund_cols + phyto_biovol_cols].first()
phyto_lookup = phyto_lookup[(phyto_lookup[phyto_abund_cols + phyto_biovol_cols] != 0).any(axis=1)]

# ─────────────────────────────────────────────
# Process images
# ─────────────────────────────────────────────
def parse_date(img_name):
    raw = img_name.split("_")[-1][:8]
    return f"{raw[6:8]}.{raw[4:6]}.{raw[:4]}"

def make_result_store():
    store = {key: {} for key in ["C_phy", "C_phy_insitu", "C_x", "C_y", "rmse"]}
    store.update({c: {} for c in C_COMPONENTS})
    # per-group in-situ abundance and biovolume (keyed by date)
    store["abund_insitu"] = {c: {} for c in C_TO_ABUNDANCE}
    store["biovol_insitu"] = {c: {} for c in C_TO_ABUNDANCE}
    return store

results = {"S3A": make_result_store(), "S3B": make_result_store()}
valid_dates = set()

for img in os.listdir(IMG_DIR):
    if img.startswith("_") or not img.endswith(".bsq"):
        continue

    satellite = "S3A" if "S3A" in img else "S3B" if "S3B" in img else None
    if satellite is None:
        continue

    date_str = parse_date(img)
    if date_str not in insitu_lookup.index:
        print(f"No in-situ match for {date_str}, skipping.")
        continue

    print(f"{'='*80}\n{img}")
    res = SinglePixelProcessor(os.path.join(IMG_DIR, img), **PROCESSOR_KWARGS)

    if not res.inv:
        continue

    p = res.inv.params
    store = results[satellite]

    # Individual C components
    for c in C_COMPONENTS:
        store[c][date_str] = p[c].value

    # Derived / auxiliary
    store["C_phy"][date_str]        = sum(p[c].value for c in C_PHY_COMPONENTS)
    store["C_phy_insitu"][date_str] = float(insitu_lookup[date_str])
    store["C_x"][date_str]          = p["C_x"].value
    store["C_y"][date_str]          = p["C_y"].value
    store["rmse"][date_str]         = res.rmse
    valid_dates.add(date_str)

    # Per-group in-situ abundance & biovolume (only if date present in phyto data)
    if date_str in phyto_lookup.index:
        row = phyto_lookup.loc[date_str]
        for c, abund_col in C_TO_ABUNDANCE.items():
            store["abund_insitu"][c][date_str]  = float(row[abund_col])
            bvol_col = abund_col.replace("abundance_", "biovolume_")
            store["biovol_insitu"][c][date_str] = float(row[bvol_col]) if bvol_col in row.index else np.nan

# ─────────────────────────────────────────────
# Export set of (satellite, date) pairs actually used in a comparison —
# i.e. successfully inverted image with >=1 non-NaN in-situ match (Chl-a
# and/or phyto abundance/biovolume) — for cross-chain image-usage counting
# (see count_used_images.py). Dates normalized to ISO YYYY-MM-DD to match
# db_thetis*.pkl's key format.
# ─────────────────────────────────────────────
import pickle
from datetime import datetime as _dt

def _to_iso(date_str):
    return _dt.strptime(date_str, "%d.%m.%Y").strftime("%Y-%m-%d")

used_images = set()
for sat, store in results.items():
    for date_str, val in store["C_phy_insitu"].items():
        if np.isfinite(val):
            used_images.add((sat, _to_iso(date_str)))
    for group_key in ("abund_insitu", "biovol_insitu"):
        for date_dict in store[group_key].values():
            for date_str, val in date_dict.items():
                if np.isfinite(val):
                    used_images.add((sat, _to_iso(date_str)))

used_images_path = BASE_DIR / "outputs_L3" / "used_images_shl2.pkl"
with open(used_images_path, "wb") as f:
    pickle.dump(used_images, f)
print(f"\n── Used images (shl2) saved → {used_images_path}  ({len(used_images)} sat/date pairs) ──")

# Convert valid retrieval dates to normalized datetime index
valid_dates = pd.to_datetime(list(valid_dates), dayfirst=True).normalize()
valid_dates = pd.Index(valid_dates)

# ─────────────────────────────────────────────
# Chl-a max depth, from the depth-tagged chla_* columns in the matchups csv
# ─────────────────────────────────────────────
# One row per date (first occurrence, same as insitu_lookup above).
# Search the FULL depth profile (0-30m), not just the near-surface columns
# used for chla_mean, so a deep chlorophyll maximum below 7.5m is found too.
chla_wide = insitu_df.groupby("insitu_date")[list(CHLA_DEPTH_COLS)].first()

# Column with the highest chla value per date -> its depth
chla_max_col = chla_wide.idxmax(axis=1, skipna=True)
chla_max_depth_lookup = chla_max_col.map(CHLA_DEPTH_COLS)

# Index by normalized datetime (dates are "dd.mm.yyyy", same as insitu_date elsewhere)
chla_max_depth_lookup.index = pd.to_datetime(
    chla_max_depth_lookup.index, format="%d.%m.%Y"
).normalize()

# Keep only dates with valid retrievals and a defined max-depth column
chla_max_depth_lookup = chla_max_depth_lookup.dropna()
chla_max_depth_lookup = chla_max_depth_lookup[chla_max_depth_lookup.index.isin(valid_dates)]

# ─────────────────────────────────────────────
# Load Secchi data
# ─────────────────────────────────────────────
secchi_df = pd.read_csv(SECCHI_PATH, sep=";", encoding="utf-8-sig")

secchi_df = secchi_df[secchi_df["time"] > 1481756400]

secchi_df["datetime"] = pd.to_datetime(secchi_df["datetime"])

secchi_df["date"] = secchi_df["datetime"].dt.normalize()

# Keep only dates with valid retrievals
secchi_df = secchi_df[
    secchi_df["date"].isin(valid_dates)
]

secchi_df = secchi_df.sort_values("datetime")

secchi_df_1 = secchi_df[secchi_df["time"] < 1654552800]
secchi_df_2 = secchi_df[secchi_df["time"] > 1654552800]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def to_series(store, key):
    dates = sorted(store[key])
    t     = pd.to_datetime(dates, format="%d.%m.%Y")
    vals  = np.array([store[key][d] for d in dates])
    return t, vals

def paired_series(store):
    """Return (t, sat, insitu) for dates present in both C_phy and C_phy_insitu."""
    dates = sorted(set(store["C_phy"]) & set(store["C_phy_insitu"]))
    t     = pd.to_datetime(dates, format="%d.%m.%Y")
    sat   = np.array([store["C_phy"][d]        for d in dates])
    ins   = np.array([store["C_phy_insitu"][d]  for d in dates])
    return t, sat, ins

def compute_stats(x, y):
    """x = satellite / retrieval, y = reference. Same convention/formula as
    plotting_thetis.py's compute_stats: log10-ratio bias[%] / MdSA[%], RMSE,
    Pearson r. Also returns a two-sided p-value (not part of plotting_thetis's
    version, but kept here for the FDR significance summary further down)."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return dict(r=np.nan, p=np.nan, bias=np.nan, rmse=np.nan, mdsa=np.nan, n=0)
    x, y = x[mask], y[mask]
    if mask.sum() < 3:
        # pearsonr needs n>=3 to return a defined p-value
        r, pval = (np.nan, np.nan)
    else:
        r, pval = pearsonr(x, y)
    pos = (x > 0) & (y > 0)
    if pos.sum() >= 1:
        ratio = x[pos] / y[pos]
        MR = np.median(np.log10(ratio))
        bias = (10 ** np.abs(MR) - 1) * np.sign(MR) * 100
        mdsa = 100 * (np.exp(np.median(np.abs(np.log(ratio)))) - 1)
    else:
        bias, mdsa = np.nan, np.nan
    return dict(r=r, p=pval, bias=float(bias), rmse=float(np.sqrt(mean_squared_error(y, x))),
                mdsa=float(mdsa), n=int(mask.sum()))

def round_rmse(x):
    """Round to two significant non-zero decimal places (matches plotting_thetis.py)."""
    if not np.isfinite(x) or x == 0:
        return x
    first_sig = -int(math.floor(math.log10(abs(x))))
    decimals = max(first_sig + 1, 0)
    if x > 1:
        decimals = 2
    return round(x, decimals)

_csv_rows = []  # accumulated CHL + phyto stats, saved to res_summary_shl2.csv at the end

def log_stat(section, variable, satellite, st):
    """Full metrics row (CHL): N, r, bias[%], RMSE, MdSA[%]."""
    _csv_rows.append(dict(
        section=section, variable=variable, satellite=satellite, N=st["n"],
        r=round(st["r"], 2) if np.isfinite(st["r"]) else np.nan,
        bias_pct=round(st["bias"], 1) if np.isfinite(st["bias"]) else np.nan,
        RMSE=round_rmse(st["rmse"]) if np.isfinite(st["rmse"]) else np.nan,
        MdSA_pct=round(st["mdsa"], 1) if np.isfinite(st["mdsa"]) else np.nan,
    ))

def log_stat_corr(section, variable, satellite, st):
    """Correlation-only row (phyto abundance/biovolume): N, r, p-value.
    No bias/RMSE/MdSA — not meaningful across the different units involved
    (model concentration vs. cell counts / biovolume)."""
    _csv_rows.append(dict(
        section=section, variable=variable, satellite=satellite, N=st["n"],
        r=round(st["r"], 2) if np.isfinite(st["r"]) else np.nan,
        p=round(st["p"], 4) if np.isfinite(st["p"]) else np.nan,
    ))

# ─────────────────────────────────────────────
# Extract arrays
# ─────────────────────────────────────────────
t_A, sat_A, ins_A = paired_series(results["S3A"])
t_B, sat_B, ins_B = paired_series(results["S3B"])

t_Acx, cx_A = to_series(results["S3A"], "C_x")
t_Bcx, cx_B = to_series(results["S3B"], "C_x")
t_Acy, cy_A = to_series(results["S3A"], "C_y")
t_Bcy, cy_B = to_series(results["S3B"], "C_y")

# ─── Chl-a max depth time series (from profile file, same pattern as Secchi)
t_depth    = pd.DatetimeIndex(chla_max_depth_lookup.index)
depth_vals = chla_max_depth_lookup.values

sort_idx   = np.argsort(t_depth)
t_depth    = t_depth[sort_idx]
depth_vals = depth_vals[sort_idx]

# ─────────────────────────────────────────────
# Plot 1 – Time series: Chl-a (+ max depth + Secchi), NAP, CDOM
# ─────────────────────────────────────────────

# ── Subplot 0: Chl-a with secondary axis for depth metrics ──────────────────
fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

# ── Subplot 0: Chl-a ────────────────────────────────────────────────────────
ax0 = axs[0]
ax0.scatter(t_A, ins_A, marker="x", color="black", alpha=0.8, label="In situ CHL$_S$")
ax0.scatter(t_B, ins_B, marker="x", color="black", alpha=0.8)
ax0.scatter(t_A, sat_A, color="tab:green", s=40, label="S3A CPHY$_{S3}$")
ax0.scatter(t_B, sat_B, marker="s", color="tab:green", s=40, label="S3B CPHY$_{S3}$")
ax0.set_ylabel("CHL$_S$ / CPHY$_{S3}$ \n [mg m$^{-3}$]", fontsize=15.5)
ax0.grid(True, alpha=0.3)
ax0.legend(fontsize=13.5, framealpha=0.9)
ax0.set_ylim(0, 15)

# ── Subplot 1: Chl-a max depth + Secchi depth ───────────────────────────────
ax1 = axs[1]

ax1.scatter(
    t_depth,
    depth_vals,
    color="tab:green",
    s=40,
    marker="^",
    label="CHL$_S$ max depth"
)

ax1.scatter(
    secchi_df["datetime"],
    secchi_df["secchi"],
    s=40,
    color="black",
    marker="v",
    alpha=0.8,
    label="Secchi depth"
)

# -------------------------------------------------------------------------
# Draw vertical lines where both measurements exist on same date
# -------------------------------------------------------------------------

# Create lookup from chl max depth
depth_lookup = pd.DataFrame({
    "datetime": pd.to_datetime(t_depth),
    "chl_max_depth": depth_vals
})

depth_lookup["date"] = depth_lookup["datetime"].dt.normalize()

# Create lookup from secchi
secchi_lookup = secchi_df.copy()
secchi_lookup["date"] = secchi_lookup["datetime"].dt.normalize()

# Merge on calendar date
merged = pd.merge(
    depth_lookup[["date", "datetime", "chl_max_depth"]],
    secchi_lookup[["date", "secchi"]],
    on="date",
    how="inner"
)

# Draw connecting lines
for _, row in merged.iterrows():
    ax1.plot(
        [row["datetime"], row["datetime"]],
        [row["chl_max_depth"], row["secchi"]],
        color="gray",
        alpha=0.5,
        linewidth=1
    )

ax1.set_ylabel("Depth [m]", fontsize=15.5)
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=13.5, framealpha=0.9)

# ── Subplot 2: NAP + CDOM (dual axis) ──────────────────────────────────────
ax2 = axs[2]
ax2b = ax2.twinx()

# NAP
nap1 = ax2.scatter(
    t_Acx,
    cx_A,
    color="tab:cyan",
    s=40,
    label="S3A NAP"
)

nap2 = ax2.scatter(
    t_Bcx,
    cx_B,
    color="tab:cyan",
    s=40,
    marker="s",
    label="S3B NAP"
)

ax2.set_ylabel("NAP$_{S3}$ [g m$^{-3}$]", fontsize=15.5)
ax2.grid(True, alpha=0.3)

# CDOM
cdom1 = ax2b.scatter(
    t_Acy,
    cy_A,
    color="tab:brown",
    s=40,
    label="S3A CDOM"
)

cdom2 = ax2b.scatter(
    t_Bcy,
    cy_B,
    color="tab:brown",
    s=40,
    marker="s",
    label="S3B CDOM"
)

ax2b.set_ylabel("CDOM$_{S3}$ [m$^{-1}$]", fontsize=15.5)

# Shared x label
ax2.set_xlabel("Time", fontsize=15.5)

# Combined legend
handles = [nap1, nap2, cdom1, cdom2]
labels = [h.get_label() for h in handles]

ax2.legend(handles, labels, fontsize=12.5, loc="upper left", framealpha=0.9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "1_timeseries_chla_secchi_nap_cdom.png", dpi=300, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# Plot 2 – Stacked-bar phytoplankton community (proportions)
# ─────────────────────────────────────────────

# ── Collect dates with biovolume data ────────────────────────────────────────
insitu_phy_all = {}
for sat in ("S3A", "S3B"):
    for d, v in results[sat]["C_phy_insitu"].items():
        insitu_phy_all[d] = v

bar_dates = sorted(
    (d for d in insitu_phy_all if d in phyto_lookup.index),
    key=lambda d: pd.to_datetime(d, format="%d.%m.%Y")
)

bar_t   = pd.to_datetime(bar_dates, format="%d.%m.%Y")
n_dates = len(bar_dates)

# ── Split dates across four rows (more horizontal space per date) ────────────
N_BAR_ROWS  = 4
split       = int(np.ceil(n_dates / N_BAR_ROWS))
date_splits = [bar_dates[i * split:(i + 1) * split] for i in range(N_BAR_ROWS)]
t_splits    = [bar_t[i * split:(i + 1) * split] for i in range(N_BAR_ROWS)]

# ── In-situ biovolume proportions ────────────────────────────────────────────
# Includes BIOVOL_EXTRA_TO_COL groups (e.g. Chrysophyceae) alongside C_1-C_5 —
# these have no satellite/abundance counterpart, so they only ever appear in
# the in-situ biovolume proportions below.
BIOVOL_TAXA_TO_COL = {**C_TO_BIOVOLUME, **BIOVOL_EXTRA_TO_COL}

biovol = {}
for c in BIOVOL_TAXA:
    bvol_col = BIOVOL_TAXA_TO_COL[c]
    biovol[c] = np.array([
        float(phyto_lookup.loc[d, bvol_col])
        if (d in phyto_lookup.index and bvol_col in phyto_lookup.columns)
        else 0.0
        for d in bar_dates
    ])

biovol_total = sum(biovol[c] for c in BIOVOL_TAXA)
biovol_total = np.where(biovol_total == 0, np.nan, biovol_total)
prop_insitu  = {c: biovol[c] / biovol_total for c in BIOVOL_TAXA}

# ── In-situ abundance proportions ────────────────────────────────────────────
abund = {}
for c in C_TAXA:
    abund_col = C_TO_ABUNDANCE[c]
    abund[c] = np.array([
        float(phyto_lookup.loc[d, abund_col])
        if (d in phyto_lookup.index and abund_col in phyto_lookup.columns)
        else 0.0
        for d in bar_dates
    ])

abund_total = sum(abund[c] for c in C_TAXA)
abund_total = np.where(abund_total == 0, np.nan, abund_total)
prop_abund  = {c: abund[c] / abund_total for c in C_TAXA}

# ── Retrieved proportions per satellite ──────────────────────────────────────
def retrieved_props(sat, dates):
    store    = results[sat]
    raw      = {c: np.array([store[c].get(d, np.nan) for d in dates]) for c in C_TAXA}
    raw_total = sum(raw[c] for c in C_TAXA)
    raw_total = np.where(raw_total == 0, np.nan, raw_total)
    props    = {c: raw[c] / raw_total for c in C_TAXA}
    has_data = np.array([d in results[sat]["C_phy"] for d in dates])
    return props, has_data

prop_A, has_A = retrieved_props("S3A", bar_dates)
prop_B, has_B = retrieved_props("S3B", bar_dates)

# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_bar(ax, xi, props, i, hatch, alpha, edge="black", taxa=C_TAXA):
    bottom = 0.0
    for c in taxa:
        val = float(np.nan_to_num(props[c][i], nan=0.0))
        ax.bar(
            xi, val,
            bottom=bottom,
            width=w,
            color=TAXA_COLORS[c],
            alpha=alpha,
            hatch=hatch,
            edgecolor=edge,
            linewidth=0.5,
        )
        bottom += val

def draw_missing_bar(ax, xi):
    """Full-height gray hatched placeholder for a slot with no retrieval."""
    ax.bar(
        xi, 1.0,
        width=w,
        color="0.85",
        alpha=0.9,
        hatch="\\\\",
        edgecolor="black",
        linewidth=0.5,
    )

# ── Figure: four rows ─────────────────────────────────────────────────────────
w   = 0.15   # slightly narrower bars, more breathing room between groups
gap = 0.02

fig, axes = plt.subplots(N_BAR_ROWS, 1, figsize=(16, 16))

for row, (ax, dates_row, t_row) in enumerate(zip(axes, date_splits, t_splits)):
    n = len(dates_row)
    if n == 0:
        ax.set_visible(False)
        continue
    x = np.arange(n)

    # Global indices into the full bar_dates list
    offset = row * split

    for i_local in range(n):
        i_global = offset + i_local

        # Always draw 4 fixed slots so every date lines up the same way;
        # a missing data source is shown as a gray "No retrieval" placeholder.
        slots = [
            ("insitu", prop_insitu, "", 0.90, not np.isnan(biovol_total[i_global]), BIOVOL_TAXA),
            ("abund",  prop_abund,  "...", 0.90, not np.isnan(abund_total[i_global]), C_TAXA),
            ("S3A",    prop_A,      "//", 0.75, has_A[i_global], C_TAXA),
            ("S3B",    prop_B,      "xx", 0.75, has_B[i_global], C_TAXA),
        ]

        n_bars  = len(slots)
        offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * (w + gap)

        for (_, props, hatch, alpha, present, taxa), xi in zip(slots, x[i_local] + offsets):
            if present:
                draw_bar(ax, xi, props, i_global, hatch, alpha, taxa=taxa)
            else:
                draw_missing_bar(ax, xi)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.strftime("%d.%m.%Y") for t in t_row],
        rotation=30, ha="right", fontsize=16.5,
    )
    # Half-width of a full date group: distance from the group center to the
    # outer edge of its outermost bar (offset to outermost slot + half its width).
    half_group_width = 1.5 * (w + gap) + w / 2
    ax.set_xlim(x[0] - half_group_width, x[-1] + half_group_width)
    ax.set_ylim(0, 1.0)           # no empty space at top
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Proportion", fontsize=21.5)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="both", which="major", labelsize=16.5)

axes[-1].set_xlabel("Date", fontsize=21.5)

# ── Legend (bottom of the figure, below all rows) ─────────────────────────────
# BIOVOL_TAXA (C_1-C_5 + Chrysophyceae) covers every group that can appear in
# any bar here — the extra groups just never show up in the abund/S3A/S3B
# slots since those props dicts only ever have C_TAXA keys.
taxa_patches = [
    mpatches.Patch(facecolor=TAXA_COLORS[c], label=GROUP_LABELS[c],
                   edgecolor="black", linewidth=0.5)
    for c in BIOVOL_TAXA
]
style_patches = [
    mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.8,
                   label="In-situ (biovolume prop.)"),
    mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.8,
                   hatch="...", label="In-situ (abundance prop.)"),
    mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.8,
                   hatch="//", label="S3A (concentration prop.)"),
    mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.8,
                   hatch="xx", label="S3B (concentration prop.)"),
    mpatches.Patch(facecolor="0.85", edgecolor="black", linewidth=0.8,
                   hatch="\\\\", label="No retrieval"),
]

plt.tight_layout(rect=[0, 0.08, 1, 1])
fig.legend(
    handles=taxa_patches + style_patches,
    fontsize=17.5,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    framealpha=0.9,
    ncol=5,
)

plt.savefig(PLOTS_DIR / "2_community_stacked_bar.png", dpi=300, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# Plot 2b – Stacked-bar phytoplankton community (biovolume reference only)
# ─────────────────────────────────────────────
# Same dates/rows and satellite retrievals as Plot 2, just without the
# abundance slot — easier to read with one less bar per date.
w_2b = 0.20   # a bit wider than Plot 2's bars since there's one fewer slot

fig, axes = plt.subplots(N_BAR_ROWS, 1, figsize=(16, 16))

for row, (ax, dates_row, t_row) in enumerate(zip(axes, date_splits, t_splits)):
    n = len(dates_row)
    if n == 0:
        ax.set_visible(False)
        continue
    x = np.arange(n)

    # Global indices into the full bar_dates list
    offset = row * split

    for i_local in range(n):
        i_global = offset + i_local

        slots = [
            ("insitu", prop_insitu, "", 0.90, not np.isnan(biovol_total[i_global]), BIOVOL_TAXA),
            ("S3A",    prop_A,      "//", 0.75, has_A[i_global], C_TAXA),
            ("S3B",    prop_B,      "xx", 0.75, has_B[i_global], C_TAXA),
        ]

        n_bars  = len(slots)
        offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * (w_2b + gap)

        for (_, props, hatch, alpha, present, taxa), xi in zip(slots, x[i_local] + offsets):
            if present:
                bottom = 0.0
                for c in taxa:
                    val = float(np.nan_to_num(props[c][i_global], nan=0.0))
                    ax.bar(
                        xi, val,
                        bottom=bottom,
                        width=w_2b,
                        color=TAXA_COLORS[c],
                        alpha=alpha,
                        hatch=hatch,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    bottom += val
            else:
                ax.bar(
                    xi, 1.0,
                    width=w_2b,
                    color="0.85",
                    alpha=0.9,
                    hatch="\\\\",
                    edgecolor="black",
                    linewidth=0.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.strftime("%d.%m.%Y") for t in t_row],
        rotation=30, ha="right", fontsize=16.5,
    )
    half_group_width = (n_bars - 1) / 2 * (w_2b + gap) + w_2b / 2
    ax.set_xlim(x[0] - half_group_width, x[-1] + half_group_width)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="both", which="major", labelsize=16.5)

axes[-1].set_xlabel("Date", fontsize=24.5)

# ── Legend (bottom of the figure, below all rows) ─────────────────────────────
taxa_patches_biovol = [
    mpatches.Patch(facecolor=TAXA_COLORS[c], label=GROUP_LABELS[c],
                   edgecolor="black", linewidth=0.5)
    for c in BIOVOL_TAXA
]
style_patches_biovol = [
    mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.8,
                   label="In-situ (biovolume prop.)"),
    mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.8,
                   hatch="//", label="S3A (concentration prop.)"),
    mpatches.Patch(facecolor="white", edgecolor="black", linewidth=0.8,
                   hatch="xx", label="S3B (concentration prop.)"),
    mpatches.Patch(facecolor="0.85", edgecolor="black", linewidth=0.8,
                   hatch="\\\\", label="No retrieval"),
]

plt.tight_layout(rect=[0, 0.12, 1, 1])  # more bottom margin for the legend's extra (3rd) row

# Single shared y-axis label for all rows, centered on the actual axes grid
# (not the whole figure -- that would sit low, skewed by the legend margin)
visible_axes = [a for a in axes if a.get_visible()]
y_top = visible_axes[0].get_position().y1
y_bottom = visible_axes[-1].get_position().y0
# x default is 0.02 (fraction of figure width, ha="left"); since tight_layout
# ran before this label existed, it never reserved room for it, so the
# default sat right on top of the y-tick numbers -- nudge it further left
fig.supylabel("Proportion", fontsize=24.5, x=-0.01, y=(y_top + y_bottom) / 2)

fig.legend(
    handles=taxa_patches_biovol + style_patches_biovol,
    fontsize=20,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    framealpha=0.9,
    ncol=4,  # 10 handles / 4 cols -> 3 rows (was ncol=5 -> 2 rows, too wide at the new fontsize)
)

plt.savefig(PLOTS_DIR / "2b_community_stacked_bar_biovolume_only.png", dpi=300, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# Plot 3 – 1:1 scatter: satellite vs in-situ Chl-a (simple, aggregate)
# ─────────────────────────────────────────────
st_A = compute_stats(sat_A, ins_A)
st_B = compute_stats(sat_B, ins_B)

sat_all = np.concatenate([sat_A, sat_B])
ins_all = np.concatenate([ins_A, ins_B])
st_all  = compute_stats(sat_all, ins_all)

log_stat("CHL_1to1", "CPHY_S3_vs_CHL_S", "S3A", st_A)
log_stat("CHL_1to1", "CPHY_S3_vs_CHL_S", "S3B", st_B)
log_stat("CHL_1to1", "CPHY_S3_vs_CHL_S", "All", st_all)

lims = [0, 15]
xfit = np.array(lims)

fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(ins_A, sat_A, color="tab:green", alpha=0.6, s=40, label=f"S3A N={st_A['n']}")
ax.scatter(ins_B, sat_B, color="tab:green", alpha=0.6, s=40, marker="s", label=f"S3B N={st_B['n']}")
ax.legend(fontsize=11.5, framealpha=0.9)  # was implicit/default-sized; now matches the other legends in this script

ax.plot(lims, lims, "k--", alpha=0.7)

mask = np.isfinite(ins_all) & np.isfinite(sat_all)
m, b = np.polyfit(ins_all[mask], sat_all[mask], 1)
ax.plot(xfit, m * xfit + b, color="k", linewidth=0.8)

ax.text(
    0.97, 0.01,
    f"N={st_all['n']}\n"
    f"r={st_all['r']:.2f}\n"
    f"bias={st_all['bias']:.1f}%\n"
    f"RMSE={st_all['rmse']:.2f}\n"
    f"MdSA={st_all['mdsa']:.1f}%",
    transform=ax.transAxes,
    fontsize=13.5,
    ha="right",
    va="bottom",
)

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("CHL$_S$ [mg m$^{-3}$]", fontsize=14.5)
ax.set_ylabel("CPHY$_{S3}$ [mg m$^{-3}$]", fontsize=14.5)
ax.tick_params(axis="both", labelsize=11.5)
ax.grid(True, alpha=0.3)  # match grid style used everywhere else in this script
plt.tight_layout()
plt.savefig(PLOTS_DIR / "3_scatter_1to1_chla.png", dpi=300, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# Plot 4 – Scatter: model C vs in-situ abundance per group
# ─────────────────────────────────────────────
def get_model_vs_insitu(group_key):
    """
    Returns (model_vals, insitu_abund_vals, insitu_biovol_vals) arrays
    for dates present in both satellites.
    """
    if group_key not in C_TO_ABUNDANCE:
        return None

    model_vals, ab_vals, bv_vals = [], [], []

    for sat in ("S3A", "S3B"):
        store = results[sat]
        model_dates = set(store[group_key])
        abund_dates = set(store["abund_insitu"][group_key])
        common = sorted(model_dates & abund_dates)

        for d in common:
            model_vals.append(store[group_key][d])
            ab_vals.append(store["abund_insitu"][group_key][d])
            bv_vals.append(store["biovol_insitu"][group_key][d])

    return np.array(model_vals), np.array(ab_vals), np.array(bv_vals)

plot_groups = C_TAXA  # C_1 - C_5

ylabel_groups = {"C_1", "C_4"}

n_pg   = len(plot_groups)
n_cols = 3
n_rows = int(np.ceil(n_pg / n_cols))

# Collect stats for the significance summary table further down
all_stats = []  # (group, target_name, stats_dict)

fig, axs = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4 * n_cols, 4 * n_rows)
)

fig.subplots_adjust(hspace=0.45, wspace=0.45)

axs = axs.ravel()

for i, group in enumerate(plot_groups):
    ax = axs[i]

    result = get_model_vs_insitu(group)
    if result is None:
        ax.set_visible(False)
        continue

    model_v, ab_v, _ = result

    mask = np.isfinite(model_v) & np.isfinite(ab_v)
    if mask.sum() < 2:
        ax.set_visible(False)
        continue

    x = model_v[mask]
    y = ab_v[mask]

    stats = compute_stats(x, y)
    all_stats.append((group, "abundance", stats))
    log_stat_corr("phyto_abundance", group, "All", stats)

    ax.scatter(
        x,
        y,
        color=GROUP_COLORS[i],
        s=30,
        alpha=0.6,
    )

    # Regression line (only for C_3 if desired)
    if group == "C_3":
        m, b = np.polyfit(x, y, 1)
        xfit = np.linspace(x.min(), x.max(), 100)

        ax.plot(
            xfit,
            m * xfit + b,
            "k--",
            lw=0.8,
        )

    ax.set_title(
        GROUP_LABELS.get(group, group),
        fontsize=15.5
    )

    ax.set_xlabel(
        f"{GROUP_LABELS.get(group, group).split()[0]} [mg m$^{{-3}}$]",
        fontsize=13.5
    )

    if group in ylabel_groups:
        ax.set_ylabel(
            "Abundance [cells mL$^{-1}$]",
            fontsize=13.5
        )

    sig = "*" if np.isfinite(stats['p']) and stats['p'] < 0.05 else ""
    ax.text(
        0.97,
        0.95,
        f"N={stats['n']}\nr={stats['r']:.2f}{sig}\np={stats['p']:.3f}",
        transform=ax.transAxes,
        fontsize=13.5,
        ha="right",
        va="top"
    )

    ax.grid(True, alpha=0.3)

    ax.tick_params(
        axis="both",
        labelsize=11.5
    )

# Hide unused axes
for ax in axs[len(plot_groups):]:
    ax.set_visible(False)

plt.savefig(PLOTS_DIR / "4a_scatter_model_vs_abundance.png", dpi=300, bbox_inches="tight")
plt.show()

# Same but for biovolume
fig, axs = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4 * n_cols, 4 * n_rows)
)

fig.subplots_adjust(hspace=0.45, wspace=0.45)

axs = axs.ravel()


for i, group in enumerate(plot_groups):
    ax = axs[i]

    result = get_model_vs_insitu(group)
    if result is None:
        ax.set_visible(False)
        continue

    model_v, _, bv_v = result

    mask = np.isfinite(model_v) & np.isfinite(bv_v)
    if mask.sum() < 2:
        ax.set_visible(False)
        continue

    x = model_v[mask]
    y = bv_v[mask]

    stats = compute_stats(x, y)
    all_stats.append((group, "biovolume", stats))
    log_stat_corr("phyto_biovolume", group, "All", stats)

    ax.scatter(
        x,
        y,
        color=GROUP_COLORS[i],
        s=30,
        alpha=0.6,
    )

    # Regression line
    m, b = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min(), x.max(), 100)

    ax.plot(
        xfit,
        m * xfit + b,
        "k--",
        lw=0.8,
    )

    ax.set_title(
        GROUP_LABELS.get(group, group),
        fontsize=15.5
    )

    ax.set_xlabel(
        f"{GROUP_LABELS.get(group, group).split()[0]} [mg m$^{{-3}}$]",
        fontsize=13.5
    )

    if group in ylabel_groups:
        ax.set_ylabel(
            "Biovolume [µm³ mL$^{-1}$]",
            fontsize=13.5
        )

    sig = "*" if np.isfinite(stats['p']) and stats['p'] < 0.05 else ""
    ax.text(
        0.97,
        0.95,
        f"N={stats['n']}\nr={stats['r']:.2f}{sig}\np={stats['p']:.3f}",
        transform=ax.transAxes,
        fontsize=13.5,
        ha="right",
        va="top"
    )

    ax.grid(True, alpha=0.3)

    # Tick label size
    ax.tick_params(
        axis="both",
        labelsize=11.5
    )

# Hide unused axes
for ax in axs[len(plot_groups):]:
    ax.set_visible(False)
plt.savefig(PLOTS_DIR / "4b_scatter_model_vs_biovolume.png", dpi=300, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# Print summary statistics
# ─────────────────────────────────────────────
for tag, st in [("S3A", st_A), ("S3B", st_B), ("Combined", st_all)]:
    print(f"\n{tag}:")
    print(f"  N         = {st['n']}")
    print(f"  Pearson r = {st['r']:.3f}")
    print(f"  p-value   = {st['p']:.4f}")
    print(f"  Bias      = {st['bias']:.1f}%")
    print(f"  RMSE      = {st['rmse']:.3f}")
    print(f"  MdSA      = {st['mdsa']:.1f}%")

# ─────────────────────────────────────────────
# Significance summary: per-group correlations, FDR-corrected (BH)
# ─────────────────────────────────────────────
if all_stats:
    valid = [(g, tgt, s) for g, tgt, s in all_stats if np.isfinite(s['p'])]
    if valid:
        pvals = [s['p'] for _, _, s in valid]
        reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

        print(f"\n{'Group':<6}{'Target':<12}{'N':<5}{'r':<8}{'p_raw':<10}{'p_adj (BH)':<12}{'sig'}")
        for (group, target, s), p_a, rej in zip(valid, p_adj, reject):
            print(f"{group:<6}{target:<12}{s['n']:<5}{s['r']:<8.2f}{s['p']:<10.4f}{p_a:<12.4f}{'*' if rej else ''}")
    else:
        print("\nNo group/target pair had enough data (N>=3) for a p-value.")

# ─────────────────────────────────────────────
# Save quality-metrics summary CSV (CHL + phyto abundance/biovolume)
# ─────────────────────────────────────────────
if _csv_rows:
    df_summary = pd.DataFrame(
        _csv_rows,
        columns=["section", "variable", "satellite", "r", "bias_pct", "RMSE", "MdSA_pct", "p", "N"],
    )
    csv_path = PLOTS_DIR / "res_summary_shl2.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\n── CSV saved → {csv_path}  ({len(df_summary)} rows) ──")
