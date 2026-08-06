"""
processing_campaigns.py
────────────────────────
Campaign station validation: satellite (S3A/S3B OLCI) WASI inversion vs
in-situ Chl-a (CHLA) / TSM / CDOM match-ups at multiple campaign stations
per image.

Adapted from pixel_processor_campaigns.py to:
  - use the current PixelProcessor / MiniWASIsafe model (PixelProcessor.py,
    not the old PixelProcessorV3.py) — the vary/init dicts only cover
    C_0-C_5, C_x, C_y (no more C_6/C_7/C_8/C_mie).
  - save the matchup figure as a .png under PLOTS_DIR (relative to this
    script's location) in addition to showing it interactively.

All paths below are overridable via environment variables so that
main_campaigns.py can control them centrally. Defaults are only used when
this script is run standalone.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from collections import defaultdict
from PixelProcessor import SinglePixelProcessor

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

IMG_DIR = os.environ.get("CAMPAIGNS_BSQ_DIR", r"C:\MSc_thesis_data\satellite\campaigns\bsq")
INSITU_PATH = os.environ.get(
    "CAMPAIGNS_INSITU_CSV",
    "C:\MSc_thesis_data\insitu\campaigns\campaigns_cleaned_all_pigments_remika.csv",
)

# outputs_L3/plots_campaigns, reconstructed relative to this script's location
PLOTS_DIR = Path(os.environ.get("CAMPAIGNS_PLOTS_DIR", str(BASE_DIR / "outputs_L3" / "plots_campaigns")))
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

C_PHY_COMPONENTS = ["C_0", "C_1", "C_2", "C_3", "C_4", "C_5"]

# Current MiniWASIsafe model only knows C_0-C_5, C_x, C_y — vary/init are
# restricted to those keys (no C_6/C_7/C_8/C_mie).
PROCESSOR_KWARGS = dict(
    a_norm_y_from_file=False,
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
# Load in-situ data
# ─────────────────────────────────────────────
insitu_df = pd.read_csv(INSITU_PATH, sep=";", encoding="utf-8")

# ─────────────────────────────────────────────
# Process images
# ─────────────────────────────────────────────
def parse_date(img_name):
    raw = img_name.split("_")[-1][:8]
    return f"{raw[6:8]}.{raw[4:6]}.{raw[:4]}"

def make_result_store():
    return {
        "C_phy":        defaultdict(dict),
        "C_x":          defaultdict(dict),
        "C_y":          defaultdict(dict),
        "C_phy_insitu": defaultdict(dict),
        "C_x_insitu":   defaultdict(dict),
        "C_y_insitu":   defaultdict(dict),
    }

results = {"S3A": make_result_store(), "S3B": make_result_store()}

for img in os.listdir(IMG_DIR):

    if img.startswith("_") or not img.endswith(".bsq"):
        continue

    satellite = "S3A" if "S3A" in img else "S3B" if "S3B" in img else None
    if satellite is None:
        continue

    date_str  = parse_date(img)
    img_path  = os.path.join(IMG_DIR, img)
    df_date   = insitu_df[insitu_df["date"] == date_str]
    stations  = df_date["point"].unique()

    print(f"{'='*80}\n{img}")

    for station in stations:
        if station == "lxp":
            continue

        print(f"\nProcessing station {station}...")
        df_station = df_date[df_date["point"] == station]

        res = SinglePixelProcessor(
            img_path,
            station_name=station,
            **PROCESSOR_KWARGS,
        )

        if not res.inv:
            continue

        # ── in-situ values ──────────────────────────────────────────────
        cdom = np.nan
        tsm  = np.nan
        chla = np.nan

        for var in df_station["var"].unique():
            df_var = df_station[df_station["var"] == var]
            value  = np.nanmean(df_var["value"][df_var["value"] >= 0])
            if var == "CDOM":
                cdom = value
            elif var == "TSM":
                tsm  = value
            elif var == "CHLA":
                chla = value

        # ── store ───────────────────────────────────────────────────────
        p     = res.inv.params
        store = results[satellite]

        store["C_phy"][station][date_str]        = sum(p[c].value for c in C_PHY_COMPONENTS)
        store["C_phy_insitu"][station][date_str] = float(chla)
        store["C_x"][station][date_str]          = p["C_x"].value
        store["C_x_insitu"][station][date_str]   = float(tsm)
        store["C_y"][station][date_str]          = p["C_y"].value
        store["C_y_insitu"][station][date_str]   = float(cdom)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def extract_paired(res_dict, var_key):
    """Collect matched (satellite, insitu, station) arrays across all stations."""
    ins_key = f"{var_key}_insitu"
    sat_vals, ins_vals, station_labels = [], [], []

    for station, sat_store in res_dict[var_key].items():
        ins_store = res_dict[ins_key].get(station, {})
        for date_str, sat_val in sat_store.items():
            if date_str in ins_store:
                sat_vals.append(sat_val)
                ins_vals.append(ins_store[date_str])
                station_labels.append(station)

    return (
        np.array(sat_vals, dtype=float),
        np.array(ins_vals, dtype=float),
        np.array(station_labels),
    )


def compute_stats(x, y):
    """Identical convention to processing_shl2.py: x = satellite, y = in-situ."""
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 2:
        return dict(r=np.nan, bias=np.nan, rmse=np.nan, mdsa=np.nan, n=0)

    x, y = x[mask], y[mask]
    r, _ = pearsonr(x, y)

    pos = (x > 0) & (y > 0)
    if pos.sum() >= 1:
        ratio = y[pos] / x[pos]
        MR    = np.median(np.log10(ratio))
        bias  = (10 ** np.abs(MR) - 1) * np.sign(MR) * 100
        mdsa  = 100 * (np.exp(np.median(np.abs(np.log(ratio)))) - 1)
    else:
        bias = np.nan
        mdsa = np.nan

    return dict(
        r=float(r),
        bias=float(bias),
        rmse=float(np.sqrt(mean_squared_error(y, x))),
        mdsa=float(mdsa),
        n=int(mask.sum()),
    )

# ─────────────────────────────────────────────
# Extract arrays for both variables
# ─────────────────────────────────────────────
sat_A_phy, ins_A_phy, st_A_phy = extract_paired(results["S3A"], "C_phy")
sat_B_phy, ins_B_phy, st_B_phy = extract_paired(results["S3B"], "C_phy")

sat_A_cx,  ins_A_cx,  st_A_cx  = extract_paired(results["S3A"], "C_x")
sat_B_cx,  ins_B_cx,  st_B_cx  = extract_paired(results["S3B"], "C_x")

# ── combined arrays ──────────────────────────────────────────────────────────
sat_all_phy = np.concatenate([sat_A_phy, sat_B_phy])
ins_all_phy = np.concatenate([ins_A_phy, ins_B_phy])
st_all_phy  = np.concatenate([st_A_phy,  st_B_phy])

sat_all_cx  = np.concatenate([sat_A_cx,  sat_B_cx])
ins_all_cx  = np.concatenate([ins_A_cx,  ins_B_cx])
st_all_cx   = np.concatenate([st_A_cx,   st_B_cx])

# ── statistics ───────────────────────────────────────────────────────────────
stats_phy = {
    "S3A":      compute_stats(sat_A_phy, ins_A_phy),
    "S3B":      compute_stats(sat_B_phy, ins_B_phy),
    "Combined": compute_stats(sat_all_phy, ins_all_phy),
}
stats_cx = {
    "S3A":      compute_stats(sat_A_cx,  ins_A_cx),
    "S3B":      compute_stats(sat_B_cx,  ins_B_cx),
    "Combined": compute_stats(sat_all_cx, ins_all_cx),
}

# ─────────────────────────────────────────────
# Plot – two-panel scatter: C_phy and C_x
# ─────────────────────────────────────────────
unique_stations = np.unique(np.concatenate([st_all_phy, st_all_cx]))
cmap            = plt.cm.get_cmap("tab10", len(unique_stations))
station_colors  = {st: cmap(i) for i, st in enumerate(unique_stations)}

def draw_scatter_panel(ax, sat_A, ins_A, sta_A, sat_B, ins_B, sta_B,
                       stats_combined, xlabel, ylabel):
    """Scatter panel matching processing_shl2.py style."""

    sat_all = np.concatenate([sat_A, sat_B])
    ins_all = np.concatenate([ins_A, ins_B])
    sta_all = np.concatenate([sta_A, sta_B])

    # ── data points ─────────────────────────────────────────────────────────
    for st in unique_stations:
        mask = sta_A == st
        if mask.any():
            ax.scatter(ins_A[mask], sat_A[mask],
                       color=station_colors[st], marker="o", s=40, alpha=0.8)

    for st in unique_stations:
        mask = sta_B == st
        if mask.any():
            ax.scatter(ins_B[mask], sat_B[mask],
                       color=station_colors[st], marker="s", s=40, alpha=0.8)

    # ── axis limits ──────────────────────────────────────────────────────────
    fin = np.isfinite(ins_all) & np.isfinite(sat_all)
    lim_max = max(ins_all[fin].max(), sat_all[fin].max()) * 1.05 if fin.any() else 1
    lims = [0, lim_max]

    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # ── 1:1 line ─────────────────────────────────────────────────────────────
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.7)

    # ── regression line ───────────────────────────────────────────────────────
    if fin.sum() >= 2:
        m, b   = np.polyfit(ins_all[fin], sat_all[fin], 1)
        xfit   = np.array(lims)
        ax.plot(xfit, m * xfit + b, color="k", linewidth=0.8)

    # ── stats text box ────────────────────────────────────────────────────────
    s = stats_combined
    ax.text(
        0.97, 0.03,
        f"N={s['n']}\n"
        f"r={s['r']:.2f}\n"
        f"bias={s['bias']:.1f}%\n"
        f"RMSE={s['rmse']:.2f}\n"
        f"MdSA={s['mdsa']:.1f}%",
        transform=ax.transAxes,
        fontsize=11,
        ha="right",
        va="bottom",
    )

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")


fig, axs = plt.subplots(1, 2, figsize=(11, 5))

draw_scatter_panel(
    axs[0],
    sat_A_phy, ins_A_phy, st_A_phy,
    sat_B_phy, ins_B_phy, st_B_phy,
    stats_phy["Combined"],
    xlabel="CPHY$_{HPLC}$ [mg m$^{-3}$]",
    ylabel="CPHY$_{S3}$ [mg m$^{-3}$]",
)

draw_scatter_panel(
    axs[1],
    sat_A_cx,  ins_A_cx,  st_A_cx,
    sat_B_cx,  ins_B_cx,  st_B_cx,
    stats_cx["Combined"],
    xlabel="TSM [g m$^{-3}$]",
    ylabel="NAP$_{S3}$ [g m$^{-3}$]",
)

# ── shared legend ─────────────────────────────────────────────────────────────
station_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=station_colors[st],
           markersize=8, label=st)
    for st in unique_stations
]
satellite_handles = [
    Line2D([0], [0], marker="o", color="grey", linestyle="None",
           markersize=8, label="S3A"),
    Line2D([0], [0], marker="s", color="grey", linestyle="None",
           markersize=8, label="S3B"),
]
fig.legend(
    handles=station_handles + satellite_handles,
    fontsize=12,
    loc="lower center",
    ncol=len(unique_stations) + 2,
    bbox_to_anchor=(0.5, -0.06),
    frameon=False,
)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "campaigns_matchup.png", dpi=150, bbox_inches="tight")
plt.show()

# ─────────────────────────────────────────────
# Print summary statistics
# ─────────────────────────────────────────────
for var_name, stats_dict in [("C_phy / Chl-a", stats_phy), ("C_x / TSM", stats_cx)]:
    print(f"\n{'='*40}\n{var_name}")
    for tag, s in stats_dict.items():
        print(f"\n  {tag}:")
        print(f"    N         = {s['n']}")
        print(f"    Pearson r = {s['r']:.3f}")
        print(f"    Bias      = {s['bias']:.1f}%")
        print(f"    RMSE      = {s['rmse']:.3f}")
        print(f"    MdSA      = {s['mdsa']:.1f}%")
