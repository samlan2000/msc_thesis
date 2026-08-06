"""
plotting_thetis.py
───────────────────
Validation plotting for db_thetis_new.pkl (S3A/S3B OLCI vs in-situ WASI
Rrs-inversion match-ups).

Data structure
──────────────
db[sat][date] -> flat dict of scalars / 10-band spectra (WAVELENGTHS_10),
already row-aligned per match-up (no cross-dict date intersection needed).

Sections
────────
  a) Chl-a validation                         (4-panel 1:1, vs CPHY_S3)
  c) Absorption validation                    (a_wc_sat vs a_wc_R, hyperspectral)
  d) Backscattering validation                (discrete bb sensor bands
                                                + bb_wc_sat vs bb_wc_R, hyperspectral)
  e) Rrs validation                           (input & WASI-fitted sat Rrs vs in-situ Rrs)
  f) NAP_R & CDOM_R validation                (C_x/C_y satellite vs NAP_R/CDOM_R)
  g) Combined annual time series
  I) Correlation-only summary grid (12 cells)

Run:  python plotting_thetis.py
"""

import os
import math
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

# ═══════════════════════════════════════════════════════════════════════════
# Config — paths are overridable via environment variables so that main_thetis.py
# can control them centrally. Defaults below are only used when this script
# is run standalone, and are relative to this repo's layout.
# ═══════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent

DB_PATH = os.environ.get("DB_THETIS_PKL", str(BASE_DIR / "outputs_L3" / "db_thetis.pkl"))
OUT_DIR = os.environ.get("THETIS_PLOTS_DIR", str(BASE_DIR / "outputs_L3" / "plots_thetis"))
CSV_PATH = os.path.join(OUT_DIR, "res_summary_thetis.csv")
os.makedirs(OUT_DIR, exist_ok=True)

with open(DB_PATH, "rb") as f:
    db = pkl.load(f)

SATS = ("S3A", "S3B")
MARKERS = {"S3A": "o", "S3B": "s"}

WAVELENGTHS_10 = np.array([400, 412.5, 442.5, 490, 510, 560, 620, 665, 681.25, 708.75])
# indices [2,3,4,5,7] -> 442.5 / 490 / 510 / 560 / 665 nm (weighted / well-constrained bands)
WEIGHTED_IDX = np.array([2, 3, 4, 5, 7])

ALH676_OFFSET = 0.0033
ALH676_SLOPE = 0.0132

SCATTER_ALPHA = 0.6
SCATTER_S = 30
FS_TITLE, FS_AXIS, FS_TICK, FS_STATS, FS_LEGEND = 13, 11, 9, 11, 9

# ═══════════════════════════════════════════════════════════════════════════
# Data capture — one flat, row-aligned table per satellite
# ═══════════════════════════════════════════════════════════════════════════
SCALAR_KEYS = [
    "C_phy_sat", "C_x_sat", "C_y_sat", "C_0_sat", "C_1_sat", "C_2_sat",
    "C_3_sat", "C_4_sat", "C_5_sat",
    "CHL_A", "CHL_F", "aLH676", "bb440", "bb532", "bb630", "bb700",
    "CPHY_R", "NAP_R", "CDOM_R",
]
VECTOR_KEYS = [
    "rrs_input_sat", "rrs_fitted_sat", "a_wc_sat", "a_phy_sat", "a_nap_sat",
    "a_cdom_sat", "bb_wc_sat", "bb_nap_sat", "bb_phy_sat",
    "Rrs", "rrs_input_R", "rrs_fitted_R", "a_wc_R", "a_phy_R", "a_nap_R",
    "a_cdom_R", "bb_wc_R", "bb_nap_R", "bb_phy_R",
    "a",
]
HYP_KEYS = [i for i in SCALAR_KEYS if i.endswith("_R")] + [i for i in VECTOR_KEYS if i.endswith("_R")]
# filter where _R is not nan despite Rrs being nan (Rrs validation with slightly stricter QC)
for sat in db:
    for date, rec in db[sat].items():
        rrs = np.asarray(rec.get('Rrs'), dtype=float)
        fit = np.asarray(rec.get('rrs_fitted_R'), dtype=float)
        mask = np.isnan(rrs) & ~np.isnan(fit)
        if mask.any():
            for k in HYP_KEYS:
                db[sat][date][k] = np.nan


def _safe_float(v):
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return np.nan


def _safe_vec(v, n=10):
    """Return a length-n float array; missing / scalar / short entries -> nan-filled."""
    if v is None or np.isscalar(v):
        return np.full(n, np.nan)
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 0 or arr.shape[0] != n:
        out = np.full(n, np.nan)
        if arr.ndim > 0:
            out[: min(n, arr.shape[0])] = arr[: min(n, arr.shape[0])]
        return out
    return arr


def build_sat_frame(sat):
    dates = sorted(db[sat].keys())
    frame = {"date": pd.to_datetime(dates, format="%Y-%m-%d")}
    for k in SCALAR_KEYS:
        frame[k] = np.array([_safe_float(db[sat][d].get(k)) for d in dates])
    for k in VECTOR_KEYS:
        frame[k] = np.stack([_safe_vec(db[sat][d].get(k)) for d in dates])
    return frame


FRAMES = {sat: build_sat_frame(sat) for sat in SATS}


def concat(key):
    return np.concatenate([FRAMES[s][key] for s in SATS])


def sat_labels_all():
    return np.concatenate([np.full(len(FRAMES[s]["date"]), s) for s in SATS])


# ═══════════════════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════════════════
_csv_rows = []


def compute_stats(x, y):
    """x = satellite / retrieval, y = reference. Returns r, bias%, RMSE, MdSA%, N."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return dict(r=np.nan, bias=np.nan, rmse=np.nan, mdsa=np.nan, n=0)
    x, y = x[mask], y[mask]
    r, _ = pearsonr(x, y)
    pos = (x > 0) & (y > 0)
    if pos.sum() >= 1:
        ratio = x[pos] / y[pos]
        MR = np.median(np.log10(ratio))
        bias = (10 ** np.abs(MR) - 1) * np.sign(MR) * 100
        mdsa = 100 * (np.exp(np.median(np.abs(np.log(ratio)))) - 1)
    else:
        bias, mdsa = np.nan, np.nan
    return dict(r=r, bias=float(bias), rmse=float(np.sqrt(mean_squared_error(y, x))),
                mdsa=float(mdsa), n=int(mask.sum()))


def round_rmse(x):
    if not np.isfinite(x) or x == 0:
        return x
    first_sig = -int(math.floor(math.log10(abs(x))))
    decimals = max(first_sig + 1, 0)
    if x > 1:
        decimals = 2
    return round(x, decimals)


def log_stat(section, variable, satellite, st):
    _csv_rows.append(dict(
        section=section, variable=variable, satellite=satellite, N=st["n"],
        r=round(st["r"], 2) if np.isfinite(st["r"]) else np.nan,
        bias_pct=round(st["bias"], 1) if np.isfinite(st["bias"]) else np.nan,
        RMSE=round_rmse(st["rmse"]) if np.isfinite(st["rmse"]) else np.nan,
        MdSA_pct=round(st["mdsa"], 1) if np.isfinite(st["mdsa"]) else np.nan,
    ))


def print_stats(label, st):
    print(f"  {label:10s}  N={st['n']:4d}  r={st['r']:.3f}  bias={st['bias']:+.1f}%  "
          f"RMSE={st['rmse']:.4f}  MdSA={st['mdsa']:.1f}%")


def fmt(val, f=".2f", threshold=None, suffix=""):
    if not np.isfinite(val):
        return "NaN"
    if threshold is not None and abs(val) > threshold:
        return "NaN"
    return format(val, f) + suffix


def add_stats_text(ax, st, loc="upper left"):
    txt = (f"N={st['n']}\nr={fmt(st['r'])}\nbias={fmt(st['bias'], '.1f', 1e4, '%')}\n"
           f"RMSE={fmt(st['rmse'])}\nMdSA={fmt(st['mdsa'], '.1f', 1e4, '%')}")
    if loc == "upper left":
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=FS_STATS, va="top", ha="left")
    else:
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, fontsize=FS_STATS, va="bottom", ha="right")


def pearson_r_positive(x, y):
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 2:
        return np.nan, int(mask.sum())
    r, _ = pearsonr(x[mask], y[mask])
    return r, int(mask.sum())


def spectral_angle_weighted(sat_spectrum, ref_spectrum, weighted_idx):
    s = np.asarray(sat_spectrum, float)[weighted_idx]
    i = np.asarray(ref_spectrum, float)[weighted_idx]
    mask = np.isfinite(s) & np.isfinite(i) & (s > 0) & (i > 0)
    if mask.sum() < 2:
        return np.nan
    s, i = s[mask], i[mask]
    cos_sim = np.dot(s, i) / (np.linalg.norm(s) * np.linalg.norm(i))
    return np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))


def fig_path(name):
    return os.path.join(OUT_DIR, name)


def build_discrete_band_array(band_values, n_bands=10):
    """
    Assemble a sparse (N, n_bands) array from discrete single-band in-situ
    measurements (e.g. bb440/bb532/bb630/bb700), placing each 1-D array at
    its matching WAVELENGTHS_10 index and leaving all other bands as NaN.
    band_values : dict {band_idx: (N,) array}
    """
    n = len(next(iter(band_values.values())))
    out = np.full((n, n_bands), np.nan)
    for idx, vals in band_values.items():
        out[:, idx] = vals
    return out


# ═══════════════════════════════════════════════════════════════════════════
# a) Chl-a 1:1 — four methods vs satellite CPHY_S3
# ═══════════════════════════════════════════════════════════════════════════
def section_a_chla():
    print("\n── a) Chl-a 1:1 ─────────────────────────────────────────────────────────")

    PANEL_CFG = [
        dict(key="aLH676", convert_alh=True,
             xlabel=r"aLH$_{676}$→CHL [mg m$^{-3}$]", title=r"aLH$_{676}$→CHL vs CPHY$_{S3}$"),
        dict(key="CHL_A", convert_alh=False,
             xlabel=r"CHL$_A$ [mg m$^{-3}$]", title=r"CHL$_A$ vs CPHY$_{S3}$"),
        dict(key="CHL_F", convert_alh=False,
             xlabel=r"CHL$_F$ [mg m$^{-3}$]", title=r"CHL$_F$ vs CPHY$_{S3}$"),
        dict(key="CPHY_R", convert_alh=False, filter_max=50,
             xlabel=r"CPHY$_R$ [mg m$^{-3}$]", title=r"CPHY$_R$ vs CPHY$_{S3}$"),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.45, wspace=0.15)
    axs = axs.flatten()
    lims = [0, 15]

    for idx, (ax, cfg) in enumerate(zip(axs, PANEL_CFG)):
        per_sat = {}
        for sat in SATS:
            ins = FRAMES[sat][cfg["key"]].copy()
            sat_v = FRAMES[sat]["C_phy_sat"].copy()
            if cfg.get("convert_alh"):
                ins = (ins - ALH676_OFFSET) / ALH676_SLOPE
            if cfg.get("filter_max") is not None:
                mask = ins <= cfg["filter_max"]
                ins, sat_v = ins[mask], sat_v[mask]
            per_sat[sat] = (ins, sat_v)

        ins_all = np.concatenate([per_sat[s][0] for s in SATS])
        sat_all = np.concatenate([per_sat[s][1] for s in SATS])
        st_all = compute_stats(sat_all, ins_all)

        print(f"\n  {cfg['title']}")
        for sat in SATS:
            ins, sat_v = per_sat[sat]
            _st = compute_stats(sat_v, ins)
            print_stats(sat, _st)
            log_stat("a_chla_1to1", cfg["title"], sat, _st)
        print_stats("All", st_all)
        log_stat("a_chla_1to1", cfg["title"], "All", st_all)

        for sat in SATS:
            ins, sat_v = per_sat[sat]
            st = compute_stats(sat_v, ins)
            ax.scatter(ins, sat_v, marker=MARKERS[sat], s=30, color="tab:green",
                       alpha=0.6, label=f"{sat}  N={st['n']}")

        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.7)
        mask = np.isfinite(ins_all) & np.isfinite(sat_all)
        if mask.sum() >= 2:
            mfit, bfit = np.polyfit(ins_all[mask], sat_all[mask], 1)
            ax.plot(np.array(lims), mfit * np.array(lims) + bfit, color="k", linewidth=0.8)

        ax.text(0.97, 0.01,
                f"N={st_all['n']:.0f}\nr={st_all['r']:.2f}\nbias={st_all['bias']:.1f}%\n"
                f"RMSE={st_all['rmse']:.2f}\nMdSA={st_all['mdsa']:.1f}%\n",
                transform=ax.transAxes, fontsize=9, ha="right", va="bottom")

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(cfg["xlabel"], fontsize=10)
        ax.set_ylabel(r"CPHY$_{S3}$ [mg m$^{-3}$]" if idx % 2 == 0 else "", fontsize=10)
        ax.set_title(cfg["title"], fontsize=11)
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left", framealpha=1, handlelength=1.2)

    plt.tight_layout()
    plt.savefig(fig_path("a_chla_1to1.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Shared: hyperspectral 1:1 grid (used by c) absorption and d) bb hyperspectral
# ═══════════════════════════════════════════════════════════════════════════
def plot_hyperspectral_1to1(x_full, y_full, band_idx, color_cols, xlabel_unit, ylabel_unit,
                             fname, section_tag, log_prefix, actual_wvls=None):
    """
    x_full, y_full : (N, 10) reference / satellite spectra (row-aligned, both sats concatenated)
    band_idx       : band indices (rows of the grid)
    color_cols     : list of dicts(values=(N,) or (N,10), norm=Normalize, cmap=str, label=str)
    actual_wvls    : optional (len(band_idx),) array of the true wavelengths to use as axis
                      labels, for cases where the nominal WAVELENGTHS_10 grid position doesn't
                      exactly match the real (e.g. in-situ sensor) wavelength.
    """
    sats_arr = sat_labels_all()
    n_rows = len(band_idx)
    n_cols = len(color_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    n_A = int(np.sum((sats_arr == "S3A") & np.isfinite(x_full[:, band_idx[-1]]) & np.isfinite(y_full[:, band_idx[-1]])))
    n_B = int(np.sum((sats_arr == "S3B") & np.isfinite(x_full[:, band_idx[-1]]) & np.isfinite(y_full[:, band_idx[-1]])))

    print(f"\n── {log_prefix} ─────────────────────────────────────────────────")

    for row, band_i in enumerate(band_idx):
        wvl = WAVELENGTHS_10[band_i]
        if actual_wvls is not None and abs(actual_wvls[row] - wvl) > 0.01:
            wvl_label = f"{wvl:g} vs {actual_wvls[row]:g} nm"
        else:
            wvl_label = f"{wvl:g} nm"
        x = x_full[:, band_i]
        y = y_full[:, band_i]

        for col, cc in enumerate(color_cols):
            ax = axs[row, col]
            c_vals = cc["values"]
            c_vals = c_vals[:, band_i] if c_vals.ndim == 2 else c_vals

            for sat in SATS:
                m = sats_arr == sat
                ax.scatter(x[m], y[m], c=c_vals[m], cmap=cc["cmap"], norm=cc["norm"],
                           marker=MARKERS[sat], s=SCATTER_S, alpha=0.7)

            valid = np.concatenate([x, y])
            valid = valid[np.isfinite(valid)]
            lims_lin = [0, np.percentile(valid, 98) * 1.05] if valid.size else [0, 1]
            ax.set_xlim(lims_lin); ax.set_ylim(lims_lin)
            ax.plot(lims_lin, lims_lin, "k--", linewidth=0.8, alpha=0.7)

            mask_all = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if mask_all.sum() >= 2:
                mfit, bfit = np.polyfit(x[mask_all], y[mask_all], 1)
                xfit = np.linspace(lims_lin[0], lims_lin[1], 50)
                ax.plot(xfit, mfit * xfit + bfit, "k-", linewidth=0.8)

            if col == 0:
                st_all = compute_stats(y[mask_all], x[mask_all])
                st_A = compute_stats(y[(sats_arr == "S3A") & mask_all], x[(sats_arr == "S3A") & mask_all])
                st_B = compute_stats(y[(sats_arr == "S3B") & mask_all], x[(sats_arr == "S3B") & mask_all])
                print(f"\n  {wvl_label}")
                print_stats("S3A", st_A); log_stat(section_tag, f"{fname}_{wvl:g}nm", "S3A", st_A)
                print_stats("S3B", st_B); log_stat(section_tag, f"{fname}_{wvl:g}nm", "S3B", st_B)
                print_stats("All", st_all); log_stat(section_tag, f"{fname}_{wvl:g}nm", "All", st_all)
                ax.text(0.03, 0.97,
                        f"N={st_all['n']}\nr={st_all['r']:.2f}\nbias={st_all['bias']:.1f}%\n"
                        f"RMSE={st_all['rmse']:.3f}\nMdSA={st_all['mdsa']:.1f}%",
                        transform=ax.transAxes, fontsize=13, va="top")

            if row == n_rows - 1:
                ax.set_xlabel(xlabel_unit, fontsize=13)
            if col == 0:
                ax.set_ylabel(f"{wvl_label}\n{ylabel_unit}", fontsize=13)
            else:
                ax.tick_params(labelleft=False)
            if row != n_rows - 1:
                ax.tick_params(labelbottom=False)
            ax.tick_params(labelsize=12)
            ax.grid(True, alpha=0.3, which="both")

            if row == 0:
                ax.set_title(cc["label"], fontsize=15)
                cbar = fig.colorbar(plt.cm.ScalarMappable(norm=cc["norm"], cmap=cc["cmap"]),
                                     ax=ax, fraction=0.04, pad=0.02)
                cbar.set_label(cc["label"], fontsize=12)

    plt.tight_layout(h_pad=0.5, w_pad=0.4)
    legend_handles = [
        Line2D([0], [0], marker="o", color="grey", linestyle="none", ms=6, label=f"S3A  N={n_A}"),
        Line2D([0], [0], marker="s", color="grey", linestyle="none", ms=6, label=f"S3B  N={n_B}"),
    ]
    axs[n_rows - 1, -1].legend(handles=legend_handles, fontsize=12, loc="lower right", framealpha=1)
    plt.savefig(fig_path(f"{fname}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return n_A, n_B


def plot_spectral_angle_hist(x_full, y_full, band_idx, title, fname):
    sam = np.array([spectral_angle_weighted(y_full[i], x_full[i], band_idx) for i in range(len(x_full))])
    sam_finite = sam[np.isfinite(sam)]
    med = np.nanmedian(sam_finite) if len(sam_finite) else np.nan
    print(f"  Spectral Angle  median={med:.3f}°   N={len(sam_finite)}")

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, np.nanpercentile(sam_finite, 99) * 1.1, 35) if len(sam_finite) else 20
    ax.hist(sam_finite, bins=bins, color="steelblue", alpha=0.75, edgecolor="white", label=f"N={len(sam_finite)}")
    ax.axvline(med, color="black", linestyle="--", linewidth=1.5, label=f"Median = {med:.2f}°")
    ax.set_xlabel("Spectral Angle [°]", fontsize=FS_AXIS)
    ax.set_ylabel("Count", fontsize=FS_AXIS)
    ax.set_title(title, fontsize=FS_TITLE)
    ax.legend(fontsize=FS_LEGEND)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path(fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return med


def plot_bb_2x2_nocolor(x_full, y_full, band_idx, actual_wvls, xlabel_unit, ylabel_unit, fname):
    """
    Plain (uncolored) 2x2 grid of 1:1 panels for a set of four wavelength comparisons.
    Uses the same fontsizes as the other single-panel ("individual") plots
    (FS_TITLE / FS_AXIS / FS_TICK / FS_STATS / FS_LEGEND).
    """
    sats_arr = sat_labels_all()
    fig, axs = plt.subplots(2, 2, figsize=(9, 8))
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    axs = axs.flatten()

    for i, (band_i, actual_wvl) in enumerate(zip(band_idx, actual_wvls)):
        ax = axs[i]
        nominal_wvl = WAVELENGTHS_10[band_i]
        wvl_label = (f"{nominal_wvl:g} vs {actual_wvl:g} nm" if abs(actual_wvl - nominal_wvl) > 0.01
                     else f"{nominal_wvl:g} nm")
        x = x_full[:, band_i]
        y = y_full[:, band_i]
        mask = np.isfinite(x) & np.isfinite(y)

        for sat in SATS:
            m = (sats_arr == sat) & mask
            ax.scatter(x[m], y[m], marker=MARKERS[sat], s=SCATTER_S, color="grey",
                       alpha=SCATTER_ALPHA, label=f"{sat}  N={m.sum()}")

        valid = np.concatenate([x[mask], y[mask]])
        lims_lin = [0, np.percentile(valid, 98) * 1.05] if valid.size else [0, 1]
        ax.plot(lims_lin, lims_lin, "k--", linewidth=0.8, alpha=0.7)

        mask_pos = mask & (x > 0) & (y > 0)
        if mask_pos.sum() >= 2:
            mfit, bfit = np.polyfit(x[mask_pos], y[mask_pos], 1)
            xfit = np.linspace(lims_lin[0], lims_lin[1], 50)
            ax.plot(xfit, mfit * xfit + bfit, "k-", linewidth=0.8)

        st = compute_stats(y[mask], x[mask])
        add_stats_text(ax, st, loc="upper left")

        ax.set_xlim(lims_lin); ax.set_ylim(lims_lin)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(wvl_label, fontsize=FS_TITLE)
        ax.set_xlabel(xlabel_unit, fontsize=FS_AXIS)
        ax.set_ylabel(ylabel_unit, fontsize=FS_AXIS)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FS_LEGEND, loc="lower right", framealpha=1, handlelength=1.2)

    plt.tight_layout()
    plt.savefig(fig_path(f"{fname}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# c) Absorption validation — a_wc_sat vs a_wc_R (hyperspectral, in-situ retrieval)
# ═══════════════════════════════════════════════════════════════════════════
def section_c_absorption():
    a_R = concat("a_wc_R")
    a_sat = concat("a_wc_sat")
    a_ref = concat("a")  # independent in-situ reference, labeled a_wc
    a_phy_sat = concat("a_phy_sat")
    a_nap_sat = concat("a_nap_sat")
    a_cdom_sat = concat("a_cdom_sat")
    cphy_sat = concat("C_phy_sat")

    log_phy_nap = np.log10(np.maximum(a_phy_sat, 1e-9) / np.maximum(a_nap_sat, 1e-9))
    log_phy_cdom = np.log10(np.maximum(a_phy_sat, 1e-9) / np.maximum(a_cdom_sat, 1e-9))
    cphy_2d = np.tile(cphy_sat[:, None], (1, 10))

    color_cols = [
        dict(values=cphy_2d, norm=mcolors.Normalize(vmin=0, vmax=9), cmap="YlGn",
             label=r"CPHY$_{S3}$ [mg m$^{-3}$]"),
        dict(values=log_phy_nap, norm=mcolors.Normalize(vmin=-2, vmax=2), cmap="GnBu_r",
             label=r"log$_{10}$(a$_{phy}$/a$_{NAP}$)"),
        dict(values=log_phy_cdom, norm=mcolors.Normalize(vmin=-10, vmax=10), cmap="BrBG",
             label=r"log$_{10}$(a$_{phy}$/a$_{CDOM}$)"),
    ]

    # ── c1) satellite vs. in-situ ── (spectral angle lives here)
    plot_hyperspectral_1to1(
        a_ref, a_sat, WEIGHTED_IDX, color_cols,
        xlabel_unit=r"a$_{wc}$ [m$^{-1}$]", ylabel_unit=r"a$_{wc,S3}$ [m$^{-1}$]",
        fname="c1_a_wc_sat_vs_a_wc", section_tag="c1_a_wc_sat_vs_a_wc",
        log_prefix="c) Absorption 1:1 — a$_{wc,S3}$ vs a$_{wc}$ (in-situ)",
    )
    plot_spectral_angle_hist(a_ref, a_sat, WEIGHTED_IDX,
                              "Spectral Angle — Absorption (weighted bands)",
                              "c1_a_wc_sat_vs_a_wc_sam.png")

    # ── c2) hyperspectral (_R) vs. in-situ ──
    plot_hyperspectral_1to1(
        a_ref, a_R, WEIGHTED_IDX, color_cols,
        xlabel_unit=r"a$_{wc}$ [m$^{-1}$]", ylabel_unit=r"a$_{wc,R}$ [m$^{-1}$]",
        fname="c2_a_wc_R_vs_a_wc", section_tag="c2_a_wc_R_vs_a_wc",
        log_prefix="c) Absorption 1:1 — a$_{wc,R}$ vs a$_{wc}$ (in-situ)",
    )

    # ── c3) satellite vs. hyperspectral ──
    plot_hyperspectral_1to1(
        a_R, a_sat, WEIGHTED_IDX, color_cols,
        xlabel_unit=r"a$_{wc,R}$ [m$^{-1}$]", ylabel_unit=r"a$_{wc,S3}$ [m$^{-1}$]",
        fname="c3_a_wc_sat_vs_a_wc_R", section_tag="c3_a_wc_sat_vs_a_wc_R",
        log_prefix="c) Absorption 1:1 — a$_{wc,S3}$ vs a$_{wc,R}$",
    )

    # correlation with CPHY_S3, per band
    print("\n── c) Absorption correlations with CPHY_S3 ────────────────────────────────")
    print(f"  {'Band':>10}  {'r(a_R,Cphy)':>15}  {'r(a_sat,Cphy)':>15}  {'N':>5}")
    for band_i in WEIGHTED_IDX:
        wvl = WAVELENGTHS_10[band_i]
        r_ref, n_ref = pearson_r_positive(a_R[:, band_i], cphy_sat)
        r_sat, _ = pearson_r_positive(a_sat[:, band_i], cphy_sat)
        print(f"  {wvl:>10.2f}  {r_ref:>+15.4f}  {r_sat:>+15.4f}  {n_ref:>5d}")


# ═══════════════════════════════════════════════════════════════════════════
# d) Backscattering validation
#    d1) discrete direct in-situ sensor bands (bb440/532/630/700) vs bb_wc_sat
#    d2) bb_wc_sat vs bb_wc      (satellite vs. in-situ, discrete bands)  — spectral angle here
#    d3) bb_wc_R vs bb_wc        (hyperspectral vs. in-situ, discrete bands)
#    d4) bb_wc_sat vs bb_wc_R    (satellite vs. hyperspectral)
# ═══════════════════════════════════════════════════════════════════════════
def section_d_backscatter():
    print("\n── d) Backscattering bb 1:1 (discrete in-situ sensor bands) ───────────────")

    def get_bb_paired(sat, ins_key, band_i):
        bb_sat = FRAMES[sat]["bb_wc_sat"][:, band_i]
        bb_ins = FRAMES[sat][ins_key]
        bb_phy = FRAMES[sat]["bb_phy_sat"][:, band_i]
        bb_nap = FRAMES[sat]["bb_nap_sat"][:, band_i]
        cphy = FRAMES[sat]["C_phy_sat"]
        cx = FRAMES[sat]["C_x_sat"]
        return bb_ins, bb_sat, bb_phy, bb_nap, cphy, cx

    def plot_bb_discrete(band_i, ins_key, ins_axis_label, sat_label, sat_axis_label):
        per_sat = {sat: get_bb_paired(sat, ins_key, band_i) for sat in SATS}
        bb_ins_all = np.concatenate([per_sat[s][0] for s in SATS])
        bb_sat_all = np.concatenate([per_sat[s][1] for s in SATS])
        cphy_all = np.concatenate([per_sat[s][4] for s in SATS])
        cx_all = np.concatenate([per_sat[s][5] for s in SATS])
        log_nap_phy = {s: np.log10(np.maximum(per_sat[s][3], 1e-9) / np.maximum(per_sat[s][2], 1e-9)) for s in SATS}

        st_all = compute_stats(bb_sat_all, bb_ins_all)
        print(f"\n  {ins_key} / {sat_label}")
        for sat in SATS:
            st = compute_stats(per_sat[sat][1], per_sat[sat][0])
            print_stats(sat, st)
            log_stat("d_bb_discrete_1to1", sat_label, sat, st)
        print_stats("All", st_all)
        log_stat("d_bb_discrete_1to1", sat_label, "All", st_all)

        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(2, 4)
        axs = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])]
        fig.subplots_adjust(hspace=0.25, wspace=0.25)
        lims_bb = [0, 0.06]

        def setup_panel(ax, c_by_sat, cmap, norm, cbar_label, show_stats):
            for sat in SATS:
                ax.scatter(per_sat[sat][0], per_sat[sat][1], marker=MARKERS[sat], s=30,
                           c=c_by_sat[sat], cmap=cmap, norm=norm, alpha=0.6)
            ax.plot(lims_bb, lims_bb, "k--", linewidth=0.8, alpha=0.7)
            mask = np.isfinite(bb_ins_all) & np.isfinite(bb_sat_all)
            if mask.sum() >= 2:
                mfit, bfit = np.polyfit(bb_ins_all[mask], bb_sat_all[mask], 1)
                ax.plot(np.array(lims_bb), mfit * np.array(lims_bb) + bfit, color="k", linewidth=0.8)
            if show_stats:
                legend_handles = [
                    Line2D([0], [0], marker="o", color="grey", linestyle="none", ms=6,
                           label=f"S3A  N={compute_stats(per_sat['S3A'][1], per_sat['S3A'][0])['n']}"),
                    Line2D([0], [0], marker="s", color="grey", linestyle="none", ms=6,
                           label=f"S3B  N={compute_stats(per_sat['S3B'][1], per_sat['S3B'][0])['n']}"),
                ]
                ax.legend(handles=legend_handles, fontsize=10, loc="upper right", framealpha=1, handlelength=1.2)
                ax.text(0.03, 0.97,
                        f"N={st_all['n']}\nr={st_all['r']:.2f}\nbias={st_all['bias']:.0f}%\n"
                        f"RMSE={st_all['rmse']:.3f}\nMdSA={st_all['mdsa']:.0f}%\n",
                        transform=ax.transAxes, fontsize=10, ha="left", va="top")
            ax.set_xlim(lims_bb); ax.set_ylim(lims_bb)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(ins_axis_label, fontsize=11)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(True, alpha=0.3)
            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(cbar_label, fontsize=11)
            cb.ax.tick_params(labelsize=10)

        setup_panel(axs[0], {s: per_sat[s][4] for s in SATS}, "YlGn",
                    mcolors.Normalize(vmin=0, vmax=10), r"CPHY$_{S3}$ [mg m$^{-3}$]", True)
        axs[0].set_ylabel(sat_axis_label, fontsize=10)

        setup_panel(axs[1], log_nap_phy, "GnBu",
                    mcolors.Normalize(vmin=-1, vmax=1), r"log$_{10}$(b$_{b,NAP}$/b$_{b,phy}$)", False)
        axs[1].tick_params(labelleft=False); axs[1].set_ylabel("")

        setup_panel(axs[2], {s: per_sat[s][5] for s in SATS}, "Blues",
                    mcolors.Normalize(vmin=0, vmax=3), r"NAP$_{S3}$ [mg m$^{-3}$]", False)
        axs[2].set_ylabel(sat_axis_label, fontsize=11)

        plt.savefig(fig_path(f"d1_bb_discrete_{ins_key}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # band indices in WAVELENGTHS_10 matched to the direct in-situ sensor bands
    plot_bb_discrete(9, "bb700", r"b$_{b,wc}$(700) [m$^{-1}$]", "bb_wc_708.75nm", r"b$_{b,wc,S3}$(708.75) [m$^{-1}$]")
    plot_bb_discrete(6, "bb630", r"b$_{b,wc}$(630) [m$^{-1}$]", "bb_wc_620nm", r"b$_{b,wc,S3}$(620) [m$^{-1}$]")
    plot_bb_discrete(4, "bb532", r"b$_{b,wc}$(532) [m$^{-1}$]", "bb_wc_510nm", r"b$_{b,wc,S3}$(510) [m$^{-1}$]")
    plot_bb_discrete(2, "bb440", r"b$_{b,wc}$(440) [m$^{-1}$]", "bb_wc_442.5nm", r"b$_{b,wc,S3}$(442.5) [m$^{-1}$]")

    # shared hyperspectral / discrete-reference quantities for d2-d4
    bb_R = concat("bb_wc_R")
    bb_sat = concat("bb_wc_sat")
    bb_phy_sat = concat("bb_phy_sat")
    bb_nap_sat = concat("bb_nap_sat")
    cphy_sat = concat("C_phy_sat")
    cx_sat = concat("C_x_sat")

    log_nap_phy = np.log10(np.maximum(bb_nap_sat, 1e-9) / np.maximum(bb_phy_sat, 1e-9))
    cphy_2d = np.tile(cphy_sat[:, None], (1, 10))
    cx_2d = np.tile(cx_sat[:, None], (1, 10))

    # satellite-quantity coloring, used for the two "satellite vs. ..." panels (d2, d4)
    color_cols_sat = [
        dict(values=cphy_2d, norm=mcolors.Normalize(vmin=0, vmax=9), cmap="YlGn",
             label=r"CPHY$_{S3}$ [mg m$^{-3}$]"),
        dict(values=log_nap_phy, norm=mcolors.Normalize(vmin=-1, vmax=1), cmap="GnBu_r",
             label=r"log$_{10}$(b$_{b,NAP}$/b$_{b,phy}$)"),
        dict(values=cx_2d, norm=mcolors.Normalize(vmin=0, vmax=3), cmap="Blues",
             label=r"NAP$_{S3}$ [mg m$^{-3}$]"),
    ]

    bb_ref = build_discrete_band_array({
        2: concat("bb440"), 4: concat("bb532"), 6: concat("bb630"), 9: concat("bb700"),
    })
    bb_ref_idx = np.array([2, 4, 6, 9])  # 442.5 / 510 / 620 / 708.75 nm (nominal) — only bands with real data
    bb_ref_actual_wvls = np.array([440, 532, 630, 700])  # true in-situ sensor band centers

    nap_R = concat("NAP_R")
    cphy_R = concat("CPHY_R")
    bb_nap_R = concat("bb_nap_R")
    bb_phy_R = concat("bb_phy_R")
    nap_R_2d = np.tile(nap_R[:, None], (1, 10))
    cphy_R_2d = np.tile(cphy_R[:, None], (1, 10))
    log_nap_phy_R = np.log10(np.maximum(bb_nap_R, 1e-9) / np.maximum(bb_phy_R, 1e-9))

    # R-quantity coloring, used for the "hyperspectral (_R) vs. in-situ" panel (d3)
    color_cols_ref = [
        dict(values=nap_R_2d, norm=mcolors.Normalize(vmin=0, vmax=3), cmap="Blues",
             label=r"NAP$_R$ [g m$^{-3}$]"),
        dict(values=cphy_R_2d, norm=mcolors.Normalize(vmin=0, vmax=9), cmap="YlGn",
             label=r"CPHY$_R$ [mg m$^{-3}$]"),
        dict(values=log_nap_phy_R, norm=mcolors.Normalize(vmin=-1, vmax=1), cmap="GnBu_r",
             label=r"log$_{10}$(b$_{b,NAP}$/b$_{b,phy}$)$_R$"),
    ]

    # ── d2) satellite vs. in-situ (discrete bands) ── (spectral angle lives here)
    plot_hyperspectral_1to1(
        bb_ref, bb_sat, bb_ref_idx, color_cols_sat,
        xlabel_unit=r"b$_{b,wc}$ [m$^{-1}$]", ylabel_unit=r"b$_{b,wc,S3}$ [m$^{-1}$]",
        fname="d2_bb_wc_sat_vs_bb_wc", section_tag="d2_bb_wc_sat_vs_bb_wc",
        log_prefix="d) Backscattering 1:1 — b$_{b,wc,S3}$ vs b$_{b,wc}$ (in-situ)",
        actual_wvls=bb_ref_actual_wvls,
    )
    plot_spectral_angle_hist(bb_ref, bb_sat, bb_ref_idx,
                              "Spectral Angle — Backscattering (discrete bands)",
                              "d2_bb_wc_sat_vs_bb_wc_sam.png")

    # ── d2b) same four wavelengths, plain 2x2 grid without any color mapping ──
    plot_bb_2x2_nocolor(
        bb_ref, bb_sat, bb_ref_idx, bb_ref_actual_wvls,
        xlabel_unit=r"b$_{b,wc}$ [m$^{-1}$]", ylabel_unit=r"b$_{b,wc,S3}$ [m$^{-1}$]",
        fname="d2_bb_wc_sat_vs_bb_wc_2x2",
    )

    # ── d3) hyperspectral (_R) vs. in-situ (discrete bands) ──
    plot_hyperspectral_1to1(
        bb_ref, bb_R, bb_ref_idx, color_cols_ref,
        xlabel_unit=r"b$_{b,wc}$ [m$^{-1}$]", ylabel_unit=r"b$_{b,wc,R}$ [m$^{-1}$]",
        fname="d3_bb_wc_R_vs_bb_wc", section_tag="d3_bb_wc_R_vs_bb_wc",
        log_prefix="d) Backscattering 1:1 — b$_{b,wc,R}$ vs b$_{b,wc}$ (in-situ)",
        actual_wvls=bb_ref_actual_wvls,
    )

    # ── d4) satellite vs. hyperspectral ──
    plot_hyperspectral_1to1(
        bb_R, bb_sat, WEIGHTED_IDX, color_cols_sat,
        xlabel_unit=r"b$_{b,wc,R}$ [m$^{-1}$]", ylabel_unit=r"b$_{b,wc,S3}$ [m$^{-1}$]",
        fname="d4_bb_wc_sat_vs_bb_wc_R", section_tag="d4_bb_wc_sat_vs_bb_wc_R",
        log_prefix="d) Backscattering 1:1 — b$_{b,wc,S3}$ vs b$_{b,wc,R}$ (hyperspectral)",
    )


# ═══════════════════════════════════════════════════════════════════════════
# e) Rrs validation
# ═══════════════════════════════════════════════════════════════════════════
BAND_YLIMS_RRS = {
    400.0: 0.025, 412.5: 0.025, 442.5: 0.025, 490.0: 0.025, 510.0: 0.025,
    560.0: 0.025, 620.0: 0.01, 665.0: 0.005, 681.25: 0.005, 708.75: 0.005,
}


def plot_rrs_grid(A, B, use_idx, fname, label, sam_idx=None):
    """A = satellite spectra, B = in-situ reference spectra (both (N,10))."""
    n = len(use_idx)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))

    _sam_idx = sam_idx if sam_idx is not None else use_idx
    sam_vals = np.array([spectral_angle_weighted(A[i], B[i], _sam_idx) for i in range(len(A))])
    sam_finite = sam_vals[np.isfinite(sam_vals)]
    med_sam = np.median(sam_finite) if len(sam_finite) else np.nan

    print(f"\n── e) Rrs 1:1 — {label} ─────────────────────────────────────────────")
    print(f"  Spectral Angle  median={med_sam:.3f}°   N={len(sam_finite)}")

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    axs = np.array(axs).flatten()

    for plot_i, band_i in enumerate(use_idx):
        ax = axs[plot_i]
        wvl = WAVELENGTHS_10[band_i]
        x, y = B[:, band_i], A[:, band_i]
        mask = np.isfinite(x) & np.isfinite(y)
        lims = [0, BAND_YLIMS_RRS.get(float(wvl), max(x[mask].max(), y[mask].max()) * 1.05 if mask.sum() else 1)]

        ax.scatter(x[mask], y[mask], color="grey", alpha=0.6, s=30, label=f"N={mask.sum()}")
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.7)
        if mask.sum() >= 2:
            mfit, bfit = np.polyfit(x[mask], y[mask], 1)
            ax.plot(np.array(lims), mfit * np.array(lims) + bfit, color="k", linewidth=0.8)

        st = compute_stats(y[mask], x[mask])
        print(f"  {wvl} nm   N={st['n']}  r={st['r']:.3f}  bias={st['bias']:+.1f}%  "
              f"RMSE={st['rmse']:.4f}  MdSA={st['mdsa']:.1f}%")
        log_stat(f"e_Rrs_{label.replace(' ', '_')}", f"Rrs_{wvl}nm", "All", st)

        ax.text(0.03, 0.97,
                f"N={st['n']}\nr={st['r']:.2f}\nbias={st['bias']:.1f}%\nRMSE={st['rmse']:.4f}\nMdSA={st['mdsa']:.1f}%\n",
                transform=ax.transAxes, fontsize=13, ha="left", va="top")

        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"\n {wvl} nm", fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        col = plot_i % n_cols
        next_row_has = (plot_i + n_cols) < n
        ax.set_ylabel(r"R$_{rs,S3}$" if col == 0 else "", fontsize=15)
        if not next_row_has:
            ax.set_xlabel("In situ R$_{rs}$", fontsize=15)

    for j in range(n, len(axs)):
        axs[j].set_visible(False)
    plt.tight_layout()
    plt.savefig(fig_path(f"{fname}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # spectra plot
    plot_wvls = WAVELENGTHS_10[use_idx]
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    for i in range(len(B)):
        ax3.plot(plot_wvls, B[i, use_idx], color="steelblue", alpha=0.25, linewidth=0.7)
        ax3.plot(plot_wvls, A[i, use_idx], color="tomato", alpha=0.25, linewidth=0.7)
    med_ins = np.nanmedian(B[:, use_idx], axis=0)
    med_sat = np.nanmedian(A[:, use_idx], axis=0)
    ax3.plot(plot_wvls, med_ins, color="steelblue", linewidth=2.5, label=f"In situ  (N={len(B)})")
    ax3.plot(plot_wvls, med_sat, color="tomato", linewidth=2.5, label=f"Satellite (N={len(A)})")
    ax3.set_xlabel("Wavelength [nm]", fontsize=13)
    ax3.set_ylabel(r"R$_{rs}$ [sr$^{-1}$]", fontsize=13)
    ax3.set_title(f"Rrs spectra — {label}", fontsize=14)
    ax3.legend(fontsize=12, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(fig_path(f"{fname}_spectra.png"), dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # SAM distribution
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, np.percentile(sam_finite, 99) * 1.1, 35) if len(sam_finite) else 20
    ax2.hist(sam_finite, bins=bins, color="steelblue", alpha=0.75, edgecolor="white", label=f"N={len(sam_finite)}")
    ax2.axvline(med_sam, color="black", linestyle="--", linewidth=1.5, label=f"Median = {med_sam:.2f}°")
    ax2.set_xlabel("Spectral Angle [°]", fontsize=FS_AXIS + 3)
    ax2.set_ylabel("Count", fontsize=FS_AXIS + 3)
    ax2.set_title(f"Spectral Angle — Rrs {label}", fontsize=FS_TITLE + 1)
    ax2.legend(fontsize=FS_LEGEND + 3)
    ax2.tick_params(labelsize=FS_TICK + 1)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path(f"{fname}_sam.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)


def section_e_rrs():
    rrs_input_sat = concat("rrs_input_sat")
    rrs_fitted_sat = concat("rrs_fitted_sat")
    rrs_ref = concat("Rrs")  # == rrs_input_R, the in-situ measured Rrs

    plot_rrs_grid(rrs_input_sat, rrs_ref, np.arange(10), "e1_rrs_input_all",
                  "S3 OLCI (input) vs In-situ Rrs — all bands", sam_idx=np.arange(10))
    plot_rrs_grid(rrs_input_sat, rrs_ref, WEIGHTED_IDX, "e1_rrs_input_weighted",
                  "S3 OLCI (input) vs In-situ Rrs — weighted", sam_idx=WEIGHTED_IDX)
    plot_rrs_grid(rrs_fitted_sat, rrs_ref, WEIGHTED_IDX, "e2_rrs_fitted_weighted",
                  "MiniWASI (fitted) vs In-situ Rrs — weighted bands", sam_idx=WEIGHTED_IDX)
    plot_rrs_grid(rrs_fitted_sat, rrs_ref, np.arange(10), "e3_rrs_fitted_all",
                  "MiniWASI (fitted) vs In-situ Rrs — all bands", sam_idx=np.arange(10))


# ═══════════════════════════════════════════════════════════════════════════
# f) NAP_R and CDOM_R validation
# ═══════════════════════════════════════════════════════════════════════════
def section_f_nap_cdom():
    TSM_R_MAX = 8
    print("\n── f) NAP & CDOM 1:1 ────────────────────────────────────────────────────")
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    for ax, sat_key, ref_key, xlabel, ylabel, title, lims_f, color in [
        (axs[0], "C_x_sat", "NAP_R", r"NAP$_R$ [g m$^{-3}$]", r"NAP$_{S3}$ [g m$^{-3}$]",
         "NAP (C$_x$): Satellite vs In-situ Rrs inv.", [0, 4], "blue"),
        (axs[1], "C_y_sat", "CDOM_R", r"CDOM$_R$ [m$^{-1}$]", r"CDOM$_{S3}$ [m$^{-1}$]",
         "CDOM (C$_y$): Satellite vs In-situ Rrs inv.", [0, 0.35], "brown"),
    ]:
        print(f"\n  {sat_key}")
        all_sv, all_iv = [], []
        for sat in SATS:
            sv = FRAMES[sat][sat_key].copy()
            iv = FRAMES[sat][ref_key].copy()
            if ref_key == "NAP_R":
                mask = iv <= TSM_R_MAX
                sv, iv = sv[mask], iv[mask]
            st = compute_stats(sv, iv)
            print_stats(sat, st)
            log_stat("f_CDOM_NAP_1to1", sat_key, sat, st)
            ax.scatter(iv, sv, marker=MARKERS[sat], s=SCATTER_S, color=f"tab:{color}",
                       alpha=SCATTER_ALPHA, label=f"{sat} N={st['n']}")
            all_sv.append(sv); all_iv.append(iv)
        sv_all, iv_all = np.concatenate(all_sv), np.concatenate(all_iv)
        st_all = compute_stats(sv_all, iv_all)
        print_stats("All", st_all)
        log_stat("f_CDOM_NAP_1to1", sat_key, "All", st_all)

        mask = np.isfinite(iv_all) & np.isfinite(sv_all)
        ax.plot(lims_f, lims_f, "k--", linewidth=0.8, alpha=0.7)
        if mask.sum() >= 2:
            mfit, bfit = np.polyfit(iv_all[mask], sv_all[mask], 1)
            ax.plot(np.array(lims_f), mfit * np.array(lims_f) + bfit, color="k", linewidth=0.8)
        add_stats_text(ax, st_all, loc="upper left")
        ax.set_xlim(lims_f); ax.set_ylim(lims_f)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel, fontsize=13); ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc="lower right", framealpha=1)

    plt.tight_layout(w_pad=0.5)
    plt.savefig(fig_path("f_cdom_nap_1to1.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# g) Combined annual time series
# ═══════════════════════════════════════════════════════════════════════════
def section_g_timeseries():
    print("\n── g) Combined annual time series ──────────────────────────────────────")

    all_years_full = sorted(set(pd.concat([pd.Series(FRAMES[s]["date"]) for s in SATS]).dt.year.unique()))
    all_years = [y for y in all_years_full if 2020 <= y <= 2024]
    N_COL = 3
    year_rows = {y: i // N_COL for i, y in enumerate(all_years)}
    year_cols = {y: i % N_COL for i, y in enumerate(all_years)}
    n_row_groups = max(year_rows.values()) + 1
    n_years = len(all_years)
    last_row_n_years = n_years - (n_row_groups - 1) * N_COL

    fig = plt.figure(figsize=(4 * N_COL, 9 * n_row_groups))
    outer = gridspec.GridSpec(n_row_groups, 1, figure=fig, hspace=0.18)
    inner = [gridspec.GridSpecFromSubplotSpec(3, N_COL, subplot_spec=outer[g],
                                               hspace=0.10, wspace=0.06) for g in range(n_row_groups)]

    axes_grid = {}
    for yg in range(n_row_groups):
        for vr in range(3):
            for c in range(N_COL):
                axes_grid[(yg, vr, c)] = fig.add_subplot(inner[yg][vr, c])

    # hide the unused cells in an incomplete last row so no empty subplot is visible
    for vr in range(3):
        for c in range(last_row_n_years, N_COL):
            axes_grid[(n_row_groups - 1, vr, c)].set_visible(False)

    for vr in range(3):
        ref_ax = axes_grid[(0, vr, 0)]
        for yg in range(n_row_groups):
            for c in range(N_COL):
                if not (yg == 0 and c == 0):
                    axes_grid[(yg, vr, c)].sharey(ref_ax)

    def year_vals(sat, key, year):
        d = FRAMES[sat]["date"]  # DatetimeIndex
        v = FRAMES[sat][key]
        m = np.asarray(d.year == year)
        return d[m], v[m]

    for year in all_years:
        yg, col = year_rows[year], year_cols[year]
        ax_top, ax_mid, ax_bot = axes_grid[(yg, 0, col)], axes_grid[(yg, 1, col)], axes_grid[(yg, 2, col)]

        for sat in SATS:
            t_ins, v_ins = year_vals(sat, "CHL_A", year)
            t_sat, v_sat = year_vals(sat, "C_phy_sat", year)
            ax_top.scatter(t_ins, v_ins, marker="x", s=SCATTER_S, color="mediumseagreen",
                           alpha=0.8, zorder=3, linewidths=1.2)
            ax_top.scatter(t_sat, v_sat, marker=MARKERS[sat], s=SCATTER_S, color="green", alpha=0.8, zorder=3)

            t_cx_s, v_cx_s = year_vals(sat, "C_x_sat", year)
            t_cx_i, v_cx_i = year_vals(sat, "NAP_R", year)
            ax_mid.scatter(t_cx_s, v_cx_s, marker=MARKERS[sat], s=SCATTER_S, color="tab:blue", alpha=0.8, zorder=3)
            ax_mid.scatter(t_cx_i, v_cx_i, marker="x", s=SCATTER_S, color="deepskyblue",
                           alpha=0.8, zorder=3, linewidths=1.2)

            t_cy_s, v_cy_s = year_vals(sat, "C_y_sat", year)
            t_cy_i, v_cy_i = year_vals(sat, "CDOM_R", year)
            ax_bot.scatter(t_cy_s, v_cy_s, marker=MARKERS[sat], s=SCATTER_S, color="tab:brown", alpha=0.8, zorder=3)
            ax_bot.scatter(t_cy_i, v_cy_i, marker="x", s=SCATTER_S, color="peru",
                           alpha=0.8, zorder=3, linewidths=1.2)

        for ax in [ax_top, ax_mid, ax_bot]:
            ax.set_xlim(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=12)
            if col > 0:
                ax.tick_params(labelleft=False)

        ax_top.tick_params(labelbottom=False)
        ax_mid.tick_params(labelbottom=False)
        ax_top.set_ylim(0, 15)
        ax_mid.set_ylim(0, 4)
        ax_bot.set_ylim(0, 0.3)
        ax_top.set_title(str(year), fontsize=16)

    for yg in range(n_row_groups):
        axes_grid[(yg, 0, 0)].set_ylabel(r"CHL / CPHY [mg m$^{-3}$]", fontsize=14)
        axes_grid[(yg, 1, 0)].set_ylabel(r"NAP [g m$^{-3}$]", fontsize=14)
        axes_grid[(yg, 2, 0)].set_ylabel(r"CDOM [m$^{-1}$]", fontsize=14)

    leg_top = [
        Line2D([0], [0], marker="o", color="green", ls="none", ms=7, label=r"CPHY$_{S3}$  S3A"),
        Line2D([0], [0], marker="s", color="green", ls="none", ms=7, label=r"CPHY$_{S3}$  S3B"),
        Line2D([0], [0], marker="x", color="mediumseagreen", ls="none", ms=7, markeredgewidth=1.5, label=r"CHL$_A$"),
    ]
    leg_mid = [
        Line2D([0], [0], marker="o", color="tab:blue", ls="none", ms=7, label=r"NAP$_{S3}$  S3A"),
        Line2D([0], [0], marker="s", color="tab:blue", ls="none", ms=7, label=r"NAP$_{S3}$  S3B"),
        Line2D([0], [0], marker="x", color="deepskyblue", ls="none", ms=7, markeredgewidth=1.5, label=r"NAP$_R$"),
    ]
    leg_bot = [
        Line2D([0], [0], marker="o", color="tab:brown", ls="none", ms=7, label=r"CDOM$_{S3}$  S3A"),
        Line2D([0], [0], marker="s", color="tab:brown", ls="none", ms=7, label=r"CDOM$_{S3}$  S3B"),
        Line2D([0], [0], marker="x", color="peru", ls="none", ms=7, markeredgewidth=1.5, label=r"CDOM$_R$"),
    ]
    axes_grid[(0, 0, N_COL - 1)].legend(handles=leg_top, fontsize=12, loc="upper right", framealpha=0.9)
    axes_grid[(0, 1, N_COL - 1)].legend(handles=leg_mid, fontsize=12, loc="upper right", framealpha=0.9)
    axes_grid[(0, 2, N_COL - 1)].legend(handles=leg_bot, fontsize=12, loc="upper right", framealpha=0.9)

    plt.savefig(fig_path("g_annual_timeseries.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# I) Correlation-only summary grid (12 cells)
# ═══════════════════════════════════════════════════════════════════════════
def section_I_correlation_grid():
    print("\n── I) Correlation-only summary grid ────────────────────────────────────")

    ins_chla = concat("CHL_A")
    ins_nap = concat("NAP_R")
    ins_cdom = concat("CDOM_R")
    sat_cphy = concat("C_phy_sat")
    sat_nap = concat("C_x_sat")
    sat_cdom = concat("C_y_sat")

    LIM_CHLA, LIM_NAP, LIM_CDOM = [0, 15], [0, 4], [0, 0.3]

    PAIRS_INS = [
        (ins_nap, ins_chla, "NAP$_R$", "CHL$_A$", LIM_NAP, LIM_CHLA),
        (ins_cdom, ins_chla, "CDOM$_R$", "CHL$_A$", LIM_CDOM, LIM_CHLA),
        (ins_cdom, ins_nap, "CDOM$_R$", "NAP$_R$", LIM_CDOM, LIM_NAP),
    ]
    PAIRS_SAT = [
        (sat_nap, sat_cphy, "NAP$_{S3}$", "CPHY$_{S3}$", LIM_NAP, LIM_CHLA),
        (sat_cdom, sat_cphy, "CDOM$_{S3}$", "CPHY$_{S3}$", LIM_CDOM, LIM_CHLA),
        (sat_cdom, sat_nap, "CDOM$_{S3}$", "NAP$_{S3}$", LIM_CDOM, LIM_NAP),
    ]
    PAIRS_CROSS_1 = [
        (ins_chla, sat_nap, "CHL$_A$", "NAP$_{S3}$", LIM_CHLA, LIM_NAP),
        (ins_chla, sat_cdom, "CHL$_A$", "CDOM$_{S3}$", LIM_CHLA, LIM_CDOM),
        (ins_nap, sat_cdom, "NAP$_R$", "CDOM$_{S3}$", LIM_NAP, LIM_CDOM),
    ]
    PAIRS_CROSS_2 = [
        (ins_nap, sat_cphy, "NAP$_R$", "CPHY$_{S3}$", LIM_NAP, LIM_CHLA),
        (ins_cdom, sat_cphy, "CDOM$_R$", "CPHY$_{S3}$", LIM_CDOM, LIM_CHLA),
        (ins_cdom, sat_nap, "CDOM$_R$", "NAP$_{S3}$", LIM_CDOM, LIM_NAP),
    ]

    ROWS = [
        (r"$In\ situ$ vs $In\ situ$", PAIRS_INS),
        ("Satellite vs. Satellite", PAIRS_SAT),
        (r"$In\ situ$ vs Satellite (1)", PAIRS_CROSS_1),
        (r"$In\ situ$ vs Satellite (2)", PAIRS_CROSS_2),
    ]

    def _short(label):
        return label.split("[")[0].strip()

    fig, ax = plt.subplots(figsize=(10, 11))
    n_rows, n_cols = len(ROWS), 3
    cmap, norm = plt.cm.RdBu, plt.Normalize(vmin=-1, vmax=1)

    for row, (row_label, pairs) in enumerate(ROWS):
        for col, (xd, yd, xl, yl, xlim, ylim) in enumerate(pairs):
            mask = (np.isfinite(xd) & np.isfinite(yd) &
                    (xd >= xlim[0]) & (xd <= xlim[1]) & (yd >= ylim[0]) & (yd <= ylim[1]))
            r = pearsonr(xd[mask], yd[mask])[0] if mask.sum() >= 2 else np.nan

            y_pos = n_rows - 1 - row
            color = cmap(norm(r)) if np.isfinite(r) else (0.85, 0.85, 0.85, 1.0)
            ax.add_patch(plt.Rectangle((col, y_pos), 1, 1, facecolor=color, edgecolor="white", linewidth=2))
            txt_color = "white" if (np.isfinite(r) and abs(r) > 0.55) else "black"
            r_txt = f"r = {r:+.2f}" if np.isfinite(r) else "r = n/a"
            ax.text(col + 0.5, y_pos + 0.62, f"{_short(xl)}\nvs\n{_short(yl)}",
                    ha="center", va="center", fontsize=15, color=txt_color)
            ax.text(col + 0.5, y_pos + 0.28, r_txt, ha="center", va="center", fontsize=15, color=txt_color)

            log_stat("I_correlation_grid", f"{_short(xl)}_vs_{_short(yl)}", "All",
                      dict(n=int(mask.sum()), r=r, bias=np.nan, rmse=np.nan, mdsa=np.nan))

    ax.set_xlim(0, n_cols); ax.set_ylim(0, n_rows)
    ax.set_xticks([])
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([r[0] for r in ROWS][::-1], fontsize=15, rotation=30, ha="right")
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Pearson r", fontsize=18)
    cbar.ax.tick_params(labelsize=13)

    plt.tight_layout()
    plt.savefig(fig_path("I_correlation_summary_grid.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    section_a_chla()
    section_c_absorption()
    section_d_backscatter()
    section_e_rrs()
    section_f_nap_cdom()
    section_g_timeseries()
    section_I_correlation_grid()

    df_summary = pd.DataFrame(_csv_rows, columns=["section", "variable", "satellite", "N", "r", "bias_pct", "RMSE", "MdSA_pct"])
    df_summary.to_csv(CSV_PATH, index=False)
    print(f"\n── CSV saved → {CSV_PATH}  ({len(df_summary)} rows) ──")
    print(f"── Figures saved → {OUT_DIR}/ ──")