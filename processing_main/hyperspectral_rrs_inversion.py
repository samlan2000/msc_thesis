import os
import pickle as pkl
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from spectral import envi
from MiniWASIsafe import MiniWasi
from resampling import resample_spectra
from rrs_qa import spectral_roughness, has_nan_spectral_gaps
#import pixel_processor_thetis_chla_v2

# ─────────────────────────────────────────────
# Config
# All paths are overridable via environment variables so that main_thetis.py
# can control them centrally. Defaults below are only used when this
# script is run standalone, and are relative to this repo's layout.
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

IMG_DIR          = os.environ.get("THETIS_BSQ_VALID_DIR", r"C:\MSc_thesis_data\satellite\thetis_valid")
DATE_TO_FILE_MAP = os.environ.get("DATE_TO_FILE_MAP", str(BASE_DIR / "LUTs" / "date_to_file_map.pkl"))

RRS_BAND_SLICE       = slice(100, 430)   # wavelength index range to use
ROUGHNESS_THRESHOLD = 0.002
RRS_MIN             = 0.001
RRS_MAX             = 0.1

RESAMPLE_WAVELENGTHS = np.array([400, 412.5, 442.5, 490, 510, 560, 620, 665, 681.25, 708.75])
RESAMPLE_FWHMS       = np.array([15, 10, 10, 10, 10, 10, 10, 10, 7.5, 10])

C_COMPONENTS     = ["C_0", "C_1", "C_2", "C_3", "C_4", "C_5"]
C_PHY_COMPONENTS = C_COMPONENTS          # all summed → C_phy

INVERSION_VARY    = {
    "C_0": False, "C_1": True, "C_2": True, "C_3": True,
    "C_4": True,  "C_5": True, "C_x": True, "C_y": True,
}
INVERSION_INIT    = {
    "C_0": 0, "C_1": 1, "C_2": 1, "C_3": 1, "C_4": 1,
    "C_5": 1, "C_x": 1, "C_y": 0.1,
}

INVERSION_WEIGHTS =  None #[1]*270 + [0]*30 + [1]*30

PLOT_DIR = os.environ.get(
    "INSITU_RRS_INV_DIAG_DIR",
    str(BASE_DIR / "outputs_intermediate" / "insitu_rrs_inversion_diagnostics"),
)
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Load date→L2 file map
# ─────────────────────────────────────────────
with open(DATE_TO_FILE_MAP, "rb") as f:
    date_to_file_map = pkl.load(f)

count=0
# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def parse_image_datetime(img_name):
    raw_date = img_name.split("_")[-1][:8]
    date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    raw_time = img_name.split(".")[0][-6:-2]
    dt = pd.to_datetime(f"{date_str} {raw_time[:2]}:{raw_time[2:]}", format="%Y-%m-%d %H:%M")
    return date_str, dt

def load_rrs(date_str, datetime_obj):
    """
    Load and validate in-situ Rrs from the L2 xarray dataset closest in time.
    Returns (Rrs_interp, wavelengths) or (None, None) if invalid.
    """
    date_key = pd.to_datetime(date_str).date()
    if date_key not in date_to_file_map:
        return None, None
    ds = xr.open_dataset(date_to_file_map[date_key])

    if "Rrs" not in ds.variables:
        return None, None
    try:
        ds_nearest = ds.sel(time=datetime_obj, method="nearest")
    except KeyError as e:
        print(f"  Time selection error: {e}")
        return None, None

    Rrs      = ds_nearest["Rrs"] / np.pi     # scale BEFORE QC checks
    Rrs      = Rrs[RRS_BAND_SLICE]
    wvls     = ds_nearest.wavelength[RRS_BAND_SLICE]

    if has_nan_spectral_gaps(Rrs, wvls)[0]:
        print("  Skipping: spectral gaps too large.")
        return None, None

    Rrs_interp = Rrs.interpolate_na(dim="wavelength", method="linear", fill_value="extrapolate")

    if (np.all(Rrs_interp < RRS_MIN)
            or spectral_roughness(Rrs_interp, wvls) > ROUGHNESS_THRESHOLD
            or np.any(Rrs_interp > RRS_MAX)):
        return None, None

    return Rrs_interp, wvls

def make_result_store():
    store = {key: {} for key in ["C_phy_insitu", "C_x", "C_y", "rmse", 
                                 "rrs_input", "rrs_fitted", "C_1_insitu", "C_2_insitu", "C_3_insitu", 
                                 "C_4_insitu", "C_5_insitu", "bb_wc", "a_wc", "a_cdom",
                                 "a_phy", "a_nap", "bb_phy", "bb_nap"]}
    store.update({c: {} for c in C_COMPONENTS})
    return store

# ─────────────────────────────────────────────
# Process images
# ─────────────────────────────────────────────
results      = {"S3A": make_result_store(), "S3B": make_result_store()}
valid_images = []

for img in os.listdir(IMG_DIR):

    if img.startswith("_") or not img.endswith(".bsq"):
        continue
    
    #if img not in pixel_processor_thetis_chla_v2.valid_images:
        #continue

    satellite = "S3A" if "S3A" in img else "S3B" if "S3B" in img else None
    if satellite is None:
        continue

    date_str, datetime_obj = parse_image_datetime(img)

    Rrs, wvls = load_rrs(date_str, datetime_obj)
    if Rrs is None:
        print(f"No valid Rrs for {date_str}, skipping.")
        continue

    # Read sza from image header
    hdr_path = os.path.join(IMG_DIR, img.replace(".bsq", ".hdr"))
    sza = float(envi.read_envi_header(hdr_path)["sza"][0])

    print(f"{'='*80}\n{img}")

    model = MiniWasi(sza=sza, wavelengths=wvls, a_norm_y_from_file=False)

    try:
        inv = model.invert(Rrs, vary=INVERSION_VARY, init=INVERSION_INIT, weights=INVERSION_WEIGHTS)
    except ValueError as e:
        print(f"  Inversion error: {e}")
        continue

    if not inv:
        continue

    p     = inv.params
    store = results[satellite]

    for c in C_COMPONENTS:
        store[c][date_str] = p[c].value

    store["C_phy_insitu"][date_str] = sum(p[c].value for c in C_PHY_COMPONENTS)
    store["C_x"][date_str]   = p["C_x"].value
    store["C_y"][date_str]   = p["C_y"].value
    store["C_1_insitu"][date_str]  = p["C_1"].value
    store["C_2_insitu"][date_str]  = p["C_2"].value
    store["C_3_insitu"][date_str]  = p["C_3"].value
    store["C_4_insitu"][date_str]  = p["C_4"].value
    store["C_5_insitu"][date_str]  = p["C_5"].value
    store["rrs_input"][date_str] = np.array(resample_spectra(Rrs, wvls, RESAMPLE_WAVELENGTHS, RESAMPLE_FWHMS))
    store["rrs_fitted"][date_str] = np.array(resample_spectra(model.R_rs, wvls, RESAMPLE_WAVELENGTHS, RESAMPLE_FWHMS))
    store["rmse"][date_str]  = float(np.mean(inv.residual ** 2))
    store["bb_wc"][date_str] = np.array(resample_spectra(model.bb_wc, wvls, RESAMPLE_WAVELENGTHS, RESAMPLE_FWHMS))
    store["bb_nap"][date_str] = np.array(resample_spectra(model.bb_nap, wvls, RESAMPLE_WAVELENGTHS, RESAMPLE_FWHMS))
    store["bb_phy"][date_str] = np.array(resample_spectra(model.bb_phy, wvls, RESAMPLE_WAVELENGTHS, RESAMPLE_FWHMS))
    store["a_wc"][date_str] = np.array(resample_spectra(model.a_wc, wvls, RESAMPLE_WAVELENGTHS, RESAMPLE_FWHMS))
    store["a_nap"][date_str] = np.array(resample_spectra(model.a_nap, wvls, RESAMPLE_WAVELENGTHS, RESAMPLE_FWHMS))
    store["a_phy"][date_str] = np.array(resample_spectra(model.a_phy, wvls, RESAMPLE_WAVELENGTHS, RESAMPLE_FWHMS))
    store["a_cdom"][date_str] = np.array(resample_spectra(model.a_cdom, wvls, RESAMPLE_WAVELENGTHS, RESAMPLE_FWHMS))

    valid_images.append(img)
    print(f"  C_phy_insitu={store['C_phy_insitu'][date_str]:.3f}, C_x={store['C_x'][date_str]:.3f}, C_y={store['C_y'][date_str]:.4f}")
    
    # ---------------------------------------------------------
    # Plot measured and inverted Rrs
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(
        wvls,
        Rrs,
        "k",
        lw=2,
        label="Measured Rrs"
    )
    
    ax.plot(
        wvls,
        model.R_rs,
        "r--",
        lw=2,
        label=(
            "Inverted Rrs\n"
            f"C_phy={sum(p[c].value for c in C_PHY_COMPONENTS):.3f}\n"
            f"C₁={p['C_1'].value:.3f}\n"
            f"C₂={p['C_2'].value:.3f}\n"
            f"C₃={p['C_3'].value:.3f}\n"
            f"C₄={p['C_4'].value:.3f}\n"
            f"C₅={p['C_5'].value:.3f}\n"
            f"C_x={p['C_x'].value:.3f}\n"
            f"C_y={p['C_y'].value:.3f}"
        ),
    )
    
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(r"$R_{rs}$ (sr$^{-1}$)")
    ax.set_title(img)
    
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    outfile = os.path.join(
        PLOT_DIR,
        img.replace(".bsq", ".png")
    )
    plt.savefig(outfile, dpi=200)
    plt.close(fig)
    
    
    
# ─────────────────────────────────────────────
# Export results to CSV
# ─────────────────────────────────────────────
OUT_CSV = os.environ.get(
    "INSITU_RRS_INV_CSV",
    str(BASE_DIR / "outputs_intermediate" / "insitu_rrs_inversion_results.csv"),
)
 
rows = []
for satellite, store in results.items():
    all_dates = sorted(store["C_phy_insitu"])
    for date_str in all_dates:
        rows.append({
            "date":      date_str,
            "satellite": satellite,
            "C_phy":     store["C_phy_insitu"].get(date_str, np.nan),
            "C_1":     store["C_1_insitu"].get(date_str, np.nan),
            "C_2":     store["C_2_insitu"].get(date_str, np.nan),
            "C_3":     store["C_3_insitu"].get(date_str, np.nan),
            "C_4":     store["C_4_insitu"].get(date_str, np.nan),
            "C_5":     store["C_5_insitu"].get(date_str, np.nan),
            "C_x":       store["C_x"].get(date_str, np.nan),   # TSM / NAP
            "C_y":       store["C_y"].get(date_str, np.nan),   # CDOM
            "bb_wc": store["bb_wc"].get(date_str, np.nan),
            "bb_nap": store["bb_nap"].get(date_str, np.nan),
            "bb_phy": store["bb_phy"].get(date_str, np.nan),
            "a_wc": store["a_wc"].get(date_str, np.nan),
            "a_nap": store["a_nap"].get(date_str, np.nan),
            "a_phy": store["a_phy"].get(date_str, np.nan),
            "a_cdom": store["a_cdom"].get(date_str, np.nan),
            "rrs_fitted":       store["rrs_fitted"].get(date_str, np.nan),
            "rrs_input":       store["rrs_input"].get(date_str, np.nan),
        })
 
df_out = pd.DataFrame(rows).sort_values(["date", "satellite"]).reset_index(drop=True)
df_out.to_csv(OUT_CSV, sep=";", encoding="utf-8-sig", index=False)
print(f"Saved {len(df_out)} rows → {OUT_CSV}")
print(df_out.head())