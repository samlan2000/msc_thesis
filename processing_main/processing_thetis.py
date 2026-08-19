import os
import pickle as pkl
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import savgol_filter
from PixelProcessor import SinglePixelProcessor
from resampling import resample_spectra
from rrs_qa import spectral_roughness, has_nan_spectral_gaps

# ═══════════════════════════════════════════════════════════════════
# config
# All paths are overridable via environment variables so that main_thetis.py
# can control them centrally. Defaults below are only used when this
# script is run standalone, and are relative to this repo's layout.
# ═══════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent

IMG_DIR_ALL = os.environ.get("THETIS_BSQ_VALID_DIR", r"C:\MSc_thesis_data\satellite\thetis_valid")

MAX_DEPTH = 5

# Rrs config
ROUGHNESS_THRESHOLD_RRS = 0.002
RRS_BAND_SLICE      = slice(100, 430)
RRS_MIN, RRS_MAX    = 0.001, 0.1
RESAMPLE_WAVELENGTHS = np.array([400, 412.5, 442.5, 490, 510, 560, 620, 665, 681.25, 708.75])
RESAMPLE_FWHMS       = np.array([15, 10, 10, 10, 10, 10, 10, 10, 7.5, 10])

# absorption config
SAVGOL_WINDOW, SAVGOL_POLYORDER = 5, 2

INSITU_PATH_CSV = os.environ.get(
    "INSITU_CHLA_CSV",
    r"C:\MSc_thesis_data\insitu\thetis\df_thetis_chla_cor.csv",
)
insitu_df = pd.read_csv(INSITU_PATH_CSV, sep=";", encoding="utf-8")
insitu_df = insitu_df[insitu_df["depth"] <= MAX_DEPTH][
    ["depth", "datetime", "chla", "aLH676", "date", "chla_corr"]
].copy()
insitu_df["datetime"] = pd.to_datetime(insitu_df["datetime"], format="%Y-%m-%d %H:%M:%S.%f")

INSITU_PATH_HYP = os.environ.get(
    "INSITU_RRS_INV_CSV",
    str(BASE_DIR / "outputs_intermediate" / "insitu_rrs_inversion_results.csv"),
)
insitu_df_hyp = pd.read_csv(INSITU_PATH_HYP, sep=";", encoding="utf-8-sig")
# FIX: was `insitu_df["date"]` — assigned the *other* dataframe's date column onto this one.
insitu_df_hyp["date"] = insitu_df_hyp["date"].astype(str)
# these are formatted as strings - convert to numpy arrays
array_cols = [
    "bb_wc", "bb_nap", "bb_phy",
    "a_wc", "a_nap", "a_phy", "a_cdom",
    "rrs_fitted", "rrs_input"
]
for col in array_cols:
    insitu_df_hyp[col] = insitu_df_hyp[col].apply(
        lambda x: np.fromstring(x.strip("[]"), sep=" ")
        if isinstance(x, str) else np.nan
    )


results = dict()
results["S3A"] = dict()
results["S3B"] = dict()


DATE_TO_FILE_MAP = os.environ.get("DATE_TO_FILE_MAP", str(BASE_DIR / "LUTs" / "date_to_file_map.pkl"))
DATE_TO_FILE_MAP_A = os.environ.get("DATE_TO_FILE_MAP_A", str(BASE_DIR / "LUTs" / "date_to_file_map_a.pkl"))
with open(DATE_TO_FILE_MAP, "rb") as f:
    date_to_file_map = pkl.load(f)
with open(DATE_TO_FILE_MAP_A, "rb") as f:
    date_to_file_map_absorption = pkl.load(f)


OUT_PATH = os.environ.get("DB_THETIS_PKL", str(BASE_DIR / "outputs_L3" / "db_thetis.pkl"))

PROCESSOR_KWARGS = dict(
    a_norm_y_from_file=False,
    station_name="lxp",
    weights=[0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    i_offset=1,
    j_offset=0,
    valid_pixel_min=2000,
    vary={
        "C_0": False, "C_1": True, "C_2": True,  "C_3": True,
        "C_4": True,  "C_5": True, "C_6": False,  "C_7": False, "C_8": False,
        "C_x": True,  "C_mie": False, "C_y": True,
    },
    init={
        "C_0": 0, "C_1": 1, "C_2": 1, "C_3": 1, "C_4": 1,
        "C_5": 1, "C_6": 0, "C_7": 0, "C_8": 0, "C_mie": 0, "C_x": 1, "C_y": 0.1,
    },
)
# NOTE: your original chla_corr script used a different weight vector
# ([0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0], band index 2 unweighted) for its own inversion.
# Since this script runs one shared inversion per image, CHL_A below is now computed
# with the standard weights instead — flagging in case that distinction mattered.

C_COMPONENTS = ["C_0", "C_1", "C_2", "C_3", "C_4", "C_5"]

VARS_VALID_RANGES = {"CHL_A": [0.001, 50], "CHL_F": [0.001, 50], "aLH676": [0.001, 0.5], 
                     "bb630": [0.0001, 0.1], "bb700": [0.0001, 0.1], "bb532": [0.0001, 0.1], 
                     "bb440": [0.0001, 0.1], "a": [0.0001, 1], "Rrs": [0.001, 0.08]}

def in_valid_range(val, var):
    # FIX: original was `if val < mx or val > mi: return True`, which is satisfied by
    # almost any real number (since mi < mx always) — it never actually rejected anything
    # except NaN. This now does the intended min <= val <= max check.
    mi, mx = VARS_VALID_RANGES[var]
    return mi <= val <= mx

def parse_image_datetime(img_name):
    """Return (date_str 'YYYY-MM-DD', datetime_obj) parsed from an image filename."""
    raw_date = img_name.split("_")[-1][:8]
    date_str = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
    raw_time = img_name.split(".")[0][-6:-2]
    dt = pd.to_datetime(f"{date_str} {raw_time[:2]}:{raw_time[2:]}", format="%Y-%m-%d %H:%M")
    return date_str, dt


def get_satellite(img_name):
    if "S3A" in img_name:
        return "S3A"
    if "S3B" in img_name:
        return "S3B"
    return None


def list_images(img_dir):
    """Yield (img_filename, satellite) for every valid .bsq image in img_dir."""
    for img in sorted(os.listdir(img_dir)):
        if img.startswith("_") or not img.endswith(".bsq"):
            continue
        sat = get_satellite(img)
        if sat is None:
            continue
        yield img, sat
        
        
def closest_insitu(df_date, datetime_obj):
    dts = pd.Series(df_date["datetime"].unique())
    closest_dt = dts.iloc[(dts - datetime_obj).abs().argmin()]
    return df_date[df_date["datetime"] == closest_dt]


def get_insitu(date_str, satellite, col):
    row = insitu_df_hyp[
        (insitu_df_hyp["date"] == date_str) &
        (insitu_df_hyp["satellite"] == satellite)
    ]
    if row.empty or col not in row.columns:
        return None
    val = row[col].iloc[0]
    # numpy array
    if isinstance(val, np.ndarray):
        return val if np.any(np.isfinite(val)) else None
    # scalar
    return float(val) if np.isfinite(val) else None
        
        
for img, sat in list_images(IMG_DIR_ALL):

    date_str, dt = parse_image_datetime(img)
    
    ################################################################################
    # Satellite processing
    res = SinglePixelProcessor(os.path.join(IMG_DIR_ALL, img), **PROCESSOR_KWARGS)
    if not res.inv:
        continue
    
    print("")
    print(img)

    p, store = res.inv.params, results[sat]
    store[date_str] = dict()
    for c in C_COMPONENTS:
        store[date_str][f"{c}_sat"] = p[c].value
    store[date_str]["C_phy_sat"] = sum(p[c].value for c in C_COMPONENTS)
    store[date_str]["C_x_sat"] = p["C_x"].value
    store[date_str]["C_y_sat"] = p["C_y"].value
    store[date_str]["rrs_fitted_sat"] = np.array(res.wasi.R_rs[:-1])
    store[date_str]["rrs_input_sat"] = np.array(res.spectrum.flatten()[:-1])
    store[date_str]["a_wc_sat"] = np.array(res.wasi.a_wc[:-1])
    store[date_str]["a_phy_sat"]  = np.array(res.wasi.a_phy[:-1])
    store[date_str]["a_nap_sat"]  = np.array(res.wasi.a_nap[:-1])
    store[date_str]["a_cdom_sat"] = np.array(res.wasi.a_cdom[:-1])
    # NOTE: your reference bb_V3 script used `res.wasi.bb`, not `bb_wc` — double-check
    # this attribute actually exists on your wasi object, or this will AttributeError.
    store[date_str]["bb_wc_sat"] = np.array(res.wasi.bb_wc[:-1])
    store[date_str]["bb_nap_sat"] = np.array(res.wasi.bb_nap[:-1])
    store[date_str]["bb_phy_sat"] = np.array(res.wasi.bb_phy[:-1])

    ################################################################################
    # CHL_A - a little more difficult as in .csv
    df_date = insitu_df[insitu_df["date"] == date_str]
    # FIX: previously fell through to `closest_insitu(df_date, dt)` even when df_date was
    # empty, which crashes with "attempt to get argmin of an empty sequence". Also, the
    # valid-range check below used to be immediately overwritten by an unconditional
    # assignment right after it, so it never actually took effect.
    if df_date.empty:
        store[date_str]["CHL_A"] = np.nan
    else:
        df_closest = closest_insitu(df_date, dt)
        chla_vals = df_closest["chla_corr"]
        # note that already only <= 5m values
        avg_chla = chla_vals.mean()
        store[date_str]["CHL_A"] = avg_chla if in_valid_range(avg_chla, "CHL_A") else np.nan
    
    ################################################################################
    #Thetis, variables: CHL_F, aLH676, bb, a, Rrs
    thetis_missing = False
    varname_LUT = {"CHL_F": "chla", "aLH676": "aLH676", "bb630": "bb630", "bb700": "bb700",
                   "bb532": "bb532", "bb440": "bb440", "a": "a", "Rrs": "Rrs"}
    vars_2d = list(k for k in varname_LUT.keys() if k not in ["a", "Rrs"])
    
    date_key = pd.to_datetime(date_str).date()
    if date_key not in date_to_file_map:
        print(f"Warning: No Thetis file found despite matched CHL_A for image [{img}]")
        thetis_missing = True
        for var in varname_LUT.keys():
            store[date_str][var] = np.nan

    if not thetis_missing:
        ds = xr.open_dataset(date_to_file_map[date_key])
        # select first 5m
        ds = ds.sel(depth=slice(0, MAX_DEPTH))
        # select nearest time coord to image acquisition
        # FIX: wrapped in try/except — an unmatched "nearest" selection (e.g. empty/degenerate
        # time index) previously raised an uncaught KeyError and killed the whole run.
        try:
            ds = ds.sel(time=dt, method="nearest")
        
            selected_time = pd.Timestamp(ds.time.values)
            if selected_time.date() != dt.date():
                raise KeyError("Nearest profile is from a different day.")
        
        except KeyError:
            print(f"Warning: No same-day Thetis profile for image [{img}]")
            thetis_missing = True
            for var in varname_LUT:
                store[date_str][var] = np.nan
        
        if not thetis_missing:
            for k, v in varname_LUT.items():
                if v not in ds.variables: 
                    store[date_str][k] = np.nan
                
    if not thetis_missing:
        # process simple 2D variables - only simple range check
        for var in vars_2d:
            ncname = varname_LUT[var]
            if ncname not in ds.variables:
                continue
            avg = np.nanmean(ds[ncname].values)
            store[date_str][var] = avg if in_valid_range(avg, var) else np.nan
            
            
        # process Rrs
        if "Rrs" not in ds.variables:
            store[date_str]["Rrs"] = np.nan
        else:
            Rrs  = ds["Rrs"] / np.pi          # scale BEFORE QC checks
            Rrs  = Rrs[RRS_BAND_SLICE]
            wvls = ds.wavelength[RRS_BAND_SLICE]
    
            if has_nan_spectral_gaps(Rrs, wvls)[0]:
                store[date_str]["Rrs"] = np.nan
            else:
                Rrs_interp = Rrs.interpolate_na(dim="wavelength", method="linear", fill_value="extrapolate")
    
                if (np.all(Rrs_interp < RRS_MIN)
                        or spectral_roughness(Rrs_interp, wvls) > ROUGHNESS_THRESHOLD_RRS
                        or np.any(Rrs_interp > RRS_MAX)):
                    store[date_str]["Rrs"] = np.nan
                else:
                    store[date_str]["Rrs"] = np.array(
                        resample_spectra(Rrs_interp, wvls, RESAMPLE_WAVELENGTHS, RESAMPLE_FWHMS)
                    )
        

    # process absorption - load re-processed hyperspectral a dataset instead of Thetis L2
    # processing failed for some absorption measurements - check needed
    thetis_a_missing = False
    try:
        ds_a = xr.open_dataset(date_to_file_map_absorption[date_key])
        ds_a = ds_a.sel(depth=slice(0, MAX_DEPTH))
        ds_a = ds_a.sel(time=dt, method="nearest")

        if pd.Timestamp(ds_a.time.values).date() != dt.date():
            raise KeyError

    except KeyError:
        thetis_a_missing = True

    if thetis_a_missing:
        store[date_str]["a"] = np.nan

    elif "a" not in ds_a.variables:
        store[date_str]["a"] = np.nan

    else:
        a_vals     = ds_a.a.values         # (depth, wavelength)
        wvl_native = ds_a.wavelength.values

        # Robust central profile (median across all available depths)
        a_avg_vals = np.nanmedian(a_vals, axis=0)
        a_avg = xr.DataArray(
            a_avg_vals,
            dims="wavelength",
            coords={"wavelength": wvl_native},
        )

        if spectral_roughness(a_avg, wvl_native) > 0.001:
            store[date_str]["a"] = np.nan

        elif has_nan_spectral_gaps(a_avg, wvl_native, max_gap_nm=10)[0]:
            store[date_str]["a"] = np.nan

        elif (
            np.any(a_avg.values < 0)
            or np.any(a_avg.values > 1)
            or np.all(a_avg.values == 0)
        ):
            store[date_str]["a"] = np.nan

        else:
            a_interp = a_avg.interpolate_na(
                dim="wavelength",
                method="linear"
            ).values

            # Mild spectral smoothing on the native grid, before sensor resampling
            a_final = a_interp
            if len(a_interp) >= SAVGOL_WINDOW + 2:
                a_final = savgol_filter(
                    a_interp,
                    window_length=SAVGOL_WINDOW,
                    polyorder=SAVGOL_POLYORDER,
                    mode="interp",
                )

            store[date_str]["a"] = resample_spectra(
                a_final,
                wvl_native,
                res.wasi.wavelengths,
                res.FWHMs,
            )[:-1]
    
    ################################################################################
    # hyperspectral in situ wasi retrievals
    ins_cphy = get_insitu(date_str, sat, "C_phy")
    if ins_cphy is None:
        store[date_str]["CPHY_R"] = np.nan
    else:
        store[date_str]["CPHY_R"] = ins_cphy
        
    ins_cx   = get_insitu(date_str, sat, "C_x")
    if ins_cx is None:
        store[date_str]["NAP_R"] = np.nan
    else:
        store[date_str]["NAP_R"] = ins_cx
        
    ins_cy   = get_insitu(date_str, sat, "C_y")
    if ins_cy is None:
        store[date_str]["CDOM_R"] = np.nan
    else:
        store[date_str]["CDOM_R"] = ins_cy
        
    for col in array_cols:
        ins = get_insitu(date_str, sat, col)
        if ins is None:
            store[date_str][f"{col}_R"] = np.nan
        else:
            store[date_str][f"{col}_R"] = ins


# ═══════════════════════════════════════════════════════════════════
# save
# ═══════════════════════════════════════════════════════════════════
with open(OUT_PATH, "wb") as f:
    pkl.dump(results, f)
print(f"\nSaved → {OUT_PATH}")