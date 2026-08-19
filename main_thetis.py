"""
main_thetis.py — Thetis processing chain orchestrator
════════════════════════════════════════════════════════

Links together the five stages of the Thetis processing chain so that the
whole pipeline — or any subset of it — can be (re-)run from one place:

    0. download_ac           run eawag/sencast (Docker) to download raw
                             satellite products and apply atmospheric
                             correction, once per .ini file in
                             THETIS_SENCAST_PARAMS_DIR
                             (processing_pre/run_sencast.py).
    1. pre_processing       convert campaign satellite .nc products to
                             band-restricted .bsq files, then filter the
                             combined Thetis .bsq archive down to images
                             with a valid in-situ match-up.
    2. insitu_inversion     hyperspectral in-situ Rrs -> WASI inversion
                             (processing_main/hyperspectral_rrs_inversion.py).
    3. processing_thetis    per-image satellite WASI inversion + match-up
                             with in-situ/Thetis reference data
                             (processing_main/processing_thetis.py).
    4. plotting_thetis      validation plots + summary CSV from the
                             processing_thetis output
                             (processing_main/plotting_thetis.py).

Toggle the RUN_* switches below to re-run only the stage(s) you need —
each stage reads/writes its inputs and outputs from disk, so earlier
stages don't need to be re-run just to redo a later one.

Path conventions
─────────────────
- Any file that lives inside this repo (MSc_thesis_samuel) is addressed
  relative to this script's location via BASE_DIR, and mirrors the
  existing folder structure (LUTs/, outputs_intermediate/, outputs_L3/).
- Any input that lives outside this repo (raw satellite/in-situ data on
  local disk) is defined as an absolute path in the "EXTERNAL INPUT
  PATHS" section below — edit those to match your machine.
"""

import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PROCESSING_PRE_DIR = BASE_DIR / "processing_pre"
PROCESSING_MAIN_DIR = BASE_DIR / "processing_main"

# make processing_pre importable (used directly, not via subprocess)
sys.path.insert(0, str(PROCESSING_PRE_DIR))


# ═══════════════════════════════════════════════════════════════════════
# EXECUTION SWITCHES — set True/False to (re-)run individual stages
# ═══════════════════════════════════════════════════════════════════════
RUN_DOWNLOAD_AC = False
RUN_PRE_PROCESSING = False
RUN_INSITU_INVERSION = False 
RUN_PROCESSING_THETIS = True
RUN_PLOTTING_THETIS = True


# ═══════════════════════════════════════════════════════════════════════
# EXTERNAL INPUT PATHS — anything NOT inside MSc_thesis_samuel.
# Edit these to match your machine / data location.
# ═══════════════════════════════════════════════════════════════════════

# -- download_ac: sencast (Docker) install + scratch space + parameters --
# SENCAST_DIR is the local sencast checkout (contains docker.ini) and is
# the same across all main_*.py scripts. DIAS/params dirs are per-chain.
SENCAST_DIR = r"C:\Users\samue\sencast"
THETIS_DIAS_TEMP_DIR = r"C:\MSc_thesis_data\satellite\thetis_combined\temp"
THETIS_SENCAST_PARAMS_DIR = r"C:\Users\samue\sencast\parameters\thetis"

# -- pre_processing: campaign .nc -> .bsq conversion (was batch_bsq_conversion.py) --
# NOTE: the raw .nc products themselves live on an external hard drive (several
# TB, not present under C:\MSc_thesis_data) — path kept as-is, pointing at the
# location they'd be mounted/copied to.
CAMPAIGN_NC_INPUT_DIR = r"C:\MSc_thesis_data\satellite\campaigns\nc\output_data"
CAMPAIGN_BSQ_OUTPUT_DIR = r"C:\MSc_thesis_data\satellite\campaigns\bsq_restricted"
CAMPAIGN_BSQ_BANDS = ["Oa3", "Oa4", "Oa5", "Oa6", "Oa8"]

# -- pre_processing: filter the combined Thetis .bsq archive down to valid images --
THETIS_BSQ_COMBINED_DIR = r"C:\MSc_thesis_data\satellite\thetis_combined\bsq"
THETIS_BSQ_VALID_DIR = r"C:\MSc_thesis_data\satellite\thetis_valid"

# -- insitu_inversion / processing_thetis: raw satellite images to process --
# (same folder as THETIS_BSQ_VALID_DIR above — the output of pre_processing)
SAT_IMG_DIR = THETIS_BSQ_VALID_DIR

# -- processing_thetis: in-situ chlorophyll-a reference csv --
INSITU_CHLA_CSV = r"C:\MSc_thesis_data\insitu\thetis\df_thetis_chla_cor.csv"


# ═══════════════════════════════════════════════════════════════════════
# INTERNAL PATHS — inside MSc_thesis_samuel, relative to this script.
# ═══════════════════════════════════════════════════════════════════════
LUT_DIR = BASE_DIR / "LUTs"
DATE_TO_FILE_MAP = LUT_DIR / "date_to_file_map.pkl"
DATE_TO_FILE_MAP_A = LUT_DIR / "date_to_file_map_a.pkl"

OUTPUTS_INTERMEDIATE_DIR = BASE_DIR / "outputs_intermediate"
INSITU_RRS_INV_CSV = OUTPUTS_INTERMEDIATE_DIR / "insitu_rrs_inversion_results.csv"
INSITU_RRS_INV_DIAG_DIR = OUTPUTS_INTERMEDIATE_DIR / "insitu_rrs_inversion_diagnostics"

OUTPUTS_L3_DIR = BASE_DIR / "outputs_L3"
DB_THETIS_PKL = OUTPUTS_L3_DIR / "db_thetis.pkl"
THETIS_PLOTS_DIR = OUTPUTS_L3_DIR / "plots_thetis"


# ═══════════════════════════════════════════════════════════════════════
# Stage implementations
# ═══════════════════════════════════════════════════════════════════════
def run_download_ac():
    """Download raw satellite products + atmospheric correction via sencast."""
    print("\n" + "=" * 80)
    print("STAGE 0/4 — download_ac")
    print("=" * 80)

    from run_sencast import run_sencast

    os.makedirs(THETIS_DIAS_TEMP_DIR, exist_ok=True)
    run_sencast(SENCAST_DIR, THETIS_DIAS_TEMP_DIR, THETIS_SENCAST_PARAMS_DIR)


def run_pre_processing():
    """Convert campaign .nc products to .bsq, then filter the combined
    Thetis .bsq archive down to images with a valid in-situ match-up."""
    print("\n" + "=" * 80)
    print("STAGE 1/4 — pre_processing")
    print("=" * 80)

    # -- 1a. campaign .nc -> band-restricted .bsq (formerly batch_bsq_conversion.py) --
    from bsqConverterPolymer import convert_polymer_batch

    os.makedirs(CAMPAIGN_BSQ_OUTPUT_DIR, exist_ok=True)
    convert_polymer_batch(
        CAMPAIGN_NC_INPUT_DIR,
        CAMPAIGN_BSQ_OUTPUT_DIR,
        bands=CAMPAIGN_BSQ_BANDS,
    )

    # -- 1b. filter combined Thetis .bsq archive down to valid images --
    env = os.environ.copy()
    env["THETIS_BSQ_COMBINED_DIR"] = THETIS_BSQ_COMBINED_DIR
    env["THETIS_BSQ_VALID_DIR"] = THETIS_BSQ_VALID_DIR
    _run_script(PROCESSING_PRE_DIR / "valid_images_thetis.py", env)


def run_insitu_inversion():
    """Hyperspectral in-situ Rrs -> WASI inversion."""
    print("\n" + "=" * 80)
    print("STAGE 2/4 — insitu_inversion")
    print("=" * 80)

    os.makedirs(INSITU_RRS_INV_DIAG_DIR, exist_ok=True)
    env = os.environ.copy()
    env["THETIS_BSQ_VALID_DIR"] = SAT_IMG_DIR
    env["DATE_TO_FILE_MAP"] = str(DATE_TO_FILE_MAP)
    env["INSITU_RRS_INV_DIAG_DIR"] = str(INSITU_RRS_INV_DIAG_DIR)
    env["INSITU_RRS_INV_CSV"] = str(INSITU_RRS_INV_CSV)
    _run_script(PROCESSING_MAIN_DIR / "hyperspectral_rrs_inversion.py", env)


def run_processing_thetis():
    """Per-image satellite WASI inversion + match-up with reference data."""
    print("\n" + "=" * 80)
    print("STAGE 3/4 — processing_thetis")
    print("=" * 80)

    os.makedirs(OUTPUTS_L3_DIR, exist_ok=True)
    env = os.environ.copy()
    env["THETIS_BSQ_VALID_DIR"] = SAT_IMG_DIR
    env["INSITU_CHLA_CSV"] = INSITU_CHLA_CSV
    env["INSITU_RRS_INV_CSV"] = str(INSITU_RRS_INV_CSV)
    env["DATE_TO_FILE_MAP"] = str(DATE_TO_FILE_MAP)
    env["DATE_TO_FILE_MAP_A"] = str(DATE_TO_FILE_MAP_A)
    env["DB_THETIS_PKL"] = str(DB_THETIS_PKL)
    _run_script(PROCESSING_MAIN_DIR / "processing_thetis.py", env)


def run_plotting_thetis():
    """Validation plots + summary CSV from the processing_thetis output."""
    print("\n" + "=" * 80)
    print("STAGE 4/4 — plotting_thetis")
    print("=" * 80)

    os.makedirs(THETIS_PLOTS_DIR, exist_ok=True)
    env = os.environ.copy()
    env["DB_THETIS_PKL"] = str(DB_THETIS_PKL)
    env["THETIS_PLOTS_DIR"] = str(THETIS_PLOTS_DIR)
    _run_script(PROCESSING_MAIN_DIR / "plotting_thetis.py", env)


def _run_script(script_path: Path, env: dict):
    """Run a processing-chain script in its own subprocess, cwd'd to its
    own folder so its bare `import sibling_module` statements resolve."""
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),
        env=env,
        check=True,
    )


# ═══════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if RUN_DOWNLOAD_AC:
        run_download_ac()
    if RUN_PRE_PROCESSING:
        run_pre_processing()
    if RUN_INSITU_INVERSION:
        run_insitu_inversion()
    if RUN_PROCESSING_THETIS:
        run_processing_thetis()
    if RUN_PLOTTING_THETIS:
        run_plotting_thetis()

    print("\nDone.")
