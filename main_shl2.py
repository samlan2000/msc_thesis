"""
main_shl2.py — SHL2 processing chain orchestrator
═══════════════════════════════════════════════════

Links together the three stages of the SHL2 station processing chain:

    0. download_ac         run eawag/sencast (Docker) to download raw
                           satellite products and apply atmospheric
                           correction, once per .ini file in
                           SHL2_SENCAST_PARAMS_DIR
                           (processing_pre/run_sencast.py).
    1. pre_processing     convert SHL2 satellite .nc products to
                           band-restricted .bsq files
                           (processing_pre/bsqConverterPolymer.py).
    2. processing_shl2    satellite WASI inversion vs in-situ Chl-a /
                           phytoplankton community match-ups, plots
                           saved to outputs_L3/plots_shl2
                           (processing_main/processing_shl2.py).

Toggle the RUN_* switches below to re-run only the stage(s) you need.

Path conventions
─────────────────
- Any file that lives inside this repo (MSc_thesis_samuel) is addressed
  relative to this script's location via BASE_DIR.
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
RUN_PRE_PROCESSING = True
RUN_PROCESSING_SHL2 = True


# ═══════════════════════════════════════════════════════════════════════
# EXTERNAL INPUT PATHS — anything NOT inside MSc_thesis_samuel.
# Edit these to match your machine / data location.
# ═══════════════════════════════════════════════════════════════════════

# -- download_ac: sencast (Docker) install + scratch space + parameters --
# SENCAST_DIR is the local sencast checkout (contains docker.ini) and is
# the same across all main_*.py scripts. DIAS/params dirs are per-chain.
SENCAST_DIR = r"C:\Users\samue\sencast"
SHL2_DIAS_TEMP_DIR = r"C:\MSc_thesis_data\satellite\shl2\temp"
SHL2_SENCAST_PARAMS_DIR = r"C:\Users\samue\sencast\parameters\shl2"

# -- pre_processing: SHL2 .nc -> band-restricted .bsq conversion --
# NOTE: the individual per-acquisition product folders (each with L1P/ and
# L2POLY/ subfolders) live under an "output_data" subfolder of the nc dir
SHL2_NC_INPUT_DIR = r"C:\MSc_thesis_data\satellite\shl2\nc\output_data"
SHL2_BSQ_DIR = r"C:\MSc_thesis_data\satellite\shl2\bsq"

# 11 bands, matching the 11-band spectrum (400-753.75nm) that
# processing_shl2.py's PROCESSOR_KWARGS weights vector expects.
SHL2_BSQ_BANDS = ['Oa1', 'Oa2', 'Oa3', 'Oa4', 'Oa5', 'Oa6', 'Oa7', 'Oa8', 'Oa10', 'Oa11', 'Oa12']

# -- processing_shl2: in-situ reference data --
SHL2_MATCHUPS_CSV = r"C:\MSc_thesis_data\insitu\shl2\matchups\matchups_shl2_v3_full_with_phyto.csv"
SHL2_SECCHI_CSV = r"C:\MSc_thesis_data\insitu\shl2\secchi\France_Geneva_secchi_postprocessed.csv"


# ═══════════════════════════════════════════════════════════════════════
# INTERNAL PATHS — inside MSc_thesis_samuel, relative to this script.
# ═══════════════════════════════════════════════════════════════════════
SHL2_PLOTS_DIR = BASE_DIR / "outputs_L3" / "plots_shl2"


# ═══════════════════════════════════════════════════════════════════════
# Stage implementations
# ═══════════════════════════════════════════════════════════════════════
def run_download_ac():
    """Download raw satellite products + atmospheric correction via sencast."""
    print("\n" + "=" * 80)
    print("STAGE 0/2 — download_ac")
    print("=" * 80)

    from run_sencast import run_sencast

    os.makedirs(SHL2_DIAS_TEMP_DIR, exist_ok=True)
    run_sencast(SENCAST_DIR, SHL2_DIAS_TEMP_DIR, SHL2_SENCAST_PARAMS_DIR)


def run_pre_processing():
    """Convert SHL2 .nc products to band-restricted .bsq files."""
    print("\n" + "=" * 80)
    print("STAGE 1/2 — pre_processing")
    print("=" * 80)

    from bsqConverterPolymer import convert_polymer_batch

    os.makedirs(SHL2_BSQ_DIR, exist_ok=True)
    convert_polymer_batch(
        SHL2_NC_INPUT_DIR,
        SHL2_BSQ_DIR,
        bands=SHL2_BSQ_BANDS,
    )


def run_processing_shl2():
    """Satellite WASI inversion vs in-situ Chl-a / phytoplankton community."""
    print("\n" + "=" * 80)
    print("STAGE 2/2 — processing_shl2")
    print("=" * 80)

    os.makedirs(SHL2_PLOTS_DIR, exist_ok=True)
    env = os.environ.copy()
    env["SHL2_BSQ_DIR"] = SHL2_BSQ_DIR
    env["SHL2_MATCHUPS_CSV"] = SHL2_MATCHUPS_CSV
    env["SHL2_SECCHI_CSV"] = SHL2_SECCHI_CSV
    env["SHL2_PLOTS_DIR"] = str(SHL2_PLOTS_DIR)
    _run_script(PROCESSING_MAIN_DIR / "processing_shl2.py", env)


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
    if RUN_PROCESSING_SHL2:
        run_processing_shl2()

    print("\nDone.")
