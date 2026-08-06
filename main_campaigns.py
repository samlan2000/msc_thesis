"""
main_campaigns.py — Campaigns processing chain orchestrator
═════════════════════════════════════════════════════════════

Links together the three stages of the campaigns station processing chain:

    1. pre_processing         convert campaign .nc satellite products to
                               band-restricted .bsq files
                               (processing_pre/bsqConverterPolymer.py).
    2. processing_campaigns    per-station satellite WASI inversion vs
                               in-situ Chl-a / TSM / CDOM match-ups, plot
                               saved to outputs_L3/plots_campaigns
                               (processing_main/processing_campaigns.py).
    3. image_processing        full-image (whole-cube) WASI inversion,
                               writing inverted .bsq + quicklook .png per
                               image to outputs_L3/images_campaigns
                               (processing_main/image_processor_campaigns.py).

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
RUN_PRE_PROCESSING = True
RUN_PROCESSING_CAMPAIGNS = True
RUN_IMAGE_PROCESSING = True


# ═══════════════════════════════════════════════════════════════════════
# EXTERNAL INPUT PATHS — anything NOT inside MSc_thesis_samuel.
# Edit these to match your machine / data location.
# ═══════════════════════════════════════════════════════════════════════

# -- pre_processing: campaign .nc -> band-restricted .bsq conversion --
CAMPAIGNS_NC_INPUT_DIR = r"C:\MSc_thesis_data\satellite\campaigns\nc\output_data"
CAMPAIGNS_BSQ_DIR = r"C:\MSc_thesis_data\satellite\campaigns\bsq"

# 11 bands, matching the 11-band spectrum (400-753.75nm) that
# processing_campaigns.py's PROCESSOR_KWARGS weights vector expects.
CAMPAIGNS_BSQ_BANDS = ['Oa1', 'Oa2', 'Oa3', 'Oa4', 'Oa5', 'Oa6', 'Oa7', 'Oa8', 'Oa10', 'Oa11', 'Oa12']

# -- processing_campaigns: in-situ reference data --
CAMPAIGNS_INSITU_CSV = r"C:\MSc_thesis_data\insitu\campaigns\campaigns_cleaned_all_pigments_remika.csv"


# ═══════════════════════════════════════════════════════════════════════
# INTERNAL PATHS — inside MSc_thesis_samuel, relative to this script.
# ═══════════════════════════════════════════════════════════════════════
CAMPAIGNS_PLOTS_DIR = BASE_DIR / "outputs_L3" / "plots_campaigns"
CAMPAIGNS_IMAGES_OUT_DIR = BASE_DIR / "outputs_L3" / "images_campaigns"


# ═══════════════════════════════════════════════════════════════════════
# Stage implementations
# ═══════════════════════════════════════════════════════════════════════
def run_pre_processing():
    """Convert campaign .nc products to band-restricted .bsq files."""
    print("\n" + "=" * 80)
    print("STAGE 1/3 — pre_processing")
    print("=" * 80)

    from bsqConverterPolymer import convert_polymer_batch

    os.makedirs(CAMPAIGNS_BSQ_DIR, exist_ok=True)
    convert_polymer_batch(
        CAMPAIGNS_NC_INPUT_DIR,
        CAMPAIGNS_BSQ_DIR,
        bands=CAMPAIGNS_BSQ_BANDS,
    )


def run_processing_campaigns():
    """Per-station satellite WASI inversion vs in-situ Chl-a / TSM / CDOM."""
    print("\n" + "=" * 80)
    print("STAGE 2/3 — processing_campaigns")
    print("=" * 80)

    os.makedirs(CAMPAIGNS_PLOTS_DIR, exist_ok=True)
    env = os.environ.copy()
    env["CAMPAIGNS_BSQ_DIR"] = CAMPAIGNS_BSQ_DIR
    env["CAMPAIGNS_INSITU_CSV"] = CAMPAIGNS_INSITU_CSV
    env["CAMPAIGNS_PLOTS_DIR"] = str(CAMPAIGNS_PLOTS_DIR)
    _run_script(PROCESSING_MAIN_DIR / "processing_campaigns.py", env)


def run_image_processing():
    """Full-image WASI inversion, writing inverted .bsq + quicklook .png per image."""
    print("\n" + "=" * 80)
    print("STAGE 3/3 — image_processing")
    print("=" * 80)

    os.makedirs(CAMPAIGNS_IMAGES_OUT_DIR, exist_ok=True)
    env = os.environ.copy()
    env["CAMPAIGNS_BSQ_DIR"] = CAMPAIGNS_BSQ_DIR
    env["CAMPAIGNS_IMAGES_OUT_DIR"] = str(CAMPAIGNS_IMAGES_OUT_DIR)
    _run_script(PROCESSING_MAIN_DIR / "image_processor_campaigns.py", env)


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
    if RUN_PRE_PROCESSING:
        run_pre_processing()
    if RUN_PROCESSING_CAMPAIGNS:
        run_processing_campaigns()
    if RUN_IMAGE_PROCESSING:
        run_image_processing()

    print("\nDone.")
