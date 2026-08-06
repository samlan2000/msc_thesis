"""
image_processor_campaigns.py
──────────────────────────────
Full-image (per-pixel, whole-cube) WASI inversion for the campaign .bsq
archive, using ImageProcessor (formerly ImageProcessorV3) instead of the
per-station SinglePixelProcessor used in processing_campaigns.py.

For every .bsq image in IMG_DIR, writes:
  - an inverted ENVI .bsq (+ .hdr) cube to OUT_DIR
  - a quicklook .png of the inverted maps (same basename) to OUT_DIR

Adapted from an older standalone reference script (single hard-coded image
via ImageProcessorV3) into a batch loop over the whole campaign archive
using the current ImageProcessor class.

All paths below are overridable via environment variables so that
main_campaigns.py can control them centrally. Defaults are only used when
this script is run standalone.
"""

import os
from pathlib import Path
from ImageProcessor import ImageProcessor

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

IMG_DIR = os.environ.get("CAMPAIGNS_BSQ_DIR", r"C:\MSc_thesis_data\satellite\campaigns\bsq")

# outputs_L3/images_campaigns, reconstructed relative to this script's location
OUT_DIR = Path(os.environ.get("CAMPAIGNS_IMAGES_OUT_DIR", str(BASE_DIR / "outputs_L3" / "images_campaigns")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSOR_KWARGS = dict(
    weights=None,
    vary={"C_x": True, "C_y": True, "C_0": False, "C_3": True, "C_5": True},
    init={"C_y": 0.1, "C_0": 0, "C_x": 1, "C_3": 2, "C_5": 2},
    output_wcs=["C_x", "C_3", "C_5", "C_y"],
    output_iops=["bb", "a", "a_cdom", "a_nap", "a_phy", "bb_nap", "bb_phy"],
)

# ─────────────────────────────────────────────
# Batch process images
# ─────────────────────────────────────────────
for img in os.listdir(IMG_DIR):
    if img.startswith("_") or not img.endswith(".bsq"):
        continue

    image_path = os.path.join(IMG_DIR, img)
    out_path = os.path.join(OUT_DIR, img)
    out_path_figure = out_path.replace(".bsq", ".png")

    print(f"{'='*80}\n{img}")

    processor = ImageProcessor(image_path, out_path, **PROCESSOR_KWARGS)
    processor.plot_results(processor.results, out_path=out_path_figure)
