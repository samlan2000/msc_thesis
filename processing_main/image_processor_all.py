"""
image_processor_all.py
────────────────────────
Full-image (per-pixel, whole-cube) WASI inversion for every satellite image
that was actually used in at least one in-situ comparison, across any of the
three processing chains (thetis, campaigns, shl2) — i.e. every (satellite,
date) pair in outputs_L3/used_images_all.pkl (produced by
count_used_images.py).

Uses ImageProcessor exactly like image_processor_campaigns.py

For every matched image, writes:
  - an inverted ENVI .bsq (+ .hdr) cube to OUT_DIR
  - a quicklook .png of the inverted maps (same basename) to OUT_DIR

Raw .bsq images are looked up, per (satellite, date) pair, across the three
raw archives in this priority order: thetis_valid, shl2/bsq, campaigns/bsq
(confirmed on disk: thetis_valid holds its .bsq/.hdr files directly, while
shl2 and campaigns each hold theirs in a "bsq" subfolder). The same (sat,
date) can physically exist in more than one archive, since the three
in-situ chains overlap (see count_used_images.py's overlap report) — each
pair is only processed once, from the first archive where it's found.

Images whose output (.bsq + .png) already exists in OUT_DIR are skipped, so
an interrupted run can simply be re-launched — a whole-cube WASI inversion
over ~365 images will take a while.

All raw-archive paths are overridable via environment variables, using the
same names main_thetis.py / main_shl2.py / main_campaigns.py already use
(THETIS_BSQ_VALID_DIR, SHL2_BSQ_DIR, CAMPAIGNS_BSQ_DIR), so a future
main_all.py could wire them centrally. Defaults match this repo's current
layout on this machine.

NOTE: this script is written but intentionally NOT executed here — running
it (the actual batch WASI inversion) is left for you to kick off, since it
will be time consuming.
"""

import glob
import os
import pickle
from pathlib import Path

from ImageProcessor import ImageProcessor

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Raw .bsq archives, searched in this order for each (sat, date) pair.
THETIS_BSQ_VALID_DIR = os.environ.get("THETIS_BSQ_VALID_DIR", r"C:\MSc_thesis_data\satellite\thetis_valid")
SHL2_BSQ_DIR = os.environ.get("SHL2_BSQ_DIR", r"C:\MSc_thesis_data\satellite\shl2\bsq")
CAMPAIGNS_BSQ_DIR = os.environ.get("CAMPAIGNS_BSQ_DIR", r"C:\MSc_thesis_data\satellite\campaigns\bsq")
IMAGE_DIRS = [THETIS_BSQ_VALID_DIR, SHL2_BSQ_DIR, CAMPAIGNS_BSQ_DIR]

# outputs_L3/images_all, reconstructed relative to this script's location
OUT_DIR = Path(os.environ.get("ALL_IMAGES_OUT_DIR", str(BASE_DIR / "outputs_L3" / "images_all")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# set of (sat, date_str) pairs actually used in >=1 in-situ comparison,
# across all three chains — already computed and saved by count_used_images.py
ALL_IMAGES_PKL = os.environ.get("ALL_IMAGES_PKL", str(BASE_DIR / "outputs_L3" / "used_images_all.pkl"))
with open(ALL_IMAGES_PKL, "rb") as f:
    all_images = pickle.load(f)

PROCESSOR_KWARGS = dict(
    weights=[0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0],
    vary={"C_x": True, "C_y": True, "C_0": False, "C_1": True, "C_2": True, "C_3": True, "C_4": True,  "C_5": True},
    init={"C_y": 0.1, "C_0": 0, "C_x": 1, "C_1": 1, "C_2": 1, "C_3": 1, "C_4": 1,  "C_5": 1},
    output_wcs=["C_x", "C_1", "C_2", "C_3", "C_4", "C_5", "C_y"],
    output_iops=["bb", "a", "a_cdom", "a_nap", "a_phy", "bb_nap", "bb_phy"],
)


def find_image_path(sat, date_str):
    """Locate the raw .bsq file for (sat, date_str) across the three raw
    archives (IMAGE_DIRS, in priority order). Filenames follow the
    "{sat}_{YYYYMMDD}T{HHMMSS}.bsq" convention (e.g. S3A_20181018T101646.bsq)
    — the time-of-day isn't part of the (sat, date) pair, so it's globbed."""
    date_compact = date_str.replace("-", "")
    pattern = f"{sat}_{date_compact}T*.bsq"
    for d in IMAGE_DIRS:
        matches = sorted(glob.glob(os.path.join(d, pattern)))
        if matches:
            if len(matches) > 1:
                print(f"  [warn] {len(matches)} matches for {sat} {date_str} in {d}, using {matches[0]}")
            return matches[0]
    return None


# ─────────────────────────────────────────────
# Batch process images
# ─────────────────────────────────────────────
processed, skipped_existing, missing = [], [], []

for sat, date_str in sorted(all_images):
    image_path = find_image_path(sat, date_str)
    if image_path is None:
        missing.append((sat, date_str))
        continue

    img_name = os.path.basename(image_path)
    out_path = os.path.join(OUT_DIR, img_name)
    out_path_figure = out_path.replace(".bsq", ".png")

    if os.path.exists(out_path) and os.path.exists(out_path_figure):
        skipped_existing.append((sat, date_str))
        continue

    print(f"{'='*80}\n{sat} {date_str}  ({img_name})")

    processor = ImageProcessor(image_path, out_path, **PROCESSOR_KWARGS)
    processor.plot_results(processor.results, out_path=out_path_figure)
    processed.append((sat, date_str))

# ─────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────
print()
print(f"processed:                          {len(processed)}")
print(f"skipped (output already existed):   {len(skipped_existing)}")
print(f"missing (not found in any archive): {len(missing)}")
if missing:
    for sat, date_str in missing:
        print(f"  missing: {sat} {date_str}")
