import os
import shutil
from PixelProcessor import SinglePixelProcessor

# ─────────────────────────────────────────────
# Paths — overridable via environment variables so main_thetis.py can control
# them centrally. Defaults below are only used when this script is run
# standalone.
# ─────────────────────────────────────────────
directory = os.environ.get("THETIS_BSQ_COMBINED_DIR", r"C:\MSc_thesis_data\satellite\thetis_combined\bsq")
copy_dir = os.environ.get("THETIS_BSQ_VALID_DIR", r"C:\MSc_thesis_data\satellite\thetis_valid")

all_valid_images = set()

os.makedirs(copy_dir, exist_ok=True)

for img in os.listdir(directory):
    if not img.endswith(".bsq"):
        continue
    bsq_path = os.path.join(directory, img)
    hdr_path = bsq_path.replace(".bsq", ".hdr")

    res = SinglePixelProcessor(bsq_path, i_offset=1, j_offset=0, valid_pixel_min=2000, station_name="lxp")
    if res.inv:
        all_valid_images.add(img)

        shutil.copy2(bsq_path, os.path.join(copy_dir, os.path.basename(bsq_path)))
        if os.path.exists(hdr_path):
            shutil.copy2(hdr_path, os.path.join(copy_dir, os.path.basename(hdr_path)))
        else:
            print(f"Warning: missing header file for {img}: {hdr_path}")
