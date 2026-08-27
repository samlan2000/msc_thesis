"""
count_used_images.py
─────────────────────
Counts the total number of distinct satellite images (satellite + date)
that were actually used in at least one in-situ comparison, across all
three processing chains (thetis, campaigns, shl2). Same (satellite, date)
image used by more than one chain is only counted once.

Requires:
  - outputs_L3/db_thetis.pkl                 (from processing_thetis.py)
  - outputs_L3/used_images_campaigns.pkl     (from processing_campaigns.py)
  - outputs_L3/used_images_shl2.pkl          (from processing_shl2.py)

The campaigns/shl2 files are only written once those scripts have been run
(they need the external data under C:\\MSc_thesis_data, so run main_campaigns.py
/ main_shl2.py first if the .pkl files are missing).

Unlike earlier versions of this script, the thetis chain is no longer read
from a separately precomputed outputs_L3/db_thetis_nonempty.pkl. Instead it
reads the raw db_thetis.pkl directly (sat -> date -> record for EVERY
processed image, whether or not it has a matching in-situ sample) and does
its own non-empty check inline: a record's satellite-derived fields (keys
ending in "_sat") are always populated for any valid image, but its in-situ
fields (every other key: the "_R" fields plus CHL_A, CHL_F, aLH676, bb440,
bb532, bb630, bb700, Rrs, a) are only non-NaN when an in-situ sample was
actually matched up to that image. A (sat, date) pair counts as "used" here
only if at least one of those in-situ fields is non-NaN.
"""

import pickle
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs_L3"


def _is_all_nan(value):
    """True if `value` (scalar or array-like) is NaN / all-NaN, or None."""
    if value is None:
        return True
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        # non-numeric field (shouldn't occur here) -> treat as present/non-empty
        return False
    return bool(np.all(np.isnan(arr)))


def record_has_insitu(record):
    """True if `record` has at least one non-NaN in-situ field (any key not
    ending in "_sat")."""
    return any(
        not _is_all_nan(value)
        for key, value in record.items()
        if not key.endswith("_sat")
    )


chains = {}

# ── thetis ── read the raw db and keep only (sat, date) pairs with >=1
# non-NaN in-situ field (see record_has_insitu above).
thetis_path = OUT_DIR / "db_thetis.pkl"
if thetis_path.exists():
    with open(thetis_path, "rb") as f:
        db = pickle.load(f)
    chains["thetis"] = {
        (sat, date)
        for sat, dates in db.items()
        for date, record in dates.items()
        if record_has_insitu(record)
    }
else:
    print(f"[skip] {thetis_path} not found")
    chains["thetis"] = set()

# ── campaigns / shl2 ── sets of (sat, date) saved directly by the scripts.
for name in ("campaigns", "shl2"):
    p = OUT_DIR / f"used_images_{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            chains[name] = pickle.load(f)
    else:
        print(f"[skip] {p} not found — run processing_{name}.py (via main_{name}.py) first")
        chains[name] = set()

# ── report ──
print()
for name, s in chains.items():
    print(f"{name:>10}: {len(s)} images")

all_images = set().union(*chains.values())
print(f"\n{'union (total)':>10}: {len(all_images)} distinct images used in comparisons")

all_images_path = OUT_DIR / "used_images_all.pkl"
with open(all_images_path, "wb") as f:
    pickle.dump(all_images, f)
print(f"\nSaved {all_images_path} ({len(all_images)} (satellite, date) pairs)")

# overlap between chains, for context
names = list(chains.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        overlap = chains[names[i]] & chains[names[j]]
        if overlap:
            print(f"  overlap {names[i]} ∩ {names[j]}: {len(overlap)}  {sorted(overlap)}")
