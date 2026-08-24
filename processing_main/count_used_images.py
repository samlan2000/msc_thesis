"""
count_used_images.py
─────────────────────
Counts the total number of distinct satellite images (satellite + date)
that were actually used in at least one in-situ comparison, across all
three processing chains (thetis, campaigns, shl2). Same (satellite, date)
image used by more than one chain is only counted once.

Requires:
  - outputs_L3/db_thetis_nonempty.pkl        (from processing_thetis.py)
  - outputs_L3/used_images_campaigns.pkl     (from processing_campaigns.py)
  - outputs_L3/used_images_shl2.pkl          (from processing_shl2.py)

The campaigns/shl2 files are only written once those scripts have been run
(they need the external data under C:\\MSc_thesis_data, so run main_campaigns.py
/ main_shl2.py first if the .pkl files are missing).
"""

import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs_L3"

chains = {}

# ── thetis ── keys of db_thetis_nonempty.pkl are (sat -> date -> record);
# every (sat, date) key pair there already has >=1 non-NaN in-situ field.
thetis_path = OUT_DIR / "db_thetis_nonempty.pkl"
if thetis_path.exists():
    with open(thetis_path, "rb") as f:
        db = pickle.load(f)
    chains["thetis"] = {(sat, date) for sat, dates in db.items() for date in dates}
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

# overlap between chains, for context
names = list(chains.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        overlap = chains[names[i]] & chains[names[j]]
        if overlap:
            print(f"  overlap {names[i]} ∩ {names[j]}: {len(overlap)}  {sorted(overlap)}")
