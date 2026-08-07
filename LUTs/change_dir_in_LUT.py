import os
import pickle

# Paths
map_file = r"C:\MSc_thesis_samuel\LUTs\date_to_file_map.pkl"
new_data_dir = r"C:\MSc_thesis_data\insitu\thetis\thetis-multi-instrument-profiler\data\Level2_orig"

# Load existing map
with open(map_file, "rb") as f:
    date_to_file_map = pickle.load(f)

# Update paths
updated_map = {
    date: os.path.join(new_data_dir, os.path.basename(filepath))
    for date, filepath in date_to_file_map.items()
}

# Save (overwrite or change output filename if preferred)
with open(map_file, "wb") as f:
    pickle.dump(updated_map, f)

print(f"Updated {len(updated_map)} entries.")