import os
import pickle


def change_dir_in_lut(map_file, new_data_dir):
    """Rewrite every entry in a date->file .pkl lookup table (LUT) so it
    points into `new_data_dir`, keeping each entry's filename unchanged.
    Overwrites `map_file` in place. Returns the updated dict.

    map_file:     path to the date_to_file_map*.pkl to rewrite
    new_data_dir: directory the rewritten entries should point to
    """
    map_file = str(map_file)

    # Load existing map
    with open(map_file, "rb") as f:
        date_to_file_map = pickle.load(f)

    # Update paths (keep filenames, swap directory)
    updated_map = {
        date: os.path.join(new_data_dir, os.path.basename(filepath))
        for date, filepath in date_to_file_map.items()
    }

    # Save (overwrite)
    with open(map_file, "wb") as f:
        pickle.dump(updated_map, f)

    print(f"Updated {len(updated_map)} entries in {map_file} -> {new_data_dir}")
    return updated_map


if __name__ == "__main__":
    # Default paths for standalone use (`python change_dir_in_LUT.py`)
    map_file = r"C:\MSc_thesis_samuel\LUTs\date_to_file_map.pkl"
    new_data_dir = r"C:\MSc_thesis_data\insitu\thetis\thetis-multi-instrument-profiler\data\Level2_orig"

    change_dir_in_lut(map_file, new_data_dir)
