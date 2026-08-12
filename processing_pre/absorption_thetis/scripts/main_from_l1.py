# -*- coding: utf-8 -*-

import os
import yaml
import glob
import netCDF4
import numpy as np

from functions import log, parse_args
from instruments import (
    process_CTD,
    process_DO,
    process_PAR,
    process_TRIP1,
    process_TRIP2,
    process_ACS,
    process_OCR1,
    process_OCR2,
    process_grid
)


def load_L1_dataset(obj, filepath):
    """Load NetCDF L1 file into instrument object."""
    log("Loading L1 file: " + filepath, 3)

    nc = netCDF4.Dataset(filepath)

    obj.data = {}
    for var in nc.variables:
        obj.data[var] = np.array(nc.variables[var][:])

    nc.close()

    return True


def find_l1_file(level1_dir, instrument, id):
    """Locate L1 NetCDF file for a given instrument and profile id."""
    folder = os.path.join(level1_dir, instrument)

    pattern = os.path.join(folder, f"L1_THETIS_{instrument}_{id}_*.nc")
    files = glob.glob(pattern)

    if len(files) == 0:
        return None

    return files[0]


# -----------------------------

with open(r"C:\thetis\thetis-multi-instrument-profiler\scripts\input_python.yaml", "r") as f:
    directories = yaml.load(f, Loader=yaml.FullLoader)


directory, ids = parse_args(directories["Level0_dir"])

log(
    "Python script to generate Level2 data from existing Level1 NetCDF files",
    start=True
)

ids.sort()

log("Looping over input ids")

for id in ids:

    log("Processing files with input id: " + id, 1)

    l2_datasets = {}

    # -----------------------
    # CTD
    # -----------------------

    log("Loading CTD L1 data", 2)

    CTD = process_CTD()
    file = find_l1_file(directories["Level1_dir"], "CTD", id)

    if file:

        load_L1_dataset(CTD, file)

        CTD_data = CTD.export_data()
        l2_datasets = CTD.resample_to_fixed_grid(l2_datasets, "depth")

    else:
        log("No CTD L1 file found", 3)
        continue

    # -----------------------
    # DO
    # -----------------------

    log("Loading DO L1 data", 2)

    DO = process_DO()
    file = find_l1_file(directories["Level1_dir"], "DO", id)

    if file:

        load_L1_dataset(DO, file)
        l2_datasets = DO.resample_to_fixed_grid(l2_datasets, "depth")

    # -----------------------
    # PAR
    # -----------------------

    log("Loading PAR L1 data", 2)

    PAR = process_PAR()
    file = find_l1_file(directories["Level1_dir"], "PAR", id)

    if file:

        load_L1_dataset(PAR, file)
        l2_datasets = PAR.resample_to_fixed_grid(l2_datasets, "depth")

    # -----------------------
    # TRIP1
    # -----------------------

    log("Loading TRIP1 L1 data", 2)

    TRIP1 = process_TRIP1()
    file = find_l1_file(directories["Level1_dir"], "TRIP1", id)

    if file:

        load_L1_dataset(TRIP1, file)
        l2_datasets = TRIP1.resample_to_fixed_grid(l2_datasets, "depth")

    # -----------------------
    # TRIP2
    # -----------------------

    log("Loading TRIP2 L1 data", 2)

    TRIP2 = process_TRIP2()
    file = find_l1_file(directories["Level1_dir"], "TRIP2", id)

    if file:

        load_L1_dataset(TRIP2, file)
        l2_datasets = TRIP2.resample_to_fixed_grid(l2_datasets, "depth")

    # -----------------------
    # ACS
    # -----------------------

    log("Loading ACS L1 data", 2)

    ACS = process_ACS()
    file = find_l1_file(directories["Level1_dir"], "ACS", id)

    if file:

        load_L1_dataset(ACS, file)
        l2_datasets = ACS.resample_to_fixed_grid(l2_datasets, "depth")

    # -----------------------
    # OCR1
    # -----------------------

    log("Loading OCR1 L1 data", 2)

    OCR1 = process_OCR1()
    file = find_l1_file(directories["Level1_dir"], "OCR1", id)

    if file:

        load_L1_dataset(OCR1, file)
        l2_datasets = OCR1.resample_to_fixed_grid(l2_datasets, "wavelength")

    # -----------------------
    # OCR2
    # -----------------------

    log("Loading OCR2 L1 data", 2)

    OCR2 = process_OCR2()
    file = find_l1_file(directories["Level1_dir"], "OCR2", id)

    if file:

        load_L1_dataset(OCR2, file)
        l2_datasets = OCR2.resample_to_fixed_grid(l2_datasets, "wavelength")

    # -----------------------
    # Create L2 grid
    # -----------------------

    grid = process_grid()

    l2_datasets = grid.radiance_products(l2_datasets)

    grid.createl2product(
        directories["Level2_dir"],
        l2_datasets
    )