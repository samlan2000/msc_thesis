# -*- coding: utf-8 -*-
import sys
import os
import yaml
from functions import log, error, parse_args
from instruments import process_CTD, process_DO, process_PAR, process_TRIP1, process_TRIP2, process_ACS, process_OCR1, process_OCR2, process_grid



    
with open(r"C:\thetis\thetis-multi-instrument-profiler\scripts\input_python.yaml", "r") as f:
    directories = yaml.load(f, Loader=yaml.FullLoader)

directory, ids = parse_args(directories["Level0_dir"])
log("Python script to process the output data of the Thetis profiler. Collecting profiles from: " + directory, start=True)

log("Creating directories")
for d in directories.values():
    if not os.path.exists(d):
        os.makedirs(d)
ids.sort()

log("Looping over input ids")
for id in ids:
    log("Processing files with input id: "+id, 1)

    l2_datasets = {}
    log("Processing CTD data", 2)
    CTD = process_CTD()
    if CTD.read_data(id, directory):
        CTD.multiple_profiles()
        CTD.quality_flags()
        CTD.to_NetCDF(directories["Level1_dir"], "L1")
        CTD.mask_data()
        CTD_data = CTD.export_data()
        l2_datasets = CTD.resample_to_fixed_grid(l2_datasets, "depth")

        log("Processing DO data", 2)
        DO = process_DO()
        if DO.read_data(id, directory, CTD_data):
            DO.multiple_profiles()
            DO.quality_flags()
            DO.to_NetCDF(directories["Level1_dir"], "L1")
            DO.mask_data()
            l2_datasets = DO.resample_to_fixed_grid(l2_datasets, "depth")

        log("Processing PAR data", 2)
        PAR = process_PAR()
        if PAR.read_data(id, directory, CTD_data):
            PAR.multiple_profiles()
            PAR.quality_flags()
            PAR.to_NetCDF(directories["Level1_dir"], "L1")
            PAR.mask_data()
            l2_datasets = PAR.resample_to_fixed_grid(l2_datasets, "depth")

        log("Processing TRIP1 data", 2)
        TRIP1 = process_TRIP1()
        if TRIP1.read_data(id, directory, CTD_data):
            TRIP1.multiple_profiles()
            TRIP1.quality_flags()
            TRIP1.to_NetCDF(directories["Level1_dir"], "L1")
            TRIP1.mask_data()
            l2_datasets = TRIP1.resample_to_fixed_grid(l2_datasets, "depth")

        log("Processing TRIP2 data", 2)
        TRIP2 = process_TRIP2()
        if TRIP2.read_data(id, directory, CTD_data):
            TRIP2.multiple_profiles()
            TRIP2.quality_flags()
            TRIP2.to_NetCDF(directories["Level1_dir"], "L1")
            TRIP2.mask_data()
            l2_datasets = TRIP2.resample_to_fixed_grid(l2_datasets, "depth")

        log("Processing ACS data", 2)
        ACS = process_ACS()
        if ACS.read_data(id, directory, directories["Calibration_dir"], CTD_data):
            ACS.quality_flags()
            ACS.to_NetCDF(directories["Level1_dir"], "L1")
            ACS.mask_data()
            # 1D variables
            ACS.grid = ACS.grid_depth
            l2_datasets = ACS.resample_to_fixed_grid(l2_datasets, "depth")
            # hyperspectral variables
            ACS.grid = ACS.grid_hyper
            l2_datasets = ACS.resample_to_fixed_grid(l2_datasets, "special")

        log("Processing OCR data", 2)
        OCR1 = process_OCR1()
        if OCR1.read_data(id, directory, directories["Calibration_dir"], CTD_data):
            OCR1.quality_flags()
            OCR1.to_NetCDF(directories["Level1_dir"], "L1") 
            OCR1.mask_data()
            l2_datasets = OCR1.resample_to_fixed_grid(l2_datasets, "wavelength")

        log("Processing OCR data", 2)
        OCR2 = process_OCR2()
        if OCR2.read_data(id, directory, directories["Calibration_dir"], CTD_data):
            OCR2.quality_flags()
            OCR2.to_NetCDF(directories["Level1_dir"], "L1")
            OCR2.mask_data()
            l2_datasets = OCR2.resample_to_fixed_grid(l2_datasets, "wavelength")

        grid = process_grid()
        l2_datasets = grid.radiance_products(l2_datasets)
        grid.createl2product(directories["Level2_dir"], l2_datasets)
