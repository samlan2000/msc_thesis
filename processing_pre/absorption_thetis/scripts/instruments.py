# -*- coding: utf-8 -*-

import netCDF4
import pandas as pd
import numpy as np
import os
import re
import sys
import glob
import math
import statistics
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime, timezone
from copy import deepcopy
import json
from functions import log, error, isnt_number, scattering_correction, position_in_array, oxygen_saturation, \
    counts_to_spectra, temperature_salinity_correction, find_closest_index, bin_array, \
    absorption_line_height, spectral_attenuation_slope, read_ocr_calibration_data, \
    spectral_light_attenuation_coefficient, read_acs_calibration_data, despike, smooth_acs_vertically


class thetis(object):
    def __init__(self):
        self.id = ""
        self.folder = ""
        self.data = {}
        self.general_attributes = {
            "institution": "EPFL",
            "source": "Thetis CTD",
            "references": "LéXPLORE common instruments camille.minaudo@ub.edu>",  # consider changing contact person
            "history": "See history on Renku",
            "conventions": "CF 1.7",
            "comment": "Data from Thetis profiler on Lexplore Platform in Lake Geneva",
            "title": "Lexplore Thetis"
        }

    def export_data(self):
        return self.data

    def to_NetCDF(self, folder, prefix):
        log("Writing " + prefix + " data to NetCDF", 3)

        dt_max = datetime.utcfromtimestamp(np.nanmax(self.data["time"]))
        today = datetime.utcnow()

        if dt_max > today:
            log("Failed to write future data to NetCDF")
            return False

        dt = datetime.utcfromtimestamp(np.nanmin(self.data["time"])).strftime('%Y%m%d_%H%M%S')

        file_folder = os.path.join(folder, self.type)
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        filename = "_".join([prefix, "THETIS", self.type, self.id, dt + ".nc"])
        filepath = os.path.join(file_folder, filename)

        nc = netCDF4.Dataset(filepath, mode='w', format='NETCDF4')

        for key in self.general_attributes:
            setattr(nc, key, self.general_attributes[key])

        for key, values in self.dimensions.items():
            nc.createDimension(values['dim_name'], values['dim_size'])

        for key, values in self.variables.items():
            var = nc.createVariable(values["var_name"], np.float64, values["dim"], fill_value=np.nan)
            var.units = values["unit"]
            var.long_name = values["longname"]
            var[:] = self.data[key]

        # Close NetCDF file
        nc.close()
        log("Successfully wrote NetCDF file", 3)

    def mask_data(self):
        log("Masking flagged data", 3)
        for var in self.variables:
            if "_qual" not in var:
                idx = self.data[var + "_qual"] > 0
                self.data[var][idx] = np.nan

    def resample_to_fixed_grid(self, products, type):
        bin_size = 0.1
        if "depth" not in products:
            products["depth"] = {'var_name': 'depth', 'dim': ('depth',), 'unit': 'm', 'longname': 'depth',
                                 "data": np.arange(1, 50.1, bin_size)}
        if "time" not in products:
            products["time"] = {'var_name': 'time', 'dim': ('time',), 'unit': 'seconds since 1970-01-01 00:00:00',
                                'longname': 'time', "data": [self.data["time"][0]]}
        if "wavelength" not in products:
            products["wavelength"] = {'var_name': 'wavelength', 'dim': ('wavelength',), 'unit': 'nm',
                                      'longname': 'wavelength', "data": np.array(range(303, 907))}
            
        if type == "special":
            log("Resampling hyperspectral data to fixed depth grid (keeping wavelength)", 3)
        
            for var in self.grid:
                products[var] = self.variables[var].copy()
                products[var]["dim"] = ('depth', 'wavelength', 'time')
        
                products[var]["data"] = np.full(
                    (
                        len(products["depth"]["data"]),
                        len(products["wavelength"]["data"]),
                        1  # single profile = 1 time step
                    ),
                    np.nan
                )
        
            depth = self.data["depth"]
            depth_min = depth.min()
            depth_max = depth.max()
        
            log("Min depth of profile: " + str(depth_min) + "m", 4)
            log("Max depth of profile: " + str(depth_max) + "m", 4)
        
            for idx, entry in enumerate(products["depth"]["data"]):
                if depth_min <= entry <= depth_max:
        
                    upper = depth[depth >= entry].min()
                    lower = depth[depth <= entry].max()
        
                    upper_index = np.where(depth == upper)[0][0]
                    lower_index = np.where(depth == lower)[0][0]
        
                    x0 = lower
                    x1 = upper
                    x = entry
        
                    for var in self.grid:
                        y0 = self.data[var][lower_index, :]
                        y1 = self.data[var][upper_index, :]
                    
                        if x1 != x0:
                            spec_depth = y0 + (x - x0) * ((y1 - y0) / (x1 - x0))
                        else:
                            spec_depth = y0
                    
                        spec_interp = np.interp(
                            products["wavelength"]["data"],
                            self.data["wavelength"],
                            spec_depth
                        )
                    
                        products[var]["data"][idx, :, 0] = spec_interp
            

        elif type == "depth":
            log("Resampling to fixed grid (1m to 50m at 0.1m intervals)", 3)
            for var in self.grid:
                products[var] = self.variables[var]
                products[var]["dim"] = ('depth', 'time')
                products[var]["data"] = np.full(products["depth"]["data"].shape, np.nan)

            depth = self.data["depth"]
            depth_min = depth.min()
            depth_max = depth.max()
            log("Min depth of profile: " + str(depth_min) + "m", 4)
            log("Max depth of profile: " + str(depth_max) + "m", 4)
            for idx, entry in enumerate(products["depth"]["data"]):
                if depth_min <= entry <= depth_max:
                    upper = depth[depth >= entry].min()
                    lower = depth[depth <= entry].max()
                    upper_index = np.where(depth == upper)[0][0]
                    lower_index = np.where(depth == lower)[0][0]
                    for var in self.grid:
                        y0 = self.data[var][lower_index]
                        y1 = self.data[var][upper_index]
                        x0 = lower
                        x1 = upper
                        x = entry
                        products[var]["data"][idx] = y0 + (x - x0) * (
                                (y1 - y0) / (x1 - x0))  # Interpolate between upper and lower
        elif type == "wavelength":
            log("Resampling to fixed grid (303nm to 907m at 1nm intervals)", 3)
            for var in self.grid:
                products[var] = self.variables[var]
                products[var]["dim"] = ('wavelength', 'time')
                products[var]["data"] = np.full(products["wavelength"]["data"].shape, np.nan)

            wavelength = self.data["wavelength"]

            wavelength_min = wavelength.min()
            wavelength_max = wavelength.max()
            log("Min wavelength of profile: " + str(wavelength_min) + "nm", 4)
            log("Max wavelength of profile: " + str(wavelength_max) + "nm", 4)
            for idx, entry in enumerate(products["wavelength"]["data"]):
                if wavelength_min <= entry <= wavelength_max:
                    upper = wavelength[wavelength >= entry].min()
                    lower = wavelength[wavelength <= entry].max()
                    upper_index = np.where(wavelength == upper)[0][0]
                    lower_index = np.where(wavelength == lower)[0][0]
                    for var in self.grid:
                        y0 = self.data[var][lower_index]
                        y1 = self.data[var][upper_index]
                        x0 = lower
                        x1 = upper
                        x = entry
                        products[var]["data"][idx] = y0 + (x - x0) * (
                                (y1 - y0) / (x1 - x0))  # Interpolate between upper and lower

        return products

    def multiple_profiles(self):
        ## Detect if the file has more than 1 profile

        # Check if more than 1 pressure measurement
        if len(self.data["depth"]) <= 1:
            return

        # Calculate 1st and 2nd derivative
        depth_1st = np.gradient(self.data["depth"])
        depth_2nd = np.gradient(depth_1st)

        # Calculate 75th percentile of 2nd derivative
        depth_2nd_median = np.percentile(depth_2nd, 75)
        depth_2nd_max = np.max(np.abs(depth_2nd))

        if depth_2nd_max >= np.abs(depth_2nd_median) * 10:
            ind = np.where(np.abs(depth_2nd) == depth_2nd_max)[0][0]
            depth_diff = np.diff(self.data["depth"])
            sum_before = sum(depth_diff[0:ind])
            sum_after = sum(depth_diff[ind:-1])
        else:
            ind = 0
            sum_before = 0
            sum_after = 0

        # Conditions more than two profiles
        if sum_before > 0 > sum_after and np.max(self.data["depth"]) >= 30 and ind >= 20:
            log("File contains more than 1 profile.", 3)

    def quality_flags(self):
        variables = self.variables.copy().items()
        for key, values in variables:
            name = key + "_qual"
            self.variables[name] = {'var_name': name, 'dim': values["dim"],
                                    'unit': '0 = nothing to report, 1 = more investigation',
                                    'longname': name, }
            qa_data = np.zeros_like(self.data[key])
            isnt_numeric = np.vectorize(isnt_number, otypes=[bool])
            qa_data[(isnt_numeric(self.data[key]))] = 1
            if "min" in values:
                qa_data[(self.data[key] < float(values["min"]))] = 1
            if "max" in values:
                qa_data[(self.data[key] > float(values["max"]))] = 1
            self.data[name] = np.array(qa_data)


class process_grid(thetis):
    def __init__(self, *args, **kwargs):
        super(process_grid, self).__init__(*args, **kwargs)

        self.general_attributes["title"] = "Lexplore Thetis Depth Time Grid"

        self.dimensions = {
            'time': {'dim_name': 'time', 'dim_size': None},
            'depth': {'dim_name': 'depth', 'dim_size': None},
            'wavelength': {'dim_name': 'wavelength', 'dim_size': None},
        }

    def radiance_products(self, l2_datasets):
        log("Creating radiance products", 2)
        if "Lu0" in l2_datasets and "Ed0" in l2_datasets:
            log("Calculating Rrs", 3)
            Rrs = l2_datasets["Lu0"]["data"] / l2_datasets["Ed0"]["data"] * math.pi * 0.544
            l2_datasets["Rrs"] = {'var_name': 'Rrs', 'dim': ('wavelength', 'time'), 'unit': '1/sr',
                                  'longname': 'Water leaving reflectance', 'data': Rrs}
        return l2_datasets

    def createl2product(self, folder, products):
        log("Writing L2 data to NetCDF", 2)
    
        if not os.path.exists(folder):
            os.makedirs(folder)
    
        timestamp = products["time"]["data"][0]
        interval = 10
        grid_start = datetime(2018, 1, 1)
        ts = datetime.utcfromtimestamp(timestamp)
    
        while grid_start < ts - dt.timedelta(days=interval):
            grid_start = grid_start + dt.timedelta(days=interval)
    
        date_time = grid_start.strftime('%Y%m%d')
        date_time_end = (grid_start + dt.timedelta(days=interval)).strftime('%Y%m%d')
    
        filename = "_".join(["L2_THETIS_GRID", date_time, date_time_end + ".nc"])
        filepath = os.path.join(folder, filename)
    
        # =========================
        # APPEND MODE
        # =========================
        if os.path.isfile(filepath):
            nc = netCDF4.Dataset(filepath, mode='a', format='NETCDF4')
            time = nc.variables['time']
    
            if timestamp in time:
                duplicate = True
                idx = list(time).index(timestamp)
            else:
                duplicate = False
                idx = position_in_array(time[:], timestamp)
                time[:] = np.insert(time[:], idx, timestamp)
    
            for product in products:
    
                # Create variable if missing
                if product in nc.variables:
                    var = nc.variables[product]
                else:
                    var = nc.createVariable(
                        products[product]["var_name"],
                        np.float64,
                        products[product]["dim"],
                        fill_value=np.nan
                    )
                    var.units = products[product]["unit"]
                    var.long_name = products[product]["longname"]
    
                # Skip coordinates
                if product in ["depth", "time", "wavelength"]:
                    continue
    
                data = products[product]["data"]
    
                # =========================
                # HANDLE DIMENSIONS
                # =========================
    
                # ---- 2D CASE (depth,time) or (wavelength,time)
                if len(var.shape) == 2:
                    if duplicate and np.all(np.isnan(var[:, idx])):
                        var[:, idx] = data
                    else:
                        end = var.shape[1] - 1
                        if idx != end:
                            var[:, end] = data
                            var[:] = var[:, np.insert(np.arange(end), idx, end)]
                        else:
                            var[:, idx] = data
    
                # ---- 3D CASE (depth,wavelength,time)
                elif len(var.shape) == 3:
                    if duplicate and np.all(np.isnan(var[:, :, idx])):
                        var[:, :, idx] = data[:, :, 0]
                    else:
                        end = var.shape[2] - 1
                        if idx != end:
                            var[:, :, end] = data[:, :, 0]
                            var[:] = var[:, :, np.insert(np.arange(end), idx, end)]
                        else:
                            var[:, :, idx] = data[:, :, 0]
    
            nc.close()
    
        # =========================
        # CREATE NEW FILE
        # =========================
        else:
            nc = netCDF4.Dataset(filepath, mode='w', format='NETCDF4')
    
            for key in self.general_attributes:
                setattr(nc, key, self.general_attributes[key])
    
            # Create dimensions
            for key, values in self.dimensions.items():
                nc.createDimension(values['dim_name'], values['dim_size'])
    
            for product in products:
                var = nc.createVariable(
                    products[product]["var_name"],
                    np.float64,
                    products[product]["dim"],
                    fill_value=np.nan
                )
    
                var.units = products[product]["unit"]
                var.long_name = products[product]["longname"]
    
                data = products[product]["data"]
    
                # =========================
                # INITIAL WRITE
                # =========================
    
                if product == "time":
                    var[0] = data
    
                elif len(products[product]["dim"]) == 1:
                    var[:] = data
    
                elif len(products[product]["dim"]) == 2:
                    var[:, 0] = data
    
                elif len(products[product]["dim"]) == 3:
                    var[:, :, 0] = data
    
            nc.close()
            log("Successfully wrote NetCDF file", 3)


class process_CTD(thetis):
    def __init__(self, *args, **kwargs):
        super(process_CTD, self).__init__(*args, **kwargs)

        self.general_attributes["title"] = "Lexplore Thetis CTD"

        self.dimensions = {
            'time': {'dim_name': 'time', 'dim_size': None}
        }
        self.variables = {
            'time': {'var_name': 'time', 'dim': ('time',), 'unit': 'seconds since 1970-01-01 00:00:00',
                     'longname': 'time', "min": 1514764800, "max": datetime.now().timestamp()},
            'cond': {'var_name': 'cond', 'dim': ('time',), 'unit': 'microS/cm', 'longname': 'conductivity', "min": 0,
                     "max": 500},
            'temp': {'var_name': 'temp', 'dim': ('time',), 'unit': 'degC', 'longname': 'temperature', "min": 2,
                     "max": 40},
            'press': {'var_name': 'press', 'dim': ('time',), 'unit': 'dbar', 'longname': 'pressure', "min": 0,
                      "max": 150},
            'sal': {'var_name': 'sal', 'dim': ('time',), 'unit': 'mg/l', 'longname': 'salinity'},
            'cond20': {'var_name': 'cond20', 'dim': ('time',), 'unit': 'microS/cm',
                       'longname': 'conductivity normalised at 20degC', "min": 0, "max": 500},
            'depth': {'var_name': 'depth', 'dim': ('time',), 'unit': 'm', 'longname': 'depth', "min": 0, "max": 100}
        }
        self.offset = 0
        self.type = "CTD"
        self.grid = ["cond", "temp", "sal", "cond20"]

    def read_data(self, id, folder):
        self.id = id
        self.folder = folder
        if os.path.isfile(os.path.join(folder, str(id) + "_PPB_CTD.txt")):
            file = os.path.join(folder, str(id) + "_PPB_CTD.txt")
        elif os.path.isfile(os.path.join(folder, str(id) + "_PPD_CTD.txt")):
            file = os.path.join(folder, str(id) + "_PPD_CTD.txt")
        else:
            log("Cannot find CTD file for id: " + id + " in folder: " + folder, 3)
            return False
        log("Reading CTD data from: " + file, 3)
        try:
            df = pd.read_csv(file, sep="\t", header=5)
            if len(df) < 10:
                log("No data for profile.", 3)
                return False
            for column in ["Date (dd/mm/yy)", "Time (hh:mm:ss.sss)"]:
                if column in df.columns:
                    df = df.drop([column], axis=1)
            df["cond"] = df["Conductivity"] * 10000
            df.rename(columns={'Temperature': 'temp', "Timestamp (s)": "time", "Pressure": "press"}, inplace=True)
            beta_s = 0.807e-3  # Haline contraction coefficient
            df["cond20"] = (1.684 - 0.04645 * df["temp"] + 0.000602 * (df["temp"]) ** 2) * df["cond"]
            df["sal"] = 0.874e-3 * df["cond20"]
            df["rho0"] = 999.84298 + 1e-3 * (
                    65.4891 * df["temp"] - 8.56272 * (df["temp"]) ** 2 + 0.059385 * (df["temp"]) ** 3)
            df["rho"] = df["rho0"] * (1 + beta_s * df["sal"])
            df["mrho"] = df["rho"].cumsum() / (df.index + 1)
            df["depth"] = 10000 * df["press"] / (df["mrho"] * 9.81) * 1.019716

            for variable in self.variables:
                self.data[variable] = np.array(df[variable])
            log("Successfully read data", 3)
        except:
            log("Failed to parse data")
            return False
        return True


class process_DO(thetis):
    def __init__(self, *args, **kwargs):
        super(process_DO, self).__init__(*args, **kwargs)

        self.general_attributes["title"] = "Lexplore Thetis Dissolved Oxygen"

        self.dimensions = {
            'time': {'dim_name': 'time', 'dim_size': None}
        }
        self.variables = {
            'time': {'var_name': 'time', 'dim': ('time',), 'unit': 'seconds since 1970-01-01 00:00:00',
                     'longname': 'time', "min": 1514764800, "max": datetime.now().timestamp()},
            'do': {'var_name': 'do', 'dim': ('time',), 'unit': 'mg/L', 'longname': 'dissolvedoxygen', "min": 0,
                   "max": 20},
            'temp': {'var_name': 'temp', 'dim': ('time',), 'unit': 'degC', 'longname': 'temperature', "min": 2,
                     "max": 40},
            'press': {'var_name': 'press', 'dim': ('time',), 'unit': 'dbar', 'longname': 'pressure', "min": 0,
                      "max": 100},
            'do_at_sat': {'var_name': 'do_at_sat', 'dim': ('time',), 'unit': 'mg/L', 'longname': 'oxygensaturation',
                          "min": 0, "max": 20},
            'dosat': {'var_name': 'dosat', 'dim': ('time',), 'unit': '%sat',
                      'longname': 'oxygen relative to saturation', "min": 0, "max": 300},
            'depth': {'var_name': 'depth', 'dim': ('time',), 'unit': 'm', 'longname': 'depth', "min": 0, "max": 100}
        }
        self.offset = 0.1
        self.type = "DO"
        self.grid = ["do", "dosat"]

    def read_data(self, id, folder, ctd):
        self.id = id
        self.folder = folder
        if os.path.isfile(os.path.join(folder, str(id) + "_PPB_DO.txt")):
            file = os.path.join(folder, str(id) + "_PPB_DO.txt")
        elif os.path.isfile(os.path.join(folder, str(id) + "_PPD_DO.txt")):
            file = os.path.join(folder, str(id) + "_PPD_DO.txt")
        else:
            log("Cannot find DO file for id: " + id + " in folder: " + folder, 3)
            return False
        log("Reading DO data from: " + file, 3)
        try:
            df = pd.read_csv(file, sep="\t", header=5)

            if len(df) < 10:
                log("No data for profile.", 3)
                return False
            df[['raw_phase_delay', 'raw_thermistor_voltage', 'DO', 'Temperature']] = df['Data'].str.split(',',
                                                                                                          expand=True)
            df = df.drop(['Data', 'raw_phase_delay', 'raw_thermistor_voltage'], axis=1)
            df[['DO', 'Temperature']] = df[['DO', 'Temperature']].apply(pd.to_numeric, errors='coerce', axis=1)
            df.rename(columns={'Timestamp (s)': 'time', "DO": "do", "Depth (dbar)": "press", "Temperature": "temp"},
                      inplace=True)
            dt = "Depth Timestamp (s)"
            if dt not in df.columns:
                dt = "time"
            df["depth"] = np.interp(np.array(df[dt]), ctd["time"], ctd["depth"]) + self.offset
            df["sal"] = np.interp(np.array(df[dt]), ctd["time"], ctd["sal"])
            df["do_at_sat"] = oxygen_saturation(np.array(df["temp"]), np.array(df["sal"]))  # [ mg/L]
            df["do"] = df["do"] * 1.42903  # from ml/L to mg/L
            df["dosat"] = df["do"] / df["do_at_sat"] * 100

            for variable in self.variables:
                self.data[variable] = np.array(df[variable])
            log("Successfully read data", 3)
        except:
            print(sys.exc_info())
            log("Failed to parse data")
            return False
        return True


class process_PAR(thetis):
    def __init__(self, *args, **kwargs):
        super(process_PAR, self).__init__(*args, **kwargs)

        self.general_attributes["title"] = "Lexplore Thetis Photosynthetically Active Radiation"

        self.dimensions = {
            'time': {'dim_name': 'time', 'dim_size': None}
        }
        self.variables = {
            'time': {'var_name': 'time', 'dim': ('time',), 'unit': 'seconds since 1970-01-01 00:00:00',
                     'longname': 'time', "min": 1514764800, "max": datetime.now().timestamp()},
            'press': {'var_name': 'press', 'dim': ('time',), 'unit': 'dbar', 'longname': 'pressure', "min": 0,
                      "max": 150},
            'depth': {'var_name': 'depth', 'dim': ('time',), 'unit': 'm', 'longname': 'depth', "min": 0, "max": 100},
            'par': {'var_name': 'par', 'dim': ('time',), 'unit': 'μmol/m²/s',
                    'longname': 'photosynthetically active radiation', "min": 0, "max": 10000},
        }
        self.offset = -0.35
        self.type = "PAR"
        self.grid = ["par"]

    def read_data(self, id, folder, ctd):
        self.id = id
        self.folder = folder
        if os.path.isfile(os.path.join(folder, str(id) + "_PPB_PARS.txt")):
            file = os.path.join(folder, str(id) + "_PPB_PARS.txt")
        elif os.path.isfile(os.path.join(folder, str(id) + "_PPD_PARS.txt")):
            file = os.path.join(folder, str(id) + "_PPD_PARS.txt")
        else:
            log("Cannot find PAR file for id: " + id + " in folder: " + folder, 3)
            return False
        log("Reading PAR data from: " + file, 3)
        types = [
            "TimestampsDateddmmyyDepthdbarData",
            "TimestampsDepthdbarDepthTimestampsData",
            "TimestampsDateddmmyyTimehhmmsssssDepthdbarData"
        ]

        try:
            with open(file) as f:
                index_begin = 0
                index_end = 0
                for idx, line in enumerate(f):
                    if "Timestamp" in line:
                        if re.sub(r'[^A-Za-z]', '', line) in types:
                            type = types.index(re.sub(r'[^A-Za-z]', '', line))
                        else:
                            log("Unrecognised file type format: " + file)
                            return False
                    if 'mvs 1' in line:
                        index_begin = idx + 1
                    if 'mvs 0' in line:
                        index_end = idx + 1
                if index_end == 0:
                    index_end = idx + 1
            index_skip = np.concatenate((np.arange(0, index_begin, 1),
                                         np.arange(index_end - 1, idx + 1, 1)))

            df = pd.read_csv(file, sep="\t", skiprows=index_skip, header=None)

            if len(df) < 10:
                log("No data for profile.", 3)
                return False

            if type == 0:
                df.columns = ["time", "Date", "press", "Date Short", "Time Short", "par"]
                df["Depth Timestamp (s)"] = df["time"]
            elif type == 1:
                df.columns = ["time", "press", "Depth Timestamp (s)", "Date", "Hour", "par"]
            elif type == 2:
                df.columns = ["time", "Date (dd/mm/yy)", "Time (hh:mm:ss.sss)", "press", "Date", "Hour", "par"]
                df["Depth Timestamp (s)"] = df["time"]

            df["depth"] = np.interp(np.array(df["Depth Timestamp (s)"]), ctd["time"], ctd["depth"]) + self.offset
            df['par'] = 1.3 * 10 ** ((df['par'] - 4415) / 2892)  # Convert to μmol photons/m²/s

            for variable in self.variables:
                self.data[variable] = np.array(df[variable])
            log("Successfully read data", 3)
        except:
            print(sys.exc_info())
            log("Failed to parse data")
            return False
        return True


class process_TRIP1(thetis):
    def __init__(self, *args, **kwargs):
        super(process_TRIP1, self).__init__(*args, **kwargs)

        self.general_attributes["title"] = "Lexplore Thetis TRIP1"

        self.dimensions = {
            'time': {'dim_name': 'time', 'dim_size': None}
        }
        self.variables = {
            'time': {'var_name': 'time', 'dim': ('time',), 'unit': 'seconds since 1970-01-01 00:00:00',
                     'longname': 'time', "min": 1514764800, "max": datetime.now().timestamp()},
            'press': {'var_name': 'press', 'dim': ('time',), 'unit': 'dbar', 'longname': 'pressure', "min": 0,
                      "max": 100},
            'depth': {'var_name': 'depth', 'dim': ('time',), 'unit': 'm', 'longname': 'depth', "min": 0, "max": 100},
            'bb440': {'var_name': 'bb440', 'dim': ('time',), 'unit': 'm-1', 'longname': 'backscattering at 440 nm',
                      "min": 0, "max": 0.5},
            'bb532': {'var_name': 'bb532', 'dim': ('time',), 'unit': 'm-1', 'longname': 'backscattering at 532 nm',
                      "min": 0, "max": 0.5},
            'bb630': {'var_name': 'bb630', 'dim': ('time',), 'unit': 'm-1', 'longname': 'backscattering at 630 nm',
                      "min": 0, "max": 0.5},
        }
        self.offset = 0.069
        self.type = "TRIP1"
        self.grid = ["bb440", "bb532", "bb630"]

    def read_data(self, id, folder, ctd):
        self.id = id
        self.folder = folder
        if os.path.isfile(os.path.join(folder, str(id) + "_PPB_TRIP1.txt")):
            file = os.path.join(folder, str(id) + "_PPB_TRIP1.txt")
        elif os.path.isfile(os.path.join(folder, str(id) + "_PPD_TRIP1.txt")):
            file = os.path.join(folder, str(id) + "_PPD_TRIP1.txt")
        else:
            log("Cannot find TRIP1 file for id: " + id + " in folder: " + folder, 3)
            return False
        log("Reading TRIP1 data from: " + file, 3)

        types = [
            "TimestampsDateddmmyyDepthdbarData",
            "TimestampsDepthdbarDepthTimestampsData",
            "TimestampsDateddmmyyTimehhmmsssssDepthdbarData"
        ]

        try:
            with open(file) as f:
                index_begin = 0
                index_end = 0
                for idx, line in enumerate(f):
                    if "Timestamp" in line:
                        if re.sub(r'[^A-Za-z]', '', line) in types:
                            type = types.index(re.sub(r'[^A-Za-z]', '', line))
                        else:
                            log("Unrecognised file type format: " + file)
                            return False
                    if len(line.split("\t")) >= 12 and index_begin == 0:
                        index_begin = idx
                    if 'mvs 0' in line:
                        index_end = idx + 1
                if index_end == 0:
                    index_end = idx + 1
            index_skip = np.concatenate((np.arange(0, index_begin, 1),
                                         np.arange(index_end - 1, idx + 1, 1)))

            df = pd.read_csv(file, sep="\t", skiprows=index_skip, header=None)

            if len(df) < 10:
                log("No data for profile.", 3)
                return False

            if type == 0:
                df.columns = ["time", "Date", "press", 'date_extract', 'time_extract',
                              'channel_1', 'bb440', 'channel_2', 'bb532', 'channel_3', 'bb630', "something"]
            elif type == 1:
                df.columns = ["time", "press", "Depth Timestamp (s)", 'date_extract', 'time_extract',
                              'channel_1', 'bb440', 'channel_2', 'bb532', 'channel_3', 'bb630', 'something']
            elif type == 2:
                df.columns = ["time", "Date", "Time", 'press', 'date_extract', 'time_extract',
                              'channel_1', 'bb440', 'channel_2', 'bb532', 'channel_3', 'bb630', "something"]

            df[['bb440', 'bb532', 'bb630']] = df[['bb440', 'bb532', 'bb630']].apply(pd.to_numeric, errors='coerce',
                                                                                    axis=1)

            df = df[df['bb630'].notna()]
            depth = []
            dt = "Depth Timestamp (s)"
            if dt not in df.columns:
                dt = "time"
            df["depth"] = np.interp(np.array(df[dt]), ctd["time"], ctd["depth"]) + self.offset

            if df["time"].values[1] < 1593561600:  # corresponds to "2020-07-01 UTC"
                # Linear interpolation to account for gradual shift in calibration coeffs
                d2 = 1592172000  # "2020-06-15 UTC"
                d1 = 1538344800  # "2018-10-01 UTC"
                c1_b440 = 1.266E-04
                c2_b440 = 2.045E-04
                c1_b532 = 1.160E-04
                c2_b532 = 1.438E-04
                c1_b630 = 8.717E-05
                c2_b630 = 8.630E-05
                mean_timestep = df["time"].sort_values().mean()
                b440_coeff = (c2_b440 - c1_b440) / (d2 - d1) * mean_timestep + c1_b440 - (c2_b440 - c1_b440) / (
                        d2 - d1) * d1
                b532_coeff = (c2_b532 - c1_b532) / (d2 - d1) * mean_timestep + c1_b532 - (c2_b532 - c1_b532) / (
                        d2 - d1) * d1
                b630_coeff = (c2_b630 - c1_b630) / (d2 - d1) * mean_timestep + c1_b630 - (c2_b630 - c1_b630) / (
                        d2 - d1) * d1
            else:
                b440_coeff = 1.757e-4
                b532_coeff = 1.258e-4
                b630_coeff = 8.658e-5

            # apply calibration coefficients to obtain scattering at 117°
            df["bb440"] = b440_coeff * (df["bb440"] - 50)
            df["bb532"] = b532_coeff * (df["bb532"] - 47)
            df["bb630"] = b630_coeff * (df["bb630"] - 49)

            # computing water volume scattering at 117°
            delta = 0.09
            theta = 117 * math.pi / 180
            beta_w_440 = 1.38 * (440 / 500) ** (-4.32) * (1 + 0.3 * 0.25 / 37) * 1e-4 * (
                    1 + (math.cos(theta)) ** 2 * (1 - delta) / (1 + delta))  # Morel 1974 in Boss et al., 2004
            beta_w_532 = 1.38 * (532 / 500) ** (-4.32) * (1 + 0.3 * 0.25 / 37) * 1e-4 * (
                    1 + (math.cos(theta)) ** 2 * (1 - delta) / (1 + delta))
            beta_w_630 = 1.38 * (630 / 500) ** (-4.32) * (1 + 0.3 * 0.25 / 37) * 1e-4 * (
                    1 + (math.cos(theta)) ** 2 * (1 - delta) / (1 + delta))

            # computing backscattering coefficients
            df["bb440"] = 2 * math.pi * 1.1 * (df["bb440"] - beta_w_440)  # Boss and Pegau, 2001 & Boss et al., 2004
            df["bb532"] = 2 * math.pi * 1.1 * (df["bb532"] - beta_w_532)
            df["bb630"] = 2 * math.pi * 1.1 * (df["bb630"] - beta_w_630)

            for variable in self.variables:
                self.data[variable] = np.array(df[variable])
            log("Successfully read data", 3)
        except:
            print(sys.exc_info())
            log("Failed to parse data")
            return False
        return True


class process_TRIP2(thetis):
    def __init__(self, *args, **kwargs):
        super(process_TRIP2, self).__init__(*args, **kwargs)

        self.general_attributes["title"] = "Lexplore Thetis TRIP2"

        self.dimensions = {
            'time': {'dim_name': 'time', 'dim_size': None}
        }
        self.variables = {
            'time': {'var_name': 'time', 'dim': ('time',), 'unit': 'seconds since 1970-01-01 00:00:00',
                     'longname': 'time', "min": 1514764800, "max": datetime.now().timestamp()},
            'press': {'var_name': 'press', 'dim': ('time',), 'unit': 'dbar', 'longname': 'pressure', "min": 0,
                      "max": 100},
            'depth': {'var_name': 'depth', 'dim': ('time',), 'unit': 'm', 'longname': 'depth', "min": 0, "max": 100},
            'bb700': {'var_name': 'bb700', 'dim': ('time',), 'unit': 'm-1', 'longname': 'backscattering at 700 nm',
                      "min": 0, "max": 0.5},
            'chla': {'var_name': 'chla', 'dim': ('time',), 'unit': 'μg/L', 'longname': 'Chlorophyll a', "min": 0,
                     "max": 100},
            'cdom': {'var_name': 'cdom', 'dim': ('time',), 'unit': 'ppb',
                     'longname': 'Chloromorphic Dissolved Organic Matter', "min": 0, "max": 100},
        }
        self.offset = 0.069
        self.type = "TRIP2"
        self.grid = ["bb700", "chla", "cdom"]

    def read_data(self, id, folder, ctd):
        self.id = id
        self.folder = folder
        if os.path.isfile(os.path.join(folder, str(id) + "_PPB_TRIP2.txt")):
            file = os.path.join(folder, str(id) + "_PPB_TRIP2.txt")
        elif os.path.isfile(os.path.join(folder, str(id) + "_PPD_TRIP2.txt")):
            file = os.path.join(folder, str(id) + "_PPD_TRIP2.txt")
        else:
            log("Cannot find TRIP2 file for id: " + id + " in folder: " + folder, 3)
            return False
        log("Reading TRIP2 data from: " + file, 3)

        types = [
            "TimestampsDateddmmyyDepthdbarData",
            "TimestampsDepthdbarDepthTimestampsData",
            "TimestampsDateddmmyyTimehhmmsssssDepthdbarData"
        ]

        try:
            with open(file) as f:
                index_begin = 0;
                index_end = 0;
                for idx, line in enumerate(f):
                    if "Timestamp" in line:
                        if re.sub(r'[^A-Za-z]', '', line) in types:
                            type = types.index(re.sub(r'[^A-Za-z]', '', line))
                        else:
                            log("Unrecognised file type format: " + file)
                            return False
                    if len(line.split("\t")) >= 12 and index_begin == 0:
                        index_begin = idx
                    if 'mvs 0' in line:
                        index_end = idx + 1
                if index_end == 0:
                    index_end = idx + 1
            index_skip = np.concatenate((np.arange(0, index_begin, 1),
                                         np.arange(index_end - 1, idx + 1, 1)))

            df = pd.read_csv(file, sep="\t", skiprows=index_skip, header=None)

            if len(df) < 10:
                log("No data for profile.", 3)
                return False

            elif type == 0:
                df.columns = ["time", "Date", "press", 'date_extract', 'time_extract',
                              'channel_1', 'bb700', 'channel_2', 'chla', 'channel_3', 'cdom', "something"]
            elif type == 1:
                df.columns = ["time", "press", "Depth Timestamp (s)", 'date_extract', 'time_extract',
                              'channel_1', 'bb700', 'channel_2', 'chla', 'channel_3', 'cdom', "something"]
            elif type == 2:
                df.columns = ["time", "Date", "Time", 'press', 'date_extract', 'time_extract',
                              'channel_1', 'bb700', 'channel_2', 'chla', 'channel_3', 'cdom', "something"]

            df[['bb700', 'chla', 'cdom']] = df[['bb700', 'chla', 'cdom']].apply(pd.to_numeric, errors='coerce',
                                                                                axis=1)
            df = df[df['chla'].notna()]
            depth = []
            dt = "Depth Timestamp (s)"
            if dt not in df.columns:
                dt = "time"
            df["depth"] = np.interp(np.array(df[dt]), ctd["time"], ctd["depth"]) + self.offset

            # get calibration coefficients
            if df["time"].values[1] < 1593561600:  # corresponds to "2020-07-01 UTC"
                # Linear interpolation to account for gradual shift in calibration coeffs
                mean_timestep = df["time"].sort_values().mean()
                d2 = 1592172000  # "2020-06-15 UTC"
                d1 = 1538344800  # "2018-10-01 UTC"

                c1_b700 = 1.980e-06
                c2_b700 = 2.296e-06
                c1_CHLa = 0.0123
                c2_CHLa = 0.0146
                c1_CDOM = 0.0907
                c2_CDOM = 0.0887

                b700_coeff = (c2_b700 - c1_b700) / (d2 - d1) * mean_timestep + c1_b700 - (c2_b700 - c1_b700) / (
                        d2 - d1) * d1
                CHLa_coeff = (c2_CHLa - c1_CHLa) / (d2 - d1) * mean_timestep + c1_CHLa - (c2_CHLa - c1_CHLa) / (
                        d2 - d1) * d1
                CDOM_coeff = (c2_CDOM - c1_CDOM) / (d2 - d1) * mean_timestep + c1_CDOM - (c2_CDOM - c1_CDOM) / (
                        d2 - d1) * d1
            else:
                b700_coeff = 2.207e-6
                CHLa_coeff = 0.0107
                CDOM_coeff = 0.0797

            # apply calibration coefficients to obtain scattering at 117°
            df["bb700"] = b700_coeff * (df["bb700"] - 49)
            df["chla"] = CHLa_coeff * (df["chla"] - 47)
            df["cdom"] = CDOM_coeff * (df["cdom"] - 50)

            # CHL correction based on ACS data
            df["chla"] = df["chla"] * 1.97 + 0.44

            # computing water volume scattering at 117°
            delta = 0.09
            theta = 117 * math.pi / 180
            beta_w_700 = 1.38 * (700 / 500) ** (-4.32) * (1 + 0.3 * 0.25 / 37) * 1e-4 * (
                    1 + (math.cos(theta)) ** 2 * (1 - delta) / (1 + delta))  # Morel 1974 in Boss et al., 2004

            # computing backscattering coefficients
            df["bb700"] = 2 * math.pi * 1.1 * (df["bb700"] - beta_w_700)  # Boss and Pegau, 2001 & Boss et al., 2004

            for variable in self.variables:
                self.data[variable] = np.array(df[variable])
            log("Successfully read data", 3)
        except:
            print(sys.exc_info())
            log("Failed to parse data")
            return False
        return True


class process_ACS(thetis):
    def __init__(self, *args, **kwargs):
        super(process_ACS, self).__init__(*args, **kwargs)

        self.general_attributes["title"] = "Lexplore Thetis ACS"

        self.dimensions = {
            'depth': {'dim_name': 'depth', 'dim_size': None},
            'wavelength': {'dim_name': 'wavelength', 'dim_size': None}
        }
        self.variables = {
            'time': {'var_name': 'time', 'dim': ('depth',), 'unit': 'seconds since 1970-01-01 00:00:00',
                     'longname': 'time', "min": 1514764800, "max": datetime.now().timestamp()},
            'wavelength': {'var_name': 'wavelength', 'dim': ('wavelength',), 'unit': 'nm', 'longname': 'wavelength'},
            'depth': {'var_name': 'depth', 'dim': ('depth',), 'unit': 'm', 'longname': 'depth', "min": 0, "max": 100},
            'a': {'var_name': 'a', 'dim': ('depth', 'wavelength'), 'unit': 'm-1',
                  'longname': 'Hyperspectral absorption', "min": 0, "max": 3},
            'b': {'var_name': 'b', 'dim': ('depth', 'wavelength'), 'unit': 'm-1',
                  'longname': 'Hyperspectral scattering', "min": 0, "max": 20},
            'c': {'var_name': 'c', 'dim': ('depth', 'wavelength'), 'unit': 'm-1',
                  'longname': 'Hyperspectral attenuation', "min": 0, "max": 20},
            'a700': {'var_name': 'a700', 'dim': ('depth',), 'unit': 'm-1',
                     'longname': 'Hyperspectral absorption at 700nm', "min": 0, "max": 3},
            'b700': {'var_name': 'b700', 'dim': ('depth',), 'unit': 'm-1',
                     'longname': 'Hyperspectral scattering at 700nm', "min": 0, "max": 20},
            'c700': {'var_name': 'c700', 'dim': ('depth',), 'unit': 'm-1',
                     'longname': 'Hyperspectral attenuation at 700nm', "min": 0, "max": 20},
            'aLH676': {'var_name': 'aLH676', 'dim': ('depth',), 'unit': 'm-1',
                       'longname': 'Absorption line height at 676 nm', "min": 0, "max": 5},
            'Sk': {'var_name': 'Sk', 'dim': ('depth',), 'unit': 'm-1',
                   'longname': 'Spectral attenuation slope', "min": 0, "max": 5},
        }
        self.offset = 0.22
        self.type = "ACS"
        # MODIFIED
        # 1D variables
        self.grid_depth = ["a700", "b700", "c700", "aLH676", "Sk"]
        
        # hyperspectral variables
        self.grid_hyper = ["a", "b", "c"]

    def read_calibration_data(self, calibration_dir, time):
        if os.path.exists(calibration_dir):
            if os.path.isfile(os.path.join(calibration_dir, "calibration.json")):
                with open(os.path.join(calibration_dir, "calibration.json"), "r") as f:
                    calibration = json.load(f)
                    try:
                        read = False
                        for calib in calibration["ACS"]:
                            if (time > calib["start"] or calib["start"] == False) and (
                                    time <= calib["end"] or calib["end"] == False):
                                # Read calibration
                                tcal, ical, t_bins, landa_C, landa_A, C0, A0, C_corr, A_corr = read_acs_calibration_data(
                                    os.path.join(calibration_dir, calib["tab_cal_ACS"]))
                                tab_Corr = pd.read_csv(os.path.join(calibration_dir, calib["tab_Corr"]), sep=" ",
                                                       header=[0, 1])
                                read = True
                        if not read:
                            log("Failed to find calibration files for time: " + str(time), 3)
                            return False
                        log("Successfully read calibration files", 3)
                        return {"tcal": tcal, "ical": ical, "t_bins": t_bins,
                                "landa_C": landa_C, "landa_A": landa_A, "C0": C0, "A0": A0,
                                "C_corr": C_corr, "A_corr": A_corr,
                                "tab_Corr": tab_Corr}
                    except:
                        print(print(sys.exc_info()))
                        log("Failed to load calibration files for ACS", 3)
                        return False
            else:
                log("Cannot find calibration master file calibration.json for ACS", 3)
                return False

        else:
            log("Cannot find calibration files for ACS", 3)
            return False

    def read_data(self, id, folder, calibration_dir, ctd, bin=0.125):
        self.id = id
        self.folder = folder
        if os.path.isfile(os.path.join(folder, str(id) + "_ACS_ACS.txt")):
            file = os.path.join(folder, str(id) + "_ACS_ACS.txt")
        elif os.path.isfile(os.path.join(folder, str(id) + "_ACD_ACS.txt")):
            file = os.path.join(folder, str(id) + "_ACD_ACS.txt")
        else:
            log("Cannot find ACS file for id: " + id + " in folder: " + folder, 3)
            return False
        time = ctd["time"][0]
        calibration = self.read_calibration_data(calibration_dir, time)
        if calibration == False:
            return False
        log("Reading ACS data from: " + file, 3)

        try:
            with open(file) as f:
                index_begin = 0
                for idx, line in enumerate(f):
                    if "Timestamp (s)" in line:
                        idx_count = line.split("\t").index("C ref dark")
                    if len(line.split("\t")) > 300:
                        index_begin = idx
                        break

            df = pd.read_csv(file, sep="\t", skiprows=index_begin, header=None, on_bad_lines="skip", engine="python")
            df.dropna(subset=[4], inplace=True)

            df["depth"] = np.interp(np.array(df[0]), ctd["time"], ctd["depth"]) + self.offset
            df["temp"] = np.interp(np.array(df[0]), ctd["time"], ctd["temp"])
            df["sal"] = np.interp(np.array(df[0]), ctd["time"], ctd["sal"])

            df = df[df['depth'] < ctd["depth"].max()]
            df = df[df['depth'] > ctd["depth"].min()]

            # simplifying variable names from calibration extraction
            landa_A = np.array(calibration["landa_A"])
            A0 = np.array(calibration["A0"])
            A_corr = np.array(calibration["A_corr"])
            t_bins = np.array(calibration["t_bins"])
            landa_C = np.array(calibration["landa_C"])
            C0 = np.array(calibration["C0"])
            C_corr = np.array(calibration["C_corr"])
            tcal = np.array(calibration["tcal"])

            n_wl = len(landa_C)

            ACS = df.to_numpy()
            C_ref_dark = ACS[:, idx_count].astype('float64')
            C_ref = ACS[:, idx_count + 1:idx_count + 1 + n_wl].astype('float64')
            C_sig_dark = ACS[:, idx_count + 1 + n_wl].astype('float64')
            C_sig = ACS[:, idx_count + 2 + n_wl:idx_count + 2 + 2 * n_wl].astype('float64')

            A_ref_dark = ACS[:, idx_count + 2 + 2 * n_wl].astype('float64')
            A_ref = ACS[:, idx_count + 3 + 2 * n_wl:idx_count + 3 + 3 * n_wl].astype('float64')
            A_sig_dark = ACS[:, idx_count + 3 + 3 * n_wl].astype('float64')
            A_sig = ACS[:, idx_count + 4 + 3 * n_wl:idx_count + 4 + 4 * n_wl].astype('float64')
            ext_tc = ACS[:, idx_count + 4 + 4 * n_wl].astype('float64')
            int_tc = ACS[:, idx_count + 5 + 4 * n_wl].astype('float64')

            # ext_t = -7.1023317e-13 * ext_tc ** 3 + 7.09341920e-8 * ext_tc ** 2 - 3.87065673e-3 * ext_tc + 95.8241397
            res = 10000 * (5 * int_tc / 65535) / (4.516 - (5 * int_tc / 65535))
            int_t = 1 / (0.00093135 + 0.000221631 * np.log(res) + 0.000000125741 * np.log(res) ** 3) - 273.15

            mat_A = counts_to_spectra(A_ref, A_sig, int_t, landa_A, A0, A_corr, t_bins)
            mat_C = counts_to_spectra(C_ref, C_sig, int_t, landa_C, C0, C_corr, t_bins)

            mat_A = temperature_salinity_correction(mat_A, landa_A, calibration["tab_Corr"], np.array(df["sal"]),
                                                    np.array(df["temp"]), tcal, "A")
            mat_C = temperature_salinity_correction(mat_C, landa_C, calibration["tab_Corr"], np.array(df["sal"]),
                                                    np.array(df["temp"]), tcal, "C")

            mat_A = scattering_correction(mat_A, mat_C, landa_C, landa_A) # based on Stockley et al., 2017, Optics Express, model PROP-RR

            # Only unique depths
            depth, unique = np.unique(np.array(df["depth"]), return_index=True)
            time = np.array(df[0])[unique]
            mat_A = mat_A[unique]
            mat_C = mat_C[unique]

            # Sort base on depth
            sort = np.argsort(depth)
            depth = depth[sort]
            time = time[sort]
            mat_A = mat_A[sort]
            mat_C = mat_C[sort]
            mat_B = mat_C - mat_A

            self.data["wavelength"] = landa_A



            ### smooth A and C vertically
            """
            df_A = pd.DataFrame(np.array(mat_A), index=depth, columns=landa_A)
            df_C = pd.DataFrame(np.array(mat_C), index=depth, columns=landa_C)
            df_A = smooth_acs_vertically(df_A)
            df_C = smooth_acs_vertically(df_C)
            mat_A = df_A.to_numpy()
            mat_C = df_C.to_numpy()
            mat_B = mat_C - mat_A
            """

            # Binning option
            """
            self.data["depth"] = np.arange(2, 50, bin)
            self.data["a"] = bin_array(mat_A, depth, self.data["depth"])
            self.data["b"] = bin_array(mat_B, depth, self.data["depth"])
            self.data["c"] = bin_array(mat_C, depth, self.data["depth"])
            self.data["time"] = bin_array(time, depth, self.data["depth"])
            """

            # Not binning
            self.data["depth"] = depth
            self.data["a"] = mat_A
            self.data["b"] = mat_B
            self.data["c"] = mat_C
            self.data["time"] = time

            w700 = find_closest_index(landa_A, 700)
            self.data["a700"] = self.data["a"][:, w700]
            self.data["b700"] = self.data["b"][:, w700]
            self.data["c700"] = self.data["c"][:, w700]

            self.data["aLH676"] = absorption_line_height(landa_A, self.data["a"], 676)
            self.data["Sk"] = spectral_attenuation_slope(landa_C, self.data["c"])

            if len(time) < 20:
                log("Erroneous profile", 3)
                return False

            log("Successfully read data", 3)
        except:
            print(sys.exc_info())
            log("Failed to parse data")
            return False
        return True

    def quality_flags(self):
        variables = self.variables.copy().items()
        for key, values in variables:
            name = key + "_qual"
            self.variables[name] = {'var_name': name, 'dim': values["dim"],
                                    'unit': '0 = nothing to report, 1 = more investigation',
                                    'longname': name, }
            qa_data = np.zeros_like(self.data[key])
            isnt_numeric = np.vectorize(isnt_number, otypes=[bool])
            qa_data[(isnt_numeric(self.data[key]))] = 1
            if "min" in values:
                qa_data[(self.data[key] < float(values["min"]))] = 1
            if "max" in values:
                qa_data[(self.data[key] > float(values["max"]))] = 1

            if key in ["a", "b", "c", "a700", "b700", "c700", "aLH676", "Sk"]:
                qa_data[despike(self.data[key], prominence=1)] = 1

            self.data[name] = np.array(qa_data)


class process_OCR1(thetis):
    def __init__(self, *args, **kwargs):
        super(process_OCR1, self).__init__(*args, **kwargs)

        self.general_attributes["title"] = "Lexplore Thetis OCR1"

        self.dimensions = {
            'depth': {'dim_name': 'depth', 'dim_size': None},
            'wavelength': {'dim_name': 'wavelength', 'dim_size': None}
        }
        self.variables = {
            'time': {'var_name': 'time', 'dim': ('depth',), 'unit': 'seconds since 1970-01-01 00:00:00',
                     'longname': 'time', "min": 1514764800, "max": datetime.now().timestamp()},
            'wavelength': {'var_name': 'wavelength', 'dim': ('wavelength',), 'unit': 'nm', 'longname': 'wavelength'},
            'depth': {'var_name': 'depth', 'dim': ('depth',), 'unit': 'm', 'longname': 'depth', "min": 0, "max": 100},
            'Ed': {'var_name': 'Ed', 'dim': ('depth', 'wavelength'), 'unit': 'μW cm-2 nm-1',
                   'longname': 'Hyperspectral downwelling irradiance', "min": 0, "max": 200},
            'Ed0': {'var_name': 'Ed0', 'dim': ('wavelength',), 'unit': 'μW cm-2 nm-1',
                    'longname': 'Surface hyperspectral downwelling irradiance', "min": 0, "max": 200},
            'kd_Ed': {'var_name': 'kd_Ed', 'dim': ('wavelength',), 'unit': 'm-1',
                      'longname': 'Spectral light attenuation coefficient', "min": 0, "max": 1},
        }
        self.offset = -0.15
        self.type = "OCR1"
        self.grid = ["Ed0", "kd_Ed"]

    def read_data(self, id, folder, calibration_dir, ctd):
        self.id = id
        self.folder = folder

        if os.path.isfile(os.path.join(folder, str(id) + "_PPB_OCR1.txt")):
            file = os.path.join(folder, str(id) + "_PPB_OCR1.txt")
        elif os.path.isfile(os.path.join(folder, str(id) + "_PPD_OCR1.txt")):
            file = os.path.join(folder, str(id) + "_PPD_OCR1.txt")
        else:
            log("Cannot find OCR file for id: " + id + " in folder: " + folder, 3)
            return False

        log("Reading OCR data from: " + file, 3)

        try:
            # Process OCR1
            with open(file) as f:
                index_begin = 0
                for idx, line in enumerate(f):
                    if len(line.split("\t")) > 100:
                        index_begin = idx
                        break
            df = pd.read_csv(file, sep="\t+|\t+\t+", skiprows=index_begin, engine='python')
            df = df[df.isnull().sum(axis=1) < 20]
            df["depth"] = np.interp(np.array(df["Timestamp (s)"]), ctd["time"], ctd["depth"]) + self.offset

            timestamp = np.array(df["Timestamp (s)"])
            depth = np.interp(timestamp, ctd["time"], ctd["depth"]) + self.offset
            OCR = np.array(df.loc[:, "Chan 1":"Chan 180"])
            int_time = np.array(df["Integration Time"]) / 1000
            dark_ave = np.array(df["Dark Ave"])
            header = np.array(df["Header"])

            OCR = (OCR.transpose() - dark_ave).transpose()

            ind_light_counts = header == "SATHPE"
            ind_dark_shutter = header == "SATPED"

            OCR_HPE = OCR[ind_light_counts]
            OCR_PED = OCR[ind_dark_shutter]

            # Calibration
            a0, a1, Im, Cint, landa_HPE = read_ocr_calibration_data(os.path.join(calibration_dir, "HPE557A.cal"))
            OCR_HPE = Im * a1 * (OCR_HPE - a0) * Cint / int_time[ind_light_counts][:, None]

            a0, a1, Im, Cint, landa_PED = read_ocr_calibration_data(os.path.join(calibration_dir, "PED557A.cal"))
            OCR_PED = Im * a1 * (OCR_PED - a0) * Cint / int_time[ind_dark_shutter][:, None]

            # Interpolate to fixed time
            OCR_HPE_interp = np.full([len(df), 180], np.nan)
            OCR_PED_interp = np.full([len(df), 180], np.nan)

            for i in range(180):
                if a0[i] > 0:
                    OCR_HPE_interp[:, i] = np.interp(timestamp, timestamp[ind_light_counts], OCR_HPE[:, i])
                    OCR_PED_interp[:, i] = np.interp(timestamp, timestamp[ind_dark_shutter], OCR_PED[:, i])

            Ed = OCR_HPE_interp - np.nan_to_num(OCR_PED_interp)

            minDepth = min(depth)
            Ed0 = np.nanmedian(Ed[depth == minDepth], axis=0)

            # Only unique depths
            depth, unique = np.unique(depth, return_index=True)
            time = timestamp[unique]
            Ed = Ed[unique]

            # Sort base on depth
            sort = np.argsort(depth)
            depth = depth[sort]
            time = time[sort]
            Ed = Ed[sort]

            Ed[0, :] = Ed0

            # Calculate kd_ed
            kd_ed = np.array([np.nan] * len(landa_HPE))
            if np.nanmax(Ed) > 0:
                kd_ed = spectral_light_attenuation_coefficient(landa_HPE, Ed, depth, self.variables["kd_Ed"]["min"],
                                                               self.variables["kd_Ed"]["max"])

            self.data["time"] = time
            self.data["wavelength"] = landa_HPE
            self.data["depth"] = depth
            self.data["Ed"] = Ed
            self.data["Ed0"] = Ed0
            self.data["kd_Ed"] = kd_ed

            if len(time) < 20:
                log("Erroneous profile", 3)
                return False

            log("Successfully read data", 3)
        except:
            print(sys.exc_info())
            log("Failed to parse data")
            return False
        return True


class process_OCR2(thetis):
    def __init__(self, *args, **kwargs):
        super(process_OCR2, self).__init__(*args, **kwargs)

        self.general_attributes["title"] = "Lexplore Thetis OCR2"

        self.dimensions = {
            'depth': {'dim_name': 'depth', 'dim_size': None},
            'wavelength': {'dim_name': 'wavelength', 'dim_size': None}
        }
        self.variables = {
            'time': {'var_name': 'time', 'dim': ('depth',), 'unit': 'seconds since 1970-01-01 00:00:00',
                     'longname': 'time', "min": 1514764800, "max": datetime.now().timestamp()},
            'wavelength': {'var_name': 'wavelength', 'dim': ('wavelength',), 'unit': 'nm', 'longname': 'wavelength'},
            'depth': {'var_name': 'depth', 'dim': ('depth',), 'unit': 'm', 'longname': 'depth', "min": 0, "max": 100},
            'Lu': {'var_name': 'Lu', 'dim': ('depth', 'wavelength'), 'unit': 'μW cm-2 nm-1 sr-1',
                   'longname': 'Hyperspectral upwelling radiance', "min": 0, "max": 5},
            'Lu0': {'var_name': 'Lu0', 'dim': ('wavelength',), 'unit': 'μW cm-2 nm-1 sr-1',
                    'longname': 'Surface hyperspectral upwelling radiance', "min": 0, "max": 5},

        }
        self.offset = -0.15
        self.type = "OCR2"
        self.grid = ["Lu0"]

    def read_data(self, id, folder, calibration_dir, ctd):
        self.id = id
        self.folder = folder

        if os.path.isfile(os.path.join(folder, str(id) + "_PPB_OCR2.txt")):
            file = os.path.join(folder, str(id) + "_PPB_OCR2.txt")
        elif os.path.isfile(os.path.join(folder, str(id) + "_PPD_OCR2.txt")):
            file = os.path.join(folder, str(id) + "_PPD_OCR2.txt")
        else:
            log("Cannot find OCR file for id: " + id + " in folder: " + folder, 3)
            return False

        log("Reading OCR data from: " + file, 3)

        try:
            # Process OCR2
            with open(file, encoding="ISO-8859-1") as f:
                index_begin = 0
                for idx, line in enumerate(f):
                    if len(line.split("\t")) > 100:
                        index_begin = idx
                        break
            df = pd.read_csv(file, sep="\t+|\t+\t+", skiprows=index_begin, engine='python')
            df = df[df.isnull().sum(axis=1) < 20]
            df["depth"] = np.interp(np.array(df["Timestamp (s)"]), ctd["time"], ctd["depth"]) + self.offset

            timestamp = np.array(df["Timestamp (s)"])
            depth = np.interp(timestamp, ctd["time"], ctd["depth"]) + self.offset
            OCR = np.array(df.loc[:, "Chan 1":"Chan 180"])
            int_time = np.array(df["Integration Time"]) / 1000
            dark_ave = np.array(df["Dark Ave"])
            header = np.array(df["Header"])

            OCR = (OCR.transpose() - dark_ave).transpose()

            ind_light_counts = header == "SATHPL"
            ind_dark_shutter = header == "SATPLD"

            OCR_HPL = OCR[ind_light_counts]
            OCR_PLD = OCR[ind_dark_shutter]

            # Calibration
            a0, a1, Im, Cint, landa_HPL = read_ocr_calibration_data(os.path.join(calibration_dir, "HPL441A.cal"))
            OCR_HPL = Im * a1 * (OCR_HPL - a0) * Cint / int_time[ind_light_counts][:, None]

            a0, a1, Im, Cint, landa_PLD = read_ocr_calibration_data(os.path.join(calibration_dir, "PLD441A.cal"))
            OCR_PLD = Im * a1 * (OCR_PLD - a0) * Cint / int_time[ind_dark_shutter][:, None]

            # Interpolate to fixed time
            OCR_HPL_interp = np.full([len(df), 180], np.nan)
            OCR_PLD_interp = np.full([len(df), 180], np.nan)

            for i in range(180):
                if a0[i] > 0:
                    OCR_HPL_interp[:, i] = np.interp(timestamp, timestamp[ind_light_counts], OCR_HPL[:, i])
                    if len(OCR_PLD) > 0:
                        OCR_PLD_interp[:, i] = np.interp(timestamp, timestamp[ind_dark_shutter], OCR_PLD[:, i])

            Lu = OCR_HPL_interp - np.nan_to_num(OCR_PLD_interp)

            minDepth = min(depth)
            Lu0 = np.nanmedian(Lu[depth == minDepth], axis=0)

            # Only unique depths
            depth, unique = np.unique(depth, return_index=True)
            time = timestamp[unique]
            Lu = Lu[unique]

            # Sort base on depth
            sort = np.argsort(depth)
            depth = depth[sort]
            time = time[sort]
            Lu = Lu[sort]

            Lu[0, :] = Lu0

            self.data["time"] = time
            self.data["wavelength"] = landa_HPL
            self.data["depth"] = depth
            self.data["Lu"] = Lu
            self.data["Lu0"] = Lu0

            if len(time) < 20:
                log("Erroneous profile", 3)
                return False

            log("Successfully read data", 3)
        except:
            print(sys.exc_info())
            log("Failed to parse data")
            return False
        return True
