import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy import interpolate
import matplotlib.pyplot as plt
from operator import itemgetter


def parse_args(directory):
    parser = argparse.ArgumentParser()

    parser.add_argument("--directory", "-d",
                        help="Directory containing raw Thetis data files. Defaults to directory in YAML file.")
    parser.add_argument("--id", "-i",
                        help="Specify id of raw data to process from folder. Defaults to all ids in directory.")

    args = parser.parse_args()
    ids = []
    if args.directory:
        if os.path.exists(args.directory):
            directory = args.directory
        else:
            error('Not a valid directory.')
    if args.id:
        ids = [args.id]
    else:
        files = os.listdir(directory)
        for file in files:
            if ".txt" in file:
                id = file.split("_")[0]
                if id not in ids:
                    ids.append(id)
    return directory, ids


def log(str, indent=0, start=False):
    if start:
        out = "\n" + str + "\n"
        with open("log.txt", "w") as file:
            file.write(out + "\n")
    else:
        out = datetime.now().strftime("%H:%M:%S.%f") + (" " * 3 * (indent + 1)) + str
        with open("log.txt", "a") as file:
            file.write(out + "\n")
    print(out)


def error(str):
    out = datetime.now().strftime("%H:%M:%S.%f") + "   ERROR: " + str
    with open("log.txt", "a") as file:
        file.write(out + "\n")
    raise ValueError(str)


def find_closest_index(arr, value):
    return min(range(len(arr)), key=lambda i: abs(arr[i] - value))


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return True


def isnt_number(n):
    try:
        float(n)
    except ValueError:
        return True
    else:
        return False


def position_in_array(arr, value):
    for i in range(len(arr)):
        if value < arr[i]:
            return i
    return len(arr)


def oxygen_saturation(temperature, salinity, altitude=372., mgL_mlL=1.42905, mmHg_mb=0.750061683,
                      mmHg_inHg=25.3970886, standard_pressure_sea_level=29.92126,
                      standard_temperature_sea_level=288.15, g=9.81, air_molar_mass=0.0289644,
                      universal_gas_constant=8.31447):
    # Calculates oxygen saturation from dissolved oxygen (mg/l) according to Garcia-Benson
    baro = (1. / mmHg_mb) * mmHg_inHg * standard_pressure_sea_level * np.exp(
        (-g * air_molar_mass * altitude) / (
                    universal_gas_constant * standard_temperature_sea_level))
    u = 10 ** (8.10765 - 1750.286 / (235 + temperature))
    press_corr = (baro * mmHg_mb - u) / (760 - u)
    Ts = np.log((298.15 - temperature) / (273.15 + temperature))
    lnC = 2.00907 + 3.22014 * Ts + 4.0501 * Ts ** 2 + 4.94457 * Ts ** 3 + -0.256847 * Ts ** 4 + 3.88767 * Ts ** 5 - \
          salinity * (0.00624523 + 0.00737614 * Ts + 0.010341 * Ts ** 2 + 0.00817083 * Ts ** 3) - 4.88682e-07 * salinity ** 2
    O2sat = np.exp(lnC)
    O2sat = O2sat * mgL_mlL * press_corr
    return O2sat

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def despike(array, prominence=0.2):
    out = np.zeros_like(array)
    out = np.array(out, dtype=bool)
    if len(array.shape) > 1:
        for i in range(array.shape[1]):
            peaks, _ = find_peaks(array[:, i], prominence=prominence)
            out[:][peaks] = True
    else:
        peaks, _ = find_peaks(array, prominence=prominence)
        out[peaks] = True
    return out


def counts_to_spectra(ref, sig, t, landa, offset, tab_corr, t_bins):
    ref[ref == 0] = np.nan
    t_calib = np.array(t_bins).astype('float64')
    t_calib = t_calib[~np.isnan(t_calib)]
    mat = np.empty((len(sig), len(landa),))
    mat[:] = np.nan
    t_max = t_calib.max()
    for i in range(len(sig)):
        if t[i] <= t_max:
            T_closest = find_nearest(t_calib, t[i])
            if t[i] - T_closest < 0:
                T_0 = t_calib[find_closest_index(t_calib, t[i])-1]
                T_1 = T_closest
            else:
               T_0 = T_closest
               T_1 = t_calib[find_closest_index(t_calib, t[i])+1]

            ind_T_0 = find_closest_index(t_calib, T_0)
            ind_T_1 = ind_T_0+1
            dT = tab_corr[:, ind_T_0] + (t[i] - T_0) / (T_1 - T_0) * (tab_corr[:, ind_T_1] - tab_corr[:, ind_T_0])
            mat[i, :] = (offset - np.log(sig[i]/ref[i]) / 0.25) - dT
    return mat


def temperature_salinity_correction(mat, landa, tab_Corr, sal, temp, T_ref, type):
    for i in range(len(landa)):
        idx = find_closest_index(np.array(tab_Corr["landa"])[:, 0], landa[i])
        if type == "A":
            psi_Sal = np.array(tab_Corr["psi_Sal_a"])[idx][0]
        elif type == "C":
            psi_Sal = np.array(tab_Corr["psi_Sal_c"])[idx][0]
        mat[:, i] = mat[:, i] - (np.array(tab_Corr["psiT"])[idx, 0] * (temp - T_ref) + psi_Sal * sal)
    return mat


def scattering_correction(mat_A, mat_C, landa_C, landa_A):
    # based on Stockley et al., 2017, Optics Express, model PROP-RR
    # https://doi.org/10.1364/OE.25.0A1139
    vect_landa = np.array(range(400, 721))
    for i in range(len(mat_A)):
        A_approx = np.interp(vect_landa, landa_A, mat_A[i, :])
        C_approx = np.interp(vect_landa, landa_C, mat_C[i, :])
        A_NIR = A_approx[vect_landa >= 690]
        C_NIR = C_approx[vect_landa >= 690]

        A_landa_ref = A_NIR.min()
        C_landa_ref = C_NIR[np.argmin(A_NIR)]

        #epsilon = (A_landa_ref-0.212*np.power(A_landa_ref, 1.135)) * (mat_C[i, :] - mat_A[i, :]) / (C_landa_ref - A_landa_ref)
        epsilon = (A_landa_ref) * (mat_C[i, :] - mat_A[i, :]) / (C_landa_ref - A_landa_ref)  # Fazel updated option
        mat_A[i, :] = mat_A[i, :] - epsilon
    return mat_A


def bin_array(values, arr, bins):
    ind = np.digitize(arr, bins)
    #print(ind)
    #exit()
    return np.array([np.nanmedian(values[ind == i], axis=0) for i in range(1, len(bins)+1)])


def absorption_line_height(landa, arr, wavelength, window=80):
    a = np.empty((len(arr)))
    a[:] = np.nan
    w_inf = wavelength - window / 2
    w_sup = wavelength + window / 2
    ind = (landa > w_inf) & (landa < w_sup)
    for i in range(len(arr)):
        a_inf = arr[i, find_closest_index(landa, w_inf)]
        a_sup = arr[i, find_closest_index(landa, w_sup)]
        spectra_BL = (a_sup - a_inf) / (w_sup - w_inf) * (landa - w_inf) + a_inf
        diff = arr[i][ind] - spectra_BL[ind]
        peak_height = max(0, diff.max())
        # landa_peak = arr[i][ind][np.argmax(diff)]
        a[i] = peak_height
    return a


def spectral_attenuation_slope(landa, arr):
    def fit(landa, k, Sk):
        return k*(landa/532)**(-Sk)
    a = np.empty((len(arr)))
    a[:] = np.nan
    for i in range(len(arr)):
        if np.count_nonzero(np.isnan(arr[i])) < 50:
            iv = landa[(landa > 450) & (landa < 650)]
            dv = arr[i][(landa > 450) & (landa < 650)]
            try:
                a[i] = curve_fit(fit, iv, dv)[0][1]
            except:
                pass
    return a


def spectral_light_attenuation_coefficient(landa, Ed, depth, min, max):
    def fit(depth, par0, kd):
        return par0*np.exp(-kd*depth)
    a = np.empty((len(landa)))
    a[:] = np.nan
    for i in range(len(landa)):
        nn = np.all([[~np.isnan(Ed[:, i])], [Ed[:, i] > min], [Ed[:, i] < max]], axis=0)[0]
        if len(depth[nn]) > 5:
            try:
                a[i] = curve_fit(fit, depth[nn], Ed[:, i][nn], p0=[np.nanmax(Ed), 0.2], bounds=([np.nanmax(Ed), 0], [10*np.nanmax(Ed), 2]))[0][1]
            except:
                pass
    return a


def read_ocr_calibration_data(file):
    data = []
    with open(file) as f:
        lines = list(f)
        for idx, line in enumerate(lines):
            if "uW/cm^2/nm" in line:
                if "OPTIC3" in line:
                    line_arr = line.replace("\n", "").split(" ") + lines[idx + 1].replace("\n", "").split("\t")
                else:
                    line_arr = line.replace("\n", "").split(" ") + [np.nan] * 4
                data.append(line_arr)
    df = pd.DataFrame(data, columns=["letter", "landa", "unit", "no1", "x", "no2", "type", "a0", "a1", "Im", "Cint"])
    a0 = np.array(df["a0"]).astype("float")
    a1 = np.array(df["a1"]).astype("float")
    Im = np.array(df["Im"]).astype("float")
    Cint = np.array(df["Cint"]).astype("float")
    landa = np.array(df["landa"]).astype("float")
    return a0, a1, Im, Cint, landa


def read_acs_calibration_data(file):
    wC_wA_C0_A0 = []
    C_corr = []
    A_corr = []

    with open(file) as f:
        lines = list(f)
        for idx, line in enumerate(lines):
            if "tcal" in line:
                t_cal = line.replace("\n", "").split(" ")
            if "number of temperature bins" in line:
                l_n_t_bins = line.replace("\t", "").split(";")
                n_t_bins = int(l_n_t_bins[0])
            if "temperature bins" in line:
                l_t_bins = line.split("\t")
            if "C and A offset, and C and A temperature correction info" in line:
                char_to_replace = {'C': '',
                               'A': ''}
                for key, value in char_to_replace.items():
                    line = line.replace(key, value)
                line_arr = line.replace("\n", "").split("\t")
            
                wC_wA_C0_A0.append(itemgetter(0,1,3,4)(line_arr))
                C_corr_temp = list(map(float, np.transpose(line_arr[6:6+n_t_bins-1])))
                A_corr_temp = list(map(float, np.transpose(line_arr[7+n_t_bins:7+2*n_t_bins-1])))
                C_corr.append(C_corr_temp)
                A_corr.append(A_corr_temp)
            
    tcal = float(t_cal[1])
    ical = float(t_cal[4])
    t_bins = list(map(float, l_t_bins[5:5+n_t_bins]))
    
    df = pd.DataFrame(wC_wA_C0_A0, columns=["wC", "wA", "C0", "A0"])
    
    C_corr = np.array(C_corr).astype("float")
    A_corr = np.array(A_corr).astype("float")
    landa_C = np.array(df["wC"]).astype("float")
    C0 = np.array(df["C0"]).astype("float")
    landa_A = np.array(df["wA"]).astype("float")
    A0 = np.array(df["A0"]).astype("float")
    return tcal, ical, t_bins, landa_C, landa_A, C0, A0, C_corr, A_corr



# Functions for ACS data QC
# func_acs_fit,acsa_complete_qc_v02,acsc_complete_qc_v02

def func_acs_fit(x, a, b):  # fitting exp funtion with ref wavelength at 400 nm
    return a * np.exp(-b * (x - 400))


def smooth_acs_vertically(df):
    ###### smooth the data in z direction
    df_smth = df.copy()
    for ii in range(len(df_smth.columns)):
        df_smth[df_smth.columns[ii]] = savgol_filter(np.array(df_smth[df_smth.columns[ii]]), 11, 3)
    return df_smth


def acsa_complete_qc_v02(df_a_raw):  # qc of absorption raw dataframe
    wl_a = np.array(df_a_raw.columns)
    """###### cut the data after 720 nm
    ind_720nm = np.min(np.where(df_a_raw.columns > 720)[0])
    if (~np.isnan(ind_720nm)):
        df_a_raw = df_a_raw.drop(columns=df_a_raw.columns[ind_720nm:])"""
    ###### Flag for nan values
    qc_flag_nan = 9
    df_a_nan_flag = pd.DataFrame(index=df_a_raw.index, columns=df_a_raw.columns)
    df_a_nan_flag.iloc[0:] = 0
    df_a_nan_flag[np.array(df_a_raw.isna())] = qc_flag_nan

    ###### Flag for negative values
    qc_flag_neg = 4
    df_a_neg_flag = pd.DataFrame(index=df_a_raw.index, columns=df_a_raw.columns)
    df_a_neg_flag.iloc[0:] = 0
    df_a_raw_neg = df_a_raw.copy()
    ind_neg = np.where(df_a_raw < 0)
    for ii in range(len(ind_neg[0])):
        df_a_raw_neg.iloc[ind_neg[0][ii]][[df_a_raw_neg.columns[ind_neg[1][ii]]]] = np.nan
        df_a_neg_flag.iloc[ind_neg[0][ii]][[df_a_neg_flag.columns[ind_neg[1][ii]]]] = qc_flag_neg

    ###### Flag for spikes
    qc_flag_spk = 3
    up_percentile = 99.5
    low_percentile = 0.5
    df_a_spike_flag = pd.DataFrame(index=df_a_raw.index, columns=df_a_raw.columns)
    df_a_spike_flag.iloc[0:] = 0
    df_a_raw_neg_spike = df_a_raw_neg.copy()
    for ii in range(len(df_a_raw.columns)):
        dum_arr = savgol_filter(np.array(df_a_raw[df_a_raw.columns[ii]]), 11, 3) - np.array(
            df_a_raw[df_a_raw.columns[ii]])
        up_bnd = np.nanpercentile(dum_arr, up_percentile)
        low_bnd = np.nanpercentile(dum_arr, low_percentile)

        ind_spike = np.where((dum_arr > up_bnd) | (dum_arr < low_bnd))[0]
        df_a_raw_neg_spike[df_a_raw_neg_spike.columns[ii]].iloc[ind_spike] = np.nan
        df_a_spike_flag[df_a_spike_flag.columns[ii]].iloc[ind_spike] = qc_flag_spk

    ###### Flag for problematic acs spectra
    df_a_raw_nan_interp = df_a_raw_neg_spike.copy()
    qc_flag_exp = 3
    log_ratio_thre = 0.3
    df_a_acs_flag = pd.DataFrame(index=df_a_raw_nan_interp.index, columns=df_a_raw_nan_interp.columns)
    df_a_acs_flag.iloc[0:] = 0
    df_a_acs_nan = df_a_raw_nan_interp.copy()

    ###### Interpolate in the z direction
    xx, yy = np.meshgrid(np.array(df_a_acs_nan.columns), np.array(df_a_acs_nan.index))
    z_arr = np.array(df_a_acs_nan)
    z_arr = np.ma.masked_invalid(z_arr)
    x1 = xx[~z_arr.mask]
    y1 = yy[~z_arr.mask]
    newarr = z_arr[~z_arr.mask]
    if (np.size(newarr) > 100):
        z_arr_interp = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')
        df_a_acs_nan_z_interp = pd.DataFrame(data=z_arr_interp, index=df_a_acs_nan.index, columns=df_a_acs_nan.columns)
        df_a_acs_nan_z_interp = df_a_acs_nan_z_interp.interpolate('index', limit_direction='both')
    else:
        df_a_acs_nan_z_interp = df_a_acs_nan.copy()
        df_a_acs_nan_z_interp = df_a_acs_nan_z_interp.interpolate('index', limit_direction='both')

    ###### smooth the data in z direction
    df_a_acs_nan_z_interp_smth = df_a_acs_nan_z_interp.copy()
    for ii in range(len(df_a_acs_nan_z_interp_smth.columns)):
        df_a_acs_nan_z_interp_smth[df_a_acs_nan_z_interp_smth.columns[ii]] = savgol_filter(
            np.array(df_a_acs_nan_z_interp_smth[df_a_acs_nan_z_interp_smth.columns[ii]]), 11, 3)

    ###### Define the QC flag for all
    df_qc_flag_all = df_a_spike_flag.copy()
    df_qc_flag_all = df_qc_flag_all.mask(df_a_acs_flag == qc_flag_exp, qc_flag_exp)
    df_qc_flag_all = df_qc_flag_all.mask(df_a_neg_flag == qc_flag_neg, qc_flag_neg)
    df_qc_flag_all = df_qc_flag_all.mask(df_a_nan_flag == qc_flag_nan, qc_flag_nan)

    return df_a_acs_nan_z_interp, df_a_acs_nan_z_interp_smth, df_qc_flag_all


def acsc_complete_qc_v02(df_c_raw):  # qc of attenuation raw dataframe

    """"###### cut the data after 720 nm
    ind_720nm = np.min(np.where(df_c_raw.columns > 720)[0])
    if (~np.isnan(ind_720nm)):
        df_c_raw = df_c_raw.drop(columns=df_c_raw.columns[ind_720nm:])"""

    ###### Flag for nan values
    qc_flag_nan = 9
    df_c_nan_flag = pd.DataFrame(index=df_c_raw.index, columns=df_c_raw.columns)
    df_c_nan_flag.iloc[0:] = 0
    df_c_nan_flag[np.array(df_c_raw.isna())] = qc_flag_nan

    ###### Flag for negative values
    qc_flag_neg = 4
    df_c_neg_flag = pd.DataFrame(index=df_c_raw.index, columns=df_c_raw.columns)
    df_c_neg_flag.iloc[0:] = 0
    df_c_raw_neg = df_c_raw.copy()
    ind_neg = np.where(df_c_raw < 0)
    for ii in range(len(ind_neg[0])):
        df_c_raw_neg.iloc[ind_neg[0][ii]][[df_c_raw_neg.columns[ind_neg[1][ii]]]] = np.nan
        df_c_neg_flag.iloc[ind_neg[0][ii]][[df_c_neg_flag.columns[ind_neg[1][ii]]]] = qc_flag_neg

    ###### Flag for spikes
    qc_flag_spk = 3
    up_percentile = 99.5
    low_percentile = 0.5
    df_c_spike_flag = pd.DataFrame(index=df_c_raw.index, columns=df_c_raw.columns)
    df_c_spike_flag.iloc[0:] = 0
    df_c_raw_neg_spike = df_c_raw_neg.copy()
    for ii in range(len(df_c_raw.columns)):
        dum_arr = savgol_filter(np.array(df_c_raw[df_c_raw.columns[ii]]), 11, 3) - np.array(
            df_c_raw[df_c_raw.columns[ii]])
        up_bnd = np.nanpercentile(dum_arr, up_percentile)
        low_bnd = np.nanpercentile(dum_arr, low_percentile)

        ind_spike = np.where((dum_arr > up_bnd) | (dum_arr < low_bnd))[0]
        df_c_raw_neg_spike[df_c_raw_neg_spike.columns[ii]].iloc[ind_spike] = np.nan
        df_c_spike_flag[df_c_spike_flag.columns[ii]].iloc[ind_spike] = qc_flag_spk

    ###### Interpolate in the z direction
    df_c_acs_nan = df_c_raw_neg_spike.copy()
    xx, yy = np.meshgrid(np.array(df_c_acs_nan.columns), np.array(df_c_acs_nan.index))
    z_arr = np.array(df_c_acs_nan)
    z_arr = np.ma.masked_invalid(z_arr)
    x1 = xx[~z_arr.mask]
    y1 = yy[~z_arr.mask]
    newarr = z_arr[~z_arr.mask]
    if (np.size(newarr) > 100):
        z_arr_interp = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')
        df_c_acs_nan_z_interp = pd.DataFrame(data=z_arr_interp, index=df_c_acs_nan.index, columns=df_c_acs_nan.columns)
        df_c_acs_nan_z_interp = df_c_acs_nan_z_interp.interpolate('index', limit_direction='both')
    else:
        df_c_acs_nan_z_interp = df_c_acs_nan.copy()
        df_c_acs_nan_z_interp = df_c_acs_nan_z_interp.interpolate('index', limit_direction='both')

    ###### Define the QC flag for all
    df_qc_flag_all = df_c_spike_flag.copy()
    df_qc_flag_all = df_qc_flag_all.mask(df_c_neg_flag == qc_flag_neg, qc_flag_neg)
    df_qc_flag_all = df_qc_flag_all.mask(df_c_nan_flag == qc_flag_nan, qc_flag_nan)

    ###### smooth the data in z direction
    df_c_acs_nan_z_interp_smth = df_c_acs_nan_z_interp.copy()
    for ii in range(len(df_c_acs_nan_z_interp_smth.columns)):
        df_c_acs_nan_z_interp_smth[df_c_acs_nan_z_interp_smth.columns[ii]] = savgol_filter(
            np.array(df_c_acs_nan_z_interp_smth[df_c_acs_nan_z_interp_smth.columns[ii]]), 11, 3)

    return df_c_acs_nan_z_interp, df_c_acs_nan_z_interp_smth, df_qc_flag_all

