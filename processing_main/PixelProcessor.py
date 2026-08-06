import MiniWASIsafe
import numpy as np
from spectral import envi
import matplotlib.pyplot as plt
import os


class SinglePixelProcessor():

    def __init__(self, img_path, station_idx=[1,1], i_offset=0, j_offset=0, valid_pixel_min=0, 
                 plot=False, weights=None, station_name=None, a_norm_y_from_file=False, skip_negative=False,
                 output_wcs=['C_0', 'C_x', 'C_y'],
                 vary={'C_0': True, 'C_x': True, 'C_y': True},
                 bounds={'C_y': (0, 0.5), 'C_0': (0, 15), 'C_x': (0, 15)},
                 init={'C_0': 4, 'C_x': 1, 'C_y': 0.1},
                 ):
        
        self.inv = None
        
        self.station_name = station_name.lower() if station_name else None
        self.station_idx = station_idx
                 
        self.plot = plot
        self.output_wcs = output_wcs

        hdr_path = img_path.replace(".bsq", ".hdr")
        img = envi.open(hdr_path, img_path)
    
        header = envi.read_envi_header(hdr_path)
        self.station_idx = header[f"{station_name.lower()}_idx"] if station_name else station_idx
        #print(station_idx)
        cube = img.load()

        band = cube[:, :, 0]   # first band
        valid_pixels = np.count_nonzero(~np.isnan(band))
        
        print("Number of valid pixels:", valid_pixels)
        if valid_pixels < valid_pixel_min:
            print(f"Number of valid pixels is below set minimum: {valid_pixels} < {valid_pixel_min}")
            return

        self.date = os.path.basename(img_path).split("_")[-1].split(".")[0]
        self._process_single_pixel(header, cube, a_norm_y_from_file, weights, init, vary, bounds, i_offset, j_offset, skip_negative)
        

    def _process_single_pixel(self, header, cube, a_norm_y_from_file, weights, init, vary, bounds, i_offset, j_offset, skip_negative):
        
        if self.station_idx[0] == "NaN":
            print(f"No valid pixel for station {self.station_name}.")
            return

        j = int(self.station_idx[1]) + j_offset
        i = int(self.station_idx[0]) + i_offset
        
        self.spectrum = cube[i, j, :]
        
        if skip_negative == True and np.any(self.spectrum[0,0][:8] < 0):
            return
    
        def has_nan(arr):
            return np.isnan(arr).any()

        if has_nan(self.spectrum):
            print("Skipped spectrum due to NaNs.")
            return

        # ---- WASI ----
        self.wavelengths = np.array([float(w) for w in header["wavelength"]])
        self.FWHMs = np.array([float(w) for w in header["fwhm"]])
        self.sza = float(header["sza"][0])
        self.va = float(header["vza"][0])
            
        self.wasi = MiniWASIsafe.MiniWasi(wavelengths=self.wavelengths, FWHMs=self.FWHMs, va=self.va, sza=self.sza, a_norm_y_from_file=a_norm_y_from_file)

        self.inv = self.wasi.invert(self.spectrum,
                                    init=init,
                                    vary=vary, 
                                    bounds=bounds,
                                    weights=weights)
        
        self.rmse = np.sqrt(np.mean(self.inv.residual**2))
        self.inv.params.pretty_print()
        self.spectrum_modelled = self.wasi.R_rs
        

    def plot_two_spectra(self):
        
        wavelengths = np.array(self.wavelengths)

        spectrum_meas = np.squeeze(np.array(self.spectrum_meas))
        spectrum_model = np.squeeze(np.array(self.spectrum_model))
    
        plt.figure(figsize=(5,3))
    
        plt.plot(wavelengths, spectrum_meas, linewidth=1.5, label="S3 OLCI Rrs")
        plt.plot(wavelengths, spectrum_model, linewidth=1.5, label="Inverted Rrs")
    
        plt.title(f"{self.date} / {self.station_name if self.station_name else self.station_idx}. RMSE = {round(self.rmse, 6)}", fontsize=14)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Rrs (sr$^{-1}$)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()