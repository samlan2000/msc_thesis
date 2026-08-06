import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.transform import rowcol
import os


class bsqConverterPolymer():

    def __init__(self, ds, out_bsq, ds_idepix=None, bands=['Oa1', 'Oa2', 'Oa3', 'Oa4', 'Oa5', 'Oa6', 'Oa7', 'Oa8', 'Oa10']):
        self.ds = ds
        self.lon, self.lat = self._get_coords()
        self.idepix_mask = ds_idepix["pixel_classif_flags"].astype("int32").values > 0 if ds_idepix else None
        if self.idepix_mask is None:
            print("Warning: No idepix masking, skipping.")
            return
            
        try:
            data, wavelengths, fwhm = self._extract_bands(bands)
        except KeyError as e:
            print("At least one band not matched:")
            print(e)
            print(f"Skipping image {out_bsq}")
            return
        
        sza = str(np.nanmean(ds["sza"].values))
        vza = str(np.nanmean(ds["vza"].values))

        self._build_bsq(out_bsq, data, wavelengths, fwhm, sza, vza)

    def _extract_bands(self, bands):
        # [centre wvl, fwhm, alias]
        band_LUT = {
            "polymer": {
                'Oa1':  [400, 15, "Rw400"],
                'Oa2':  [412.5, 10, "Rw412"],
                'Oa3':  [442.5, 10, "Rw443"],
                'Oa4':  [490, 10, "Rw490"],
                'Oa5':  [510, 10, "Rw510"],
                'Oa6':  [560, 10, "Rw560"],
                'Oa7':  [620, 10, "Rw620"],
                'Oa8':  [665, 10, "Rw665"],
                'Oa9':  False,
                'Oa10': [681.25, 7.5, "Rw681"],
                'Oa11': [708.75, 10, "Rw709"],
                'Oa12': [753.75, 7.5, "Rw754"],
                'Oa13': False,
                'Oa14': False,
                'Oa15': False,
                'Oa16': [778.75, 15, "Rw779"],
                'Oa17': [865, 20, "Rw865"],
                'Oa18': False,
                'Oa19': False,
                'Oa20': False,
                'Oa21': [1029, 40, "Rw1020"]
                }
            }
        
        band_LUT["polymer"] = {
            k: v
            for k, v in band_LUT["polymer"].items()
            if k in bands and v
        }

        band_alias_list = [l[2] for l in band_LUT["polymer"].values()]
        
        # Stack into numpy array (bands, y, x)
        bands = []
        for band_alias in band_alias_list:
            bands.append(self.ds[band_alias].values)
        
        data = np.stack(bands)   # shape: (B, H, W)
        # Water leaving reflectance to Rrs conversion
        data = data / np.pi
            
        # Mask invalid pixels using IDEPIX
        if self.idepix_mask is not None:
            # Ensure mask shape matches spatial dims
            if self.idepix_mask.shape == data.shape[1:]:
                data = data.copy()  # avoid modifying original array
                data[:, self.idepix_mask] = np.nan
            else:
                raise ValueError(
                    f"IDEPix mask shape {self.idepix_mask.shape} "
                    f"does not match data shape {data.shape[1:]}"
                )

        wavelengths = [l[0] for l in band_LUT["polymer"].values()]

        fwhm = [l[1] for l in band_LUT["polymer"].values()]

        return data, wavelengths, fwhm

    def _get_coords(self):
        lon, lat = self.ds["longitude"].values, self.ds["latitude"].values

        return lon, lat

    def _build_bsq(self, out_bsq, data, wavelengths, fwhm, sza, vza):

        
        # Use min/max to create affine transform   
        min_lon, max_lon = self.lon.min(), self.lon.max()
        min_lat, max_lat = self.lat.min(), self.lat.max()
       
        height = self.lon.shape[0]
        width  = self.lon.shape[1]
        
        self.transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
        
        crs = "EPSG:4326"
        
        # ----------------------------------------------------------------------------------
        # WRITE ENVI BSQ FILE
        # ----------------------------------------------------------------------------------       
        profile = {
            "driver": "ENVI",
            "dtype": "float32",
            "count": data.shape[0],
            "height": height,
            "width": width,
            "crs": crs,
            "transform": self.transform,
            "interleave": "band"
        }
        
        with rasterio.open(out_bsq, "w", **profile) as dst:
            dst.write(data.astype(np.float32))
        
        print("BSQ file written:", out_bsq)

        # ==============================
        # Reference stations (lat, lon)
        # ==============================
        stations = {
            "shl2": (46.453015, 6.588760),
            "lxp": (46.500283, 6.660889),
            "P1": (46.4985137, 6.6669396),
            "P2": (46.474122, 6.666252),
            "P3": (46.46751, 6.62626),
            "P4": (46.49445, 6.61614),
            "P5": (46.49522, 6.57657),
            "P6": (46.47770, 6.57972),
            "S1": (46.486196, 6.60979),
            "S2": (46.48018, 6.619005),
            "S3": (46.46535, 6.639),
            "S4": (46.46532, 6.65768),
            "S5": (46.473523, 6.674493),
        }
        
        # ==============================
        # Convert lon/lat to row/col
        # ==============================
        station_indices = {}
        
        for name, (lat, lon) in stations.items():
        
            # Convert coordinates (note: rowcol expects lon, lat)
            row, col = rowcol(self.transform, lon, lat)
        
            # Check raster bounds
            if (0 <= row < height) and (0 <= col < width):
                idx_str = f"{row}, {col}"
            else:
                idx_str = "NaN, NaN"
        
            station_indices[name] = idx_str
        
        # ----------------------------------------------------------------------------------
        # MODIFY THE HDR TO ADD WAVELENGTH INFORMATION
        # ----------------------------------------------------------------------------------
        hdr_file = out_bsq.replace(".bsq", ".hdr")
        
        with open(hdr_file, "a") as hdr:
            hdr.write("\nwavelength units = Nanometers\n")
            
            hdr.write("wavelength = {\n")
            hdr.write(", ".join(str(w) for w in wavelengths))
            hdr.write("}\n\n")
        
            hdr.write("fwhm = {\n")
            hdr.write(", ".join(str(s) for s in fwhm))
            hdr.write("}\n\n")
        
            hdr.write("sza = {\n")
            hdr.write(str(sza))
            hdr.write("}\n\n")
        
            hdr.write("vza = {\n")
            hdr.write(str(vza))
            hdr.write("}\n\n")
        
            # ==============================
            # Write all station indices
            # ==============================
            for station, idx in station_indices.items():
                hdr.write(f"{station.lower()}_idx = {{\n")
                hdr.write(str(idx))
                hdr.write("}\n\n")
            
            print("HDR updated with wavelength and observation geometry information:", hdr_file)
        
        
                    
def convert_polymer_batch(basepath_in, basepath_out, bands=['Oa1', 'Oa2', 'Oa3', 'Oa4', 'Oa5', 'Oa6', 'Oa7', 'Oa8', 'Oa10']):
    for product_folder in os.listdir(basepath_in):
        product_folder_path = os.path.join(basepath_in, product_folder)
        for L2_folder in os.listdir(product_folder_path):
            if L2_folder == "L2POLY":
                L2_folder_path = os.path.join(product_folder_path, L2_folder)
                for product in os.listdir(L2_folder_path):
                    if product.endswith(".nc"):
                        acolite_alias = product.split("_")[3] + "_" + product.split("____")[-1][:15]
                        product_path = os.path.join(L2_folder_path, product)
                        idepix_equivalent = product.replace("L2POLY_reproj_NASA", "L1P_reproj")
                        idepix_path = os.path.join(product_folder_path, "L1P", idepix_equivalent)
                        ds = xr.open_dataset(product_path)
                        ds_idepix = xr.open_dataset(idepix_path) if os.path.exists(idepix_path) else None
                        path_out = os.path.join(basepath_out, acolite_alias + ".bsq")
                        print(f"Converting product {acolite_alias}...")
                        bsqConverterPolymer(ds, path_out, ds_idepix=ds_idepix, bands=bands)
                        print("")