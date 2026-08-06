import MiniWASIsafe
import numpy as np
from spectral import envi
import matplotlib.pyplot as plt
from tqdm import tqdm


class ImageProcessor():

    def __init__(self, img_path, out_path, weights=None, use_previous_init=False,
                 vary={'C_0': True, 'C_x': True, 'C_y': True},
                 bounds={'C_y': (0, 0.5), 'C_0': (0, 15), 'C_x': (0, 10)},
                 init={'C_y': 0.1, 'C_0': 4, 'C_x': 1},
                 output_wcs=['C_0', 'C_x', 'C_y'],
                 output_iops=["a","a_cdom","a_nap","a_phy","bb","bb_nap","bb_phy"]
                 ):
        
        self.output_wcs = output_wcs
        output_iops.sort()
        
        self.results = self._process_image(img_path, out_path, weights, vary, bounds, init, output_iops, use_previous_init)
        

    def _process_image(self, img_path, out_path, weights, vary, bounds, init, output_iops, use_previous_init):
        """
        Process ENVI BSQ image with WASI inversion and write ENVI BSQ output
        using georeference from input header.
        """
    
        hdr_path = img_path.replace(".bsq", ".hdr")
    
        # -------------------------------------------------
        # Read ENVI header (metadata + georeference)
        # -------------------------------------------------
        header = envi.read_envi_header(hdr_path)
    
        wavelengths = np.array([float(w) for w in header["wavelength"]])
        FWHMs = np.array([float(w) for w in header["fwhm"]])
        sza = float(header["sza"][0])
        vza = float(header["vza"][0])
    
        # -------------------------------------------------
        # Load BSQ cube
        # -------------------------------------------------
        img = envi.open(hdr_path, img_path)
        cube = img.load()  # (lines, samples, bands)
    
        lines, samples, bands = cube.shape
    
        # -------------------------------------------------
        # Create WASI object
        # -------------------------------------------------
        wasi = MiniWASIsafe.MiniWasi(
            wavelengths=wavelengths,
            FWHMs=FWHMs,
            va=vza,
            sza=sza,
        )
        
        n_scalar = len(self.output_wcs)
        n_iop = len(output_iops)
        n_spec = len(wavelengths)  # spectral length
        
        # One extra band for residuum
        nb_out = n_scalar + n_iop * n_spec + 1

        # -------------------------------------------------
        # Allocate output cube (BSQ)
        # -------------------------------------------------
        results = np.zeros((lines, samples, nb_out), dtype=np.float32)
    
        # -------------------------------------------------
        # Invert every pixel
        # -------------------------------------------------
        for i in tqdm(range(lines), desc="Processing lines"):
            for j in range(samples):
    
                spectrum = cube[i, j, :]
    
                if not np.all(np.isfinite(spectrum)) or np.all(spectrum == 0):
                    continue
                
                inv = wasi.invert(spectrum, vary=vary, bounds=bounds, init=init, weights=weights)
                residuum = np.mean(inv.residual ** 2)
                params = inv.params
                results[i, j, -1] = residuum
                
                # Standard concentrations
                for k, name in enumerate(self.output_wcs):
                    results[i, j, k] = params[name].value
                    
                wasi_iop_LUT = {
                    "a": wasi.a,
                    "a_cdom": wasi.a_cdom,
                    "a_nap": wasi.a_nap,
                    "a_phy": wasi.a_phy,
                    "bb": wasi.bb,
                    "bb_nap": wasi.bb_nap,
                    "bb_phy": wasi.bb_phy
                }
                    
                # IOP values
                iop_spectra = [v for k, v in wasi_iop_LUT.items() if k in output_iops]
                
                offset = n_scalar
                
                if use_previous_init:
                    init = {var: params[var].value for var in init.keys()}

                for k, spectrum in enumerate(iop_spectra):
                    start = offset + k * n_spec
                    end = start + n_spec
                    results[i, j, start:end] = spectrum

    
        # -------------------------------------------------
        # Build output ENVI header
        # -------------------------------------------------
        out_hdr = header.copy()
        
        expanded_iop_names = []

        for name in output_iops:
            for wl in wavelengths:
                expanded_iop_names.append(f"{name}_{int(wl)}")
    
        out_hdr["samples"] = samples
        out_hdr["lines"] = lines
        out_hdr["bands"] = nb_out
        out_hdr["interleave"] = "bsq"
        out_hdr["data type"] = 4  # float32
        
        expanded_iop_names = []

        for name in output_iops:
            for wl in wavelengths:
                expanded_iop_names.append(f"{name}_{int(wl)}")
        
        out_hdr["band names"] = self.output_wcs + expanded_iop_names + ["residual"]

        out_wavelengths = []

        # scalars get dummy wavelengths
        out_wavelengths.extend([0] * n_scalar)
        
        for _ in output_iops:
            out_wavelengths.extend(wavelengths.tolist())
            
        # One extra dummy for residuum
        out_wavelengths.append(0)
        
        out_hdr["wavelength"] = out_wavelengths
        out_hdr.pop("fwhm", None)
    
        # -------------------------------------------------
        # Write ENVI BSQ
        # -------------------------------------------------
        envi.save_image(
            out_path.replace(".bsq", ".hdr"),
            results,
            dtype=np.float32,
            interleave="bsq",
            metadata=out_hdr,
            force=True,
        )
    
        print("Saved ENVI BSQ:", out_path)
    
        return results


    
    def plot_results(self, results, out_path=None):
    
        def pclip(a, low=2, high=98):
            vmin, vmax = np.percentile(a, [low, high])
            return vmin, vmax
        
        unit_LUT = {"C_0": "µg/L",
                    "C_1": "µg/L",
                    "C_2": "µg/L",
                    "C_3": "µg/L",
                    "C_4": "µg/L",
                    "C_5": "µg/L",
                    "C_x": "mg/L",
                    "C_y": "1/m"
                    }
        
        band_LUT = {}
        for k, wc in enumerate(self.output_wcs):
            band_LUT[wc] = results[:, :, k]
            
           
        n = len(band_LUT)

        fig, axs = plt.subplots(1, n, figsize=(6 * n, 6))
    
        # If only one subplot, make it iterable
        if n == 1:
            axs = [axs]
    
        for ax, (param, data) in zip(axs, band_LUT.items()):
    
            vmin, vmax = pclip(data)
    
            im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
    
            unit = unit_LUT.get(param, "")
            ax.set_title(f"{param} [{unit}]")
    
            plt.colorbar(im, ax=ax, fraction=0.046)
    
        plt.tight_layout()
    
        if out_path:
            plt.savefig(out_path, dpi=300)
            print(f".png of inversion results saved at: {out_path}")
    
        plt.show()
    