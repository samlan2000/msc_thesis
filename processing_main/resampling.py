import os
import numpy as np
import pandas as pd
from spectral import BandResampler
from pathlib import Path

### ADAPTED FROM THE BIO-OPTICS PACKAGE SOURCECODE
# König, M., Noel, P., Hondula. K.L., Jamalinia, E., Dai, J., Vaughn, N.R., Asner, G.P. (2023): 
# bio_optics python package (Version x) [Software]. Available from https://github.com/CMLandOcean/bio_optics. 
# [https://doi.org/10.5281/zenodo.10246860]

# Path to THIS script
script_dir = Path(__file__).resolve().parent

# Path to data folder next to script
data_dir = script_dir / "data"

### Water

def resample_a_w(wavelengths = np.arange(400,800), FWHMs = None):
    """
    Absorption coefficient of pure water [m-1] at a reference temperature of 20 degC 
    as a compilation from different sources as distributed with the Water Color Simulator 6 (WASI6) [1]

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to resample the absorption coefficient of pure water to
    :return: absorption coefficient of pure water absorption resampled to input wavelengths
    """
    a_w_db = pd.read_csv(os.path.join(data_dir, 'a_w.txt'), skiprows=14, sep='\t', usecols=[0,1])
    # resample to sensor bands
    band_resampler = BandResampler(a_w_db.wavelength_nm.values, wavelengths, fwhm1=None, fwhm2=FWHMs)
    a_w = band_resampler(a_w_db["a"])
    return a_w


def resample_da_W_div_dT(wavelengths = np.arange(400,800), FWHMs = None):
    """
    Temperature gradient of pure water absorption [m-1  degC-1]
    after Roettgers et al. (2013) [1] as distributed with the Water Color Simulator 6 (WASI6) [2]

    [1] Roettgers et al. (2013): Pure water spectral absorption, scattering, and real part of refractive index model.
                                 Algorithm Theoretical Basis Document "The Water Optical Properties Processor (WOPP).
                                 Distribution: Marc Bouvet, ESA/ESRIN
                                 Revision 7, May 2013
    [2] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

    :param wavelengths: wavelengths to resample the temperature gradient of pure water absorption to
    :return: temperature gradient of pure water absorption resampled to input wavelengths
    """
    # read file
    da_W_div_dT_db = pd.read_csv(os.path.join(data_dir, 'daWdT.txt'), skiprows=9, sep='\t')
    # resample to sensor bands
    band_resampler = BandResampler(da_W_div_dT_db.wavelength_nm.values, wavelengths, fwhm1=None, fwhm2=FWHMs)
    da_W_div_dT = band_resampler(da_W_div_dT_db["daW/dT"])
    return da_W_div_dT


def resample_a_i_spec(wavelengths = np.arange(400,800), FWHMs = None):
    """
    Specific absorption coefficients [m2 mg-1] of six phytoplankton types compiled from multiple sources
    as distributed with the Water Color Simulator 6 (WASI6) [1] 

    1. phytoplankton
    2. cryptophyta
    3. cyanobacteria
    4. diatoms
    5. dinoflagellates
    6. green algae

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

    :param wavelengths: wavelengths to resample the specific absorption coefficients to
    :return: specific absorption coefficients of six phytoplankton types resampled to input wavelengths
    """
    # read file
    a_phyto_db = pd.read_csv(os.path.join(data_dir, 'a_phy_spec.txt'), skiprows=25, sep=",")
    # resample to sensor bands
    #band_resampler = BandResampler(a_phyto_db.wavelength_nm.values, wavelengths)
    band_resampler = BandResampler(a_phyto_db.wavelength_nm.values, wavelengths, fwhm1=None, fwhm2=FWHMs)
    a_i_spec = band_resampler(np.asarray(a_phyto_db)[:,1:])
    
    return a_i_spec

def resample_a_i_spec_MiniWASI(wavelengths = np.arange(400,800), FWHMs = None):
    """
    [1]
    1. phytoplankton
    2. cryptophyta
    3. diatoms
    4. dinoflagellates
    5. green algae
    
    [2]
    6. Brown group
    7. Cyanobacteria blue
    8. Cyanobacteria red
    

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    [2] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.1196352] 

    :param wavelengths: wavelengths to resample the specific absorption coefficients to
    :return: specific absorption coefficients of six phytoplankton types resampled to input wavelengths
    """
    # read file
    a_phyto_db = pd.read_csv(os.path.join(data_dir, 'a_phy_spec_MiniWASI.csv'), sep=";")
    # resample to sensor bands
    #band_resampler = BandResampler(a_phyto_db.wavelength_nm.values, wavelengths)
    band_resampler = BandResampler(a_phyto_db["wavelengths"].values, wavelengths, fwhm1=None, fwhm2=FWHMs)
    a_i_spec = band_resampler(np.asarray(a_phyto_db)[:,1:])
    
    return a_i_spec


def resample_a_i_spec_EnSAD(wavelengths = np.arange(400,720), FWHMs = None):
    """
    Specific absorption coefficients [m2 mg-1] of eight phytoplankton types from the supplemental data of [1] and WASI [2].

    1. Brown group
    2. Green group
    3. Cryptophyte
    4. Cyanobacteria blue
    5. Cyanobacteria red
    6. Coccolithophore
    7. Dinoflagellates from [2]
    8. Phytoplankton Case 1

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.1196352] 
    [2] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.

    :param wavelengths: wavelengths to resample the specific absorption coefficients to
    :return: specific absorption coefficients of seven phytoplankton types resampled to input wavelengths
    """
    # read file
    a_phyto_db = pd.read_csv(os.path.join(data_dir, 'a_phy_spec_EnSAD.txt'), skiprows=11, sep=",")
    # resample to sensor bands
    band_resampler = BandResampler(a_phyto_db.wavelength_nm.values, wavelengths, fwhm1=None, fwhm2=FWHMs)
    a_i_spec = band_resampler(np.asarray(a_phyto_db)[:,1:])
    
    return a_i_spec


def resample_b_i_spec_EnSAD(wavelengths = np.arange(400,720), FWHMs = None):
    """
    Specific scattering coefficients [m2 mg-1] of seven phytoplankton types from the supplemental data of [1].

    1. Brown group
    2. Green group
    3. Cryptophyte
    4. Cyanobacteria blue
    5. Cyanobacteria red
    6. Coccolithophore
    7. Dinoflagellates (identical to Brown group)
    8. Phytoplankton Case-1

    [1] Bi et al. (2023): Bio-geo-optical modelling of natural waters [10.3389/fmars.2023.1196352] 

    :param wavelengths: wavelengths to resample the specific absorption coefficients to
    :return: specific absorption coefficients of seven phytoplankton types resampled to input wavelengths
    """
    # read file
    b_phyto_db = pd.read_csv(os.path.join(data_dir, 'b_phy_spec_EnSAD.txt'), skiprows=4, sep=",")
    # resample to sensor bands
    band_resampler = BandResampler(b_phyto_db.wavelength_nm.values, wavelengths, fwhm1=None, fwhm2=FWHMs)
    b_i_spec = band_resampler(np.asarray(b_phyto_db)[:,1:])
    
    return b_i_spec


def resample_b_phy_norm(wavelengths = np.arange(400,800), FWHMs = None):
    """
    Normalized backscattering coefficient of phytoplankton as distributed with the Water Color Simulator 6 (WASI6) [1]
    obtained by fitting a measurement of b_b_phy(lambda) for green algae from Lake Garda in the range from 400 to 900 nm (Giardino, personal communication) 
    and extrapolating the fit curve to the range from 350 to 1000 nm.

    [1] Gege (2021): The Water Colour Simulator WASI. User manual for WASI version 6.
    
    :param wavelengths: wavelengths to compute normalized backscattering coefficient of phytoplankton for
    :return: normalized backscattering coefficient of phytoplankton for input wavelengths
    """
    # READ DATA FROM DATABASE
    b_phy_norm = pd.read_csv(os.path.join(data_dir, 'b_phy_norm.txt'), skiprows=6, sep="\t")
    # RESAMPLE TO WAVELENGTHS
    band_resampler = BandResampler(b_phy_norm.wavelength_nm.values, wavelengths, fwhm1=None, fwhm2=FWHMs)
    b_phy_norm = band_resampler(b_phy_norm["bb_phy_norm"])
    
    return b_phy_norm
    

### Generic
def resample_spectra(spectra, in_wavelengths, out_wavelengths, FWHMs):
    """
    Generic resampling function
    """    
    band_resampler = BandResampler(in_wavelengths, out_wavelengths, fwhm1=None, fwhm2=FWHMs)
    resampled_spectra = band_resampler(spectra)

    return resampled_spectra


def resample_a_Y_norm(wavelengths = np.arange(400,800), FWHMs = None):
    """
    Gaussian approximation of normalized cdom absorption (Lake Constance)
    """
    a_Y = pd.read_csv(os.path.join(data_dir, 'Y_Gauss.txt'), skiprows=11, sep=' ', usecols=[0,1])
    # resample to sensor bands
    band_resampler = BandResampler(a_Y.nm.values, wavelengths, fwhm1=None, fwhm2=FWHMs)
    a_Y_res = band_resampler(a_Y["dimensionless"])
    return a_Y_res
    
    