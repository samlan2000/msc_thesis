import numpy as np
import lmfit
import resampling
import warnings
warnings.filterwarnings("ignore", message="__array_wrap__")
warnings.filterwarnings("ignore", 'Image data contains NaN values.')


class MiniWasi():

    def __init__(self, wavelengths = np.arange(400,900), FWHMs = None,
                 T=20, va=0.0001, sza=40,
                 bb_nap_spec=0.0086, a_spec_nap_440nm=0.041, bb_phy_spec=0.001, s_cdom=0.014, s_nap=0.011,
                 a_norm_y_from_file = False
                 ):

        self.wavelengths = wavelengths
        self.T = T
        
        self.bb_nap_spec = bb_nap_spec
        self.a_spec_nap_440nm = a_spec_nap_440nm
        self.bb_phy_spec=bb_phy_spec
        self.s_cdom = s_cdom
        self.s_nap = s_nap
        
        # Normalized cdom absorption coefficient at 440 nm
        self.a_norm_y = resampling.resample_a_Y_norm(wavelengths, FWHMs) if a_norm_y_from_file else np.exp(-self.s_cdom*(self.wavelengths-440))
        
        # Normalized nap absorption coefficient at 440 nm
        self.a_norm_nap = np.exp(-self.s_nap*(self.wavelengths-440))
        
        # Geometry-dependent constants
        self.va = np.radians(va) if va != 0 else np.radians(0.0001)  # avoid division by zero
        self.sza = np.radians(sza)
        # Viewing angle in water
        self.inwater_va = np.arcsin(np.sin(self.va)/1.33)
        # Sun zenith angle in water
        self.inwater_sza = np.arcsin(np.sin(self.sza)/1.33)
        # Fresnel reflectance at the air–water interface
        num_sin = np.sin(self.va - self.inwater_va) ** 2
        den_sin = np.sin(self.va + self.inwater_va) ** 2
        num_tan = np.tan(self.va - self.inwater_va) ** 2
        den_tan = np.tan(self.va + self.inwater_va) ** 2
        self.rho_L = 0.5 * (num_sin / den_sin + num_tan / den_tan) # ca. 0.02 (nadir)

        # Absorption of water
        self.a_w_res = resampling.resample_a_w(wavelengths, FWHMs)
        # Specific absorption coefficients of 6 phytoplankton types
        self.a_i_spec_res = resampling.resample_a_i_spec(wavelengths, FWHMs)
        #print(a_i_spec_res[:,0])
        # Normalized backscattering coefficient of phytoplankton
        self.bb_phy_norm_res = resampling.resample_b_phy_norm(wavelengths, FWHMs)
        # Temperature gradient of pure water absorption
        self.da_W_div_dT_res = resampling.resample_da_W_div_dT(wavelengths, FWHMs)
        

    def forward(self, C_x = 1, C_y = 0.2, C_0 = 2, C_1 = 0, C_2 = 0, C_3 = 0, C_4 = 0, C_5 = 0, 
                return_spectrum=False):
        
        ####
        # Relate IOPs to LUTs
        ####

        ## ABSORPTION
        
        # CDOM component
        # C_y = a_cdom_440nm
        self.a_cdom = C_y * self.a_norm_y

        # Phytoplankton component
        self.a_phy = C_0*self.a_i_spec_res[:,0] + C_1*self.a_i_spec_res[:,1] + C_2*self.a_i_spec_res[:,2] + C_3*self.a_i_spec_res[:,3] + C_4*self.a_i_spec_res[:,4] + C_5*self.a_i_spec_res[:,5]
        C_phy = C_0 + C_1 + C_2 + C_3 + C_4 + C_5

        # NAP component
        # Normalized nap absorption coefficient
        C_nap = C_x # + C_mie
        # WASI a_spec_nap_440nm: 0.055, manual: 0.041
        self.a_nap = C_nap * self.a_spec_nap_440nm * self.a_norm_nap

        # Bulk absorption
        T0 = 20
        self.a_wc = self.a_cdom + self.a_nap + self.a_phy
        self.a = self.a_w_res + (self.T - T0) * self.da_W_div_dT_res + self.a_wc

        ## BACKSCATTERING

        # Water
        self.bb_w = 0.00111 * (self.wavelengths/500)**(-4.32)

        # Phytoplankton - ONLY ONE ref spectrum
        self.bb_phy = C_phy * self.bb_phy_spec * self.bb_phy_norm_res

        # NAP
        # WASI bb_nap_spec: 0.013, WASI manual: 0.0086
        self.bb_nap = self.bb_nap_spec * C_x * np.ones(self.wavelengths.shape)

        self.bb_wc = self.bb_phy + self.bb_nap
        # Bulk backscattering
        self.bb = self.bb_w + self.bb_wc
        
        ####
        # Relate IOPs to Rrs
        ####
        wb = (self.bb/(self.a+self.bb))
        wb = np.clip(wb, 0, 1)
        
        # Account for anisotropy
        f_rs = 0.0512 * (1 + 4.6659 * wb - 7.8387 * wb**2 + 5.4571 * wb**3) * (1 + 0.1098/np.cos(self.inwater_sza)) * (1 + 0.4021/np.cos(self.inwater_va))

        # below water
        self.r_rs_below = f_rs * wb

        xi = (1-0.03)*(1-self.rho_L)/1.33**2 # ca. 0.53
        
        self.R_rs = xi * (self.r_rs_below/(1-0.54*5*self.r_rs_below))
        
        self.R_rs = np.nan_to_num(
                        self.R_rs,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0
                    )
                
        if return_spectrum:
            return self.R_rs
           

    def invert(self, Rrs_measured, weights=None, vary=None,
           init=None, bounds=None, minimizer="leastsq"):
        """
        Use like this:
        result = model.invert(
                    Rrs_measured,
                    vary={'C_x': True, 'C_y': True},
                    init={'C_x': 5.0},
                    bounds={'C_y': (0, 1.0)}
                )
        minimizers: https://lmfit.github.io/lmfit-py/fitting.html#the-minimize-function
        """
    
        # defaults
        param_names = ['C_x', 'C_y', 'C_0', 'C_1', 'C_2', 'C_3', 'C_4', 'C_5']
    
        default_init = {
            'C_x': 1.0,
            'C_y': 0.1,
            'C_0': 4.0,
            'C_1': 0.0,
            'C_2': 0.0,
            'C_3': 0.0,
            'C_4': 0.0,
            'C_5': 0.0,
        }
    
        default_bounds = {
            'C_x': (0, 10),
            'C_y': (0, 0.5),
            'C_0': (0, 15),
            'C_1': (0, 15),
            'C_2': (0, 15),
            'C_3': (0, 15),
            'C_4': (0, 15),
            'C_5': (0, 15),
        }
        
        # spectral weighting
        if weights is None:
            weights = np.ones_like(Rrs_measured)
        weights = np.asarray(weights, dtype=float)
        weights /= np.mean(weights)
        sqrt_w = np.sqrt(weights)
    
        # user overrides
        vary = vary or {}
        init = {**default_init, **(init or {})}
        bounds = {**default_bounds, **(bounds or {})}
    
        p = lmfit.Parameters()
    
        for name in param_names:
            p.add(
                name,
                value=init[name],
                min=bounds[name][0],
                max=bounds[name][1],
                vary=vary.get(name, False)   # default: fixed
            )
    
        def residual(params):
            model_Rrs = self.forward(
                C_x=params['C_x'],
                C_y=params['C_y'],
                C_0=params['C_0'],
                C_1=params['C_1'],
                C_2=params['C_2'],
                C_3=params['C_3'],
                C_4=params['C_4'],
                C_5=params['C_5'],
                return_spectrum=True
            )
            return sqrt_w * (model_Rrs - Rrs_measured)
    
        result = lmfit.minimize(residual, p, method=minimizer)
        return result