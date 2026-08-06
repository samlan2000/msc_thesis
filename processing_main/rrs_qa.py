import numpy as np

def spectral_roughness(rrs, wavelengths):
    rrs = np.asarray(rrs, dtype=float)
    wvl = np.asarray(wavelengths, dtype=float)

    mask = np.isfinite(rrs)
    rrs = rrs[mask]
    wvl = wvl[mask]

    if rrs.size < 5:
        return np.nan

    # second derivative wrt wavelength
    d2 = np.gradient(np.gradient(rrs, wvl), wvl)

    roughness = np.nanmean(np.abs(d2)) / np.nanmean(np.abs(rrs))
    return roughness


def has_nan_spectral_gaps(rrs, wavelengths, max_gap_nm=10):
    rrs = np.asarray(rrs)
    wvl = np.asarray(wavelengths)

    valid = np.isfinite(rrs)

    if valid.sum() < 2:
        return True, None

    nan_wvl = wvl[~valid]

    if nan_wvl.size == 0:
        return False, 0

    gaps = []
    start = nan_wvl[0]
    prev = nan_wvl[0]

    spacing = np.median(np.diff(wvl))

    for wl in nan_wvl[1:]:
        if wl - prev <= spacing * 1.5:
            prev = wl
        else:
            gaps.append(prev - start)
            start = prev = wl

    gaps.append(prev - start)

    max_gap = max(gaps) if gaps else 0

    print(max_gap)

    return max_gap > max_gap_nm, max_gap