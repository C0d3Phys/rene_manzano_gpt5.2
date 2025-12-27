import numpy as np

def weights_from_distance(d_obs: np.ndarray, *, sigma0_mm: float, ppm: float) -> np.ndarray:
    """
    sigma(d) = sqrt( (sigma0)^2 + (ppm*d)^2 ), con ppm en ppm y d en m.
    Devuelve w = 1/sigma^2 (1/m^2)
    """
    sigma0_m = sigma0_mm / 1000.0
    sigma_ppm_m = (ppm * 1e-6) * d_obs
    sigma_m = np.sqrt(sigma0_m**2 + sigma_ppm_m**2)
    return np.where(sigma_m > 0, 1.0 / (sigma_m**2), 0.0)
