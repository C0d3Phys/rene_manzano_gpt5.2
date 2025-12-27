import numpy as np
import pandas as pd
from src.tools.geometry import distance_3d_np, distance_2d_np

def compute_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula distancias 2D/3D y diferencias vs observada (vectorizado real)."""
    out = df.copy()

    X1 = out["X1"].to_numpy(dtype=float)
    Y1 = out["Y1"].to_numpy(dtype=float)
    Z1 = out["Z1"].to_numpy(dtype=float)
    X2 = out["X2"].to_numpy(dtype=float)
    Y2 = out["Y2"].to_numpy(dtype=float)
    Z2 = out["Z2"].to_numpy(dtype=float)

    dist_3d = distance_3d_np(X1, Y1, Z1, X2, Y2, Z2)
    dist_2d = distance_2d_np(X1, Y1, X2, Y2)

    out["dist_calc_3d"] = dist_3d
    out["dist_calc_2d"] = dist_2d
    out["delta_3d_2d"]  = dist_3d - dist_2d

    dist_obs = out["distancia"].to_numpy(dtype=float)

    out["dif_m_3d"]   = dist_3d - dist_obs
    out["dif_mm_3d"]  = out["dif_m_3d"] * 1000.0
    out["dif_ppm_3d"] = np.where(dist_obs != 0.0, (out["dif_m_3d"] / dist_obs) * 1_000_000.0, 0.0)

    out["dif_m_2d"]   = dist_2d - dist_obs
    out["dif_mm_2d"]  = out["dif_m_2d"] * 1000.0
    out["dif_ppm_2d"] = np.where(dist_obs != 0.0, (out["dif_m_2d"] / dist_obs) * 1_000_000.0, 0.0)

    return out
