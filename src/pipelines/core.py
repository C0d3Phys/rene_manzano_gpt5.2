import numpy as np
import pandas as pd

from src.tools.parsers import ddmmssss_to_deg
from src.tools.ecef import geo_to_ecef_np
from src.tools.geometry import distance_3d_np, distance_2d_np
from src.pipelines.qa import qa_columns


def decode_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decodifica lat/lon codificados DD.MMssssss -> grados decimales."""
    out = df.copy()
    out["lat1"] = out["lat1_coded"].map(ddmmssss_to_deg)
    out["lon1"] = out["lon1_coded"].map(ddmmssss_to_deg)
    out["lat2"] = out["lat2_coded"].map(ddmmssss_to_deg)
    out["lon2"] = out["lon2_coded"].map(ddmmssss_to_deg)
    return out


def ecef_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte lat/lon/h -> ECEF para ambos puntos (vectorizado real)."""
    out = df.copy()

    lat1 = out["lat1"].to_numpy(dtype=float)
    lon1 = out["lon1"].to_numpy(dtype=float)
    h1   = out["h1"].to_numpy(dtype=float)

    lat2 = out["lat2"].to_numpy(dtype=float)
    lon2 = out["lon2"].to_numpy(dtype=float)
    h2   = out["h2"].to_numpy(dtype=float)

    X1, Y1, Z1 = geo_to_ecef_np(lat1, lon1, h1)
    X2, Y2, Z2 = geo_to_ecef_np(lat2, lon2, h2)

    out["X1"], out["Y1"], out["Z1"] = X1, Y1, Z1
    out["X2"], out["Y2"], out["Z2"] = X2, Y2, Z2
    return out


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
    out["delta_3d_2d"] = dist_3d - dist_2d  # cuánto “sube” por Z

    dist_obs = out["distancia"].to_numpy(dtype=float)

    # Residuales (calc - obs)
    out["dif_m_3d"]  = dist_3d - dist_obs
    out["dif_mm_3d"] = out["dif_m_3d"] * 1000.0
    out["dif_ppm_3d"] = np.where(dist_obs != 0.0, (out["dif_m_3d"] / dist_obs) * 1_000_000.0, 0.0)

    out["dif_m_2d"]  = dist_2d - dist_obs
    out["dif_mm_2d"] = out["dif_m_2d"] * 1000.0
    out["dif_ppm_2d"] = np.where(dist_obs != 0.0, (out["dif_m_2d"] / dist_obs) * 1_000_000.0, 0.0)

    return out

def run_apoyo_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    df1 = decode_columns(df_raw)
    df2 = ecef_columns(df1)
    df3 = compute_distances(df2)

    df3, stats3d = qa_columns(df3, col_res_mm="dif_mm_3d")
    df3, stats2d = qa_columns(df3, col_res_mm="dif_mm_2d")

    # Opcional: guardar resúmenes en attrs para inspección
    df3.attrs["qa_3d"] = stats3d
    df3.attrs["qa_2d"] = stats2d
    return df3