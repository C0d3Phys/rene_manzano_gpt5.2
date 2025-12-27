import numpy as np
import pandas as pd

from src.tools.parsers import ddmmssss_to_deg
from src.tools.ecef import geo_to_ecef_np
from src.tools.geometry import distance_3d_np


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
    """Calcula distancia ECEF y diferencias vs observada (vectorizado real)."""
    out = df.copy()

    dist_calc = distance_3d_np(
        out["X1"].to_numpy(dtype=float),
        out["Y1"].to_numpy(dtype=float),
        out["Z1"].to_numpy(dtype=float),
        out["X2"].to_numpy(dtype=float),
        out["Y2"].to_numpy(dtype=float),
        out["Z2"].to_numpy(dtype=float),
    )

    out["dist_calc"] = dist_calc

    # Diferencias
    dist_obs = out["distancia"].to_numpy(dtype=float)
    dif_m = dist_calc - dist_obs

    out["dif_m"]  = dif_m
    out["dif_mm"] = dif_m * 1000.0
    out["dif_ppm"] = np.where(dist_obs != 0.0, (dif_m / dist_obs) * 1_000_000.0, 0.0)

    return out


def run_apoyo_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Orquestador: aplica el pipeline completo."""
    df1 = decode_columns(df_raw)
    df2 = ecef_columns(df1)
    df3 = compute_distances(df2)
    return df3
