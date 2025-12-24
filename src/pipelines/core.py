import numpy as np
import pandas as pd

from src.tools.parsers import ddmmssss_to_deg
from src.tools.ecef import geo_to_ecef
from src.tools.geometry import distance_3d


def decode_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decodifica lat/lon codificados DD.MMssssss -> grados decimales."""
    out = df.copy()
    out["lat1"] = out["lat1_coded"].map(ddmmssss_to_deg)
    out["lon1"] = out["lon1_coded"].map(ddmmssss_to_deg)
    out["lat2"] = out["lat2_coded"].map(ddmmssss_to_deg)
    out["lon2"] = out["lon2_coded"].map(ddmmssss_to_deg)
    return out


def ecef_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte lat/lon/h -> ECEF para ambos puntos."""
    out = df.copy()

    # Vectorizar: construir arrays
    lat1 = out["lat1"].to_numpy()
    lon1 = out["lon1"].to_numpy()
    h1   = out["h1"].to_numpy()

    lat2 = out["lat2"].to_numpy()
    lon2 = out["lon2"].to_numpy()
    h2   = out["h2"].to_numpy()

    # geo_to_ecef actual trabaja escalar; si quieres máximo rendimiento,
    # hacemos wrapper vectorizado con np.vectorize (simple) o reescribimos para arrays.
    v_geo_to_ecef = np.vectorize(geo_to_ecef, otypes=[float, float, float])

    X1, Y1, Z1 = v_geo_to_ecef(lat1, lon1, h1)
    X2, Y2, Z2 = v_geo_to_ecef(lat2, lon2, h2)

    out["X1"], out["Y1"], out["Z1"] = X1, Y1, Z1
    out["X2"], out["Y2"], out["Z2"] = X2, Y2, Z2
    return out


def compute_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula distancia ECEF y diferencias vs observada."""
    out = df.copy()

    X1 = out["X1"].to_numpy()
    Y1 = out["Y1"].to_numpy()
    Z1 = out["Z1"].to_numpy()
    X2 = out["X2"].to_numpy()
    Y2 = out["Y2"].to_numpy()
    Z2 = out["Z2"].to_numpy()

    # Vectorizado 3D
    dX = X2 - X1
    dY = Y2 - Y1
    dZ = Z2 - Z1
    dist_calc = np.sqrt(dX*dX + dY*dY + dZ*dZ)

    out["dist_calc"] = dist_calc
    out["dif_m"] = out["dist_calc"] - out["distancia"]
    out["dif_mm"] = out["dif_m"] * 1000.0

    dist_obs = out["distancia"].to_numpy()
    out["dif_ppm"] = np.where(dist_obs != 0.0, (out["dif_m"].to_numpy() / dist_obs) * 1_000_000.0, 0.0)

    return out


def run_apoyo_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Orquestador: aplica el pipeline completo.
    """
    df1 = decode_columns(df_raw)
    df2 = ecef_columns(df1)
    df3 = compute_distances(df2)
    return df3
