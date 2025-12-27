import pandas as pd
from src.tools.ecef import geo_to_ecef_np

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
