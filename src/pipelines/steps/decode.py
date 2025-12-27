import pandas as pd
from src.tools.parsers import ddmmssss_to_deg

def decode_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Decodifica lat/lon codificados DD.MMssssss -> grados decimales."""
    out = df.copy()
    out["lat1"] = out["lat1_coded"].map(ddmmssss_to_deg)
    out["lon1"] = out["lon1_coded"].map(ddmmssss_to_deg)
    out["lat2"] = out["lat2_coded"].map(ddmmssss_to_deg)
    out["lon2"] = out["lon2_coded"].map(ddmmssss_to_deg)
    return out
