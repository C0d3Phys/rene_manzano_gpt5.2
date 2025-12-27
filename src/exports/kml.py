from pathlib import Path
import pandas as pd

def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def build_unique_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye puntos únicos usando IDs si existen en el DataFrame.
    Devuelve columnas: name, lat, lon, h
    """
    id1_col = _first_existing_col(df, ["p1_id", "from_id", "station1", "id1", "pt1", "name1", "punto1", "est1"])
    id2_col = _first_existing_col(df, ["p2_id", "to_id",   "station2", "id2", "pt2", "name2", "punto2", "est2"])

    p1 = df[["lat1", "lon1", "h1"]].copy()
    p1.columns = ["lat", "lon", "h"]
    p1["name"] = df[id1_col].astype(str) if id1_col else "P1"

    p2 = df[["lat2", "lon2", "h2"]].copy()
    p2.columns = ["lat", "lon", "h"]
    p2["name"] = df[id2_col].astype(str) if id2_col else "P2"

    pts = pd.concat([p1, p2], ignore_index=True)
    pts["name"] = pts["name"].astype(str).str.strip()

    has_real_names = bool(id1_col or id2_col)

    if has_real_names:
        pts["lat_r"] = pts["lat"].round(8)
        pts["lon_r"] = pts["lon"].round(8)
        pts["h_r"]   = pts["h"].round(4)
        pts_u = pts.drop_duplicates(subset=["name", "lat_r", "lon_r", "h_r"]).copy()
        pts_u = pts_u.drop(columns=["lat_r", "lon_r", "h_r"])
        return pts_u[["name", "lat", "lon", "h"]].reset_index(drop=True)

    pts["lat_r"] = pts["lat"].round(8)
    pts["lon_r"] = pts["lon"].round(8)
    pts["h_r"]   = pts["h"].round(4)

    pts_u = pts.drop_duplicates(subset=["lat_r", "lon_r", "h_r"]).copy().reset_index(drop=True)
    pts_u["name"] = pts_u.apply(lambda r: f"P_{r['lat_r']:.8f}_{r['lon_r']:.8f}", axis=1)
    pts_u = pts_u.drop(columns=["lat_r", "lon_r", "h_r"])
    return pts_u[["name", "lat", "lon", "h"]]

def write_kml_points(points: pd.DataFrame, kml_path: Path, document_name: str = "Puntos únicos") -> None:
    """Escribe un KML con placemarks para cada punto."""
    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                 .replace('"', "&quot;").replace("'", "&apos;"))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "  <Document>",
        f"    <name>{esc(document_name)}</name>",
    ]

    for _, r in points.iterrows():
        name = esc(str(r["name"]))
        lat = float(r["lat"])
        lon = float(r["lon"])
        h = float(r["h"]) if pd.notna(r["h"]) else 0.0
        lines += [
            "    <Placemark>",
            f"      <name>{name}</name>",
            "      <Point>",
            f"        <coordinates>{lon:.8f},{lat:.8f},{h:.3f}</coordinates>",
            "      </Point>",
            "    </Placemark>",
        ]

    lines += ["  </Document>", "</kml>"]
    kml_path.write_text("\n".join(lines), encoding="utf-8")
