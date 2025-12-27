from pathlib import Path
import pandas as pd

from src.io.reader import read_apoyo_file
from src.pipelines.core import run_apoyo_pipeline
from src.reports.console import print_summary, print_detail

def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_unique_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye puntos únicos usando IDs si existen en el DataFrame.
    Devuelve columnas: name, lat, lon, h

    Soporta IDs con nombres típicos:
      - p1_id / p2_id
      - from_id / to_id
      - station1 / station2
      - id1 / id2
      - pt1 / pt2
      - name1 / name2
    Si no existen, genera un nombre estable: P_<lat>_<lon>
    """
    # Detectar columnas de ID si existen
    id1_col = _first_existing_col(df, ["p1_id", "from_id", "station1", "id1", "pt1", "name1", "punto1", "est1"])
    id2_col = _first_existing_col(df, ["p2_id", "to_id",   "station2", "id2", "pt2", "name2", "punto2", "est2"])

    # Construir puntos
    p1 = df[["lat1", "lon1", "h1"]].copy()
    p1.columns = ["lat", "lon", "h"]
    if id1_col:
        p1["name"] = df[id1_col].astype(str)
    else:
        p1["name"] = "P1"

    p2 = df[["lat2", "lon2", "h2"]].copy()
    p2.columns = ["lat", "lon", "h"]
    if id2_col:
        p2["name"] = df[id2_col].astype(str)
    else:
        p2["name"] = "P2"

    pts = pd.concat([p1, p2], ignore_index=True)

    # Normalizar strings (por si vienen con espacios)
    pts["name"] = pts["name"].str.strip()

    # Deduplicación principal:
    # - Si hay nombres reales: deduplicamos por name (y si hay conflicto, también por coord)
    # - Si no hay nombres reales: deduplicamos por coord redondeada
    has_real_names = bool(id1_col or id2_col)

    if has_real_names:
        # Si un mismo nombre aparece con coordenadas distintas, lo separamos para que lo notes.
        pts["lat_r"] = pts["lat"].round(8)
        pts["lon_r"] = pts["lon"].round(8)
        pts["h_r"]   = pts["h"].round(4)

        # Primero agrupar por (name, coord)
        pts_u = pts.drop_duplicates(subset=["name", "lat_r", "lon_r", "h_r"]).copy()
        pts_u = pts_u.drop(columns=["lat_r", "lon_r", "h_r"])
        return pts_u[["name", "lat", "lon", "h"]].reset_index(drop=True)

    # Sin IDs: nombre estable por coordenadas
    pts["lat_r"] = pts["lat"].round(8)
    pts["lon_r"] = pts["lon"].round(8)
    pts["h_r"]   = pts["h"].round(4)

    pts_u = pts.drop_duplicates(subset=["lat_r", "lon_r", "h_r"]).copy()
    pts_u = pts_u.reset_index(drop=True)

    # Nombre estable: P_lat_lon (con redondeo)
    pts_u["name"] = pts_u.apply(lambda r: f"P_{r['lat_r']:.8f}_{r['lon_r']:.8f}", axis=1)

    pts_u = pts_u.drop(columns=["lat_r", "lon_r", "h_r"])
    return pts_u[["name", "lat", "lon", "h"]]

def write_kml_points(points: pd.DataFrame, kml_path: Path, document_name: str = "Puntos únicos"):
    """
    Escribe un KML con placemarks para cada punto.
    points: DataFrame con columnas name, lat, lon, h
    """
    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                 .replace('"', "&quot;").replace("'", "&apos;"))

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    lines.append("  <Document>")
    lines.append(f"    <name>{esc(document_name)}</name>")

    for _, r in points.iterrows():
        name = esc(str(r["name"]))
        lat = float(r["lat"])
        lon = float(r["lon"])
        h = float(r["h"]) if pd.notna(r["h"]) else 0.0

        # KML usa: lon,lat,alt
        lines.append("    <Placemark>")
        lines.append(f"      <name>{name}</name>")
        lines.append("      <Point>")
        lines.append(f"        <coordinates>{lon:.8f},{lat:.8f},{h:.3f}</coordinates>")
        lines.append("      </Point>")
        lines.append("    </Placemark>")

    lines.append("  </Document>")
    lines.append("</kml>")

    kml_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    out_dir = base_dir / "output"

    data_file = data_dir / "data.mes"
    out_file = out_dir / "resultados.csv"
    kml_file = out_dir / "puntos_unicos.kml"

    out_dir.mkdir(exist_ok=True)

    # Leer + pipeline
    df_raw = read_apoyo_file(str(data_file))
    df = run_apoyo_pipeline(df_raw)

    # Reportes
    print_summary(df)
    print_detail(df)

    # Export resultados
    df.to_csv(out_file, sep=",", index=False, float_format="%.6f")
    print(f"\nArchivo generado correctamente: {out_file}")

    # Export KML puntos únicos
    pts_unique = build_unique_points(df)
    write_kml_points(pts_unique, kml_file, document_name="Puntos únicos (lat/lon decodificados)")
    print(f"KML generado correctamente: {kml_file}")

    # Sanity check
    print("\nEjemplo de conversión (primera línea):")
    print(f"Lat1 codificada: {df_raw.iloc[0]['lat1_coded']} → {df.iloc[0]['lat1']:.8f}°")
    print(f"Lon1 codificada: {df_raw.iloc[0]['lon1_coded']} → {df.iloc[0]['lon1']:.8f}°")


if __name__ == "__main__":
    main()
