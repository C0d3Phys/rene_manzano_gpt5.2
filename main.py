from pathlib import Path
import pandas as pd

from src.io.reader import read_apoyo_file
from src.pipelines.core import run_apoyo_pipeline
from src.reports.console import print_summary, print_detail
from src.reports.qa import print_qa_resume
from src.exports.kml import build_unique_points, write_kml_points

def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    out_dir = base_dir / "output"

    data_file = data_dir / "data.mes"
    out_file = out_dir / "resultados.csv"
    kml_file = out_dir / "puntos_unicos.kml"

    out_dir.mkdir(exist_ok=True)

    df_raw = read_apoyo_file(str(data_file))
    df = run_apoyo_pipeline(df_raw)

    # QA (elige 3D o 2D)
    print_qa_resume(df, col_res_mm="dif_mm_3d")

    # Reportes
    print_summary(df)
    print_detail(df)

    # Export CSV
    df.to_csv(out_file, sep=",", index=False, float_format="%.6f")
    print(f"\nArchivo generado correctamente: {out_file}")

    # Export KML
    pts_unique = build_unique_points(df)
    write_kml_points(pts_unique, kml_file, document_name="Puntos únicos (lat/lon decodificados)")
    print(f"KML generado correctamente: {kml_file}")

    # Sanity check
    print("\nEjemplo de conversión (primera línea):")
    print(f"Lat1 codificada: {df_raw.iloc[0]['lat1_coded']} → {df.iloc[0]['lat1']:.8f}°")
    print(f"Lon1 codificada: {df_raw.iloc[0]['lon1_coded']} → {df.iloc[0]['lon1']:.8f}°")

if __name__ == "__main__":
    main()
