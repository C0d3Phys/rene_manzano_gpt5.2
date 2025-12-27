from pathlib import Path
import sys
import pandas as pd

from src.io.reader import read_apoyo_file
from src.pipelines.core import run_apoyo_pipeline
from src.reports.console import print_summary, print_detail
from src.reports.qa import print_qa_resume
from src.exports.kml import build_unique_points, write_kml_points
from src.adjust.runner import run_adjustment


def cmd_prepare(data_file: Path, out_dir: Path, model: str) -> int:
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "resultados.csv"
    kml_file = out_dir / "puntos_unicos.kml"

    df_raw = read_apoyo_file(str(data_file))
    df = run_apoyo_pipeline(df_raw)

    col_res = "dif_mm_3d" if model.lower() == "3d" else "dif_mm_2d"
    print_qa_resume(df, col_res_mm=col_res)

    print_summary(df)
    print_detail(df)

    df.to_csv(out_file, sep=",", index=False, float_format="%.6f")
    print(f"\nArchivo generado correctamente: {out_file}")

    pts_unique = build_unique_points(df)
    write_kml_points(pts_unique, kml_file, document_name="Puntos únicos (lat/lon decodificados)")
    print(f"KML generado correctamente: {kml_file}")

    # Sanity check
    print("\nEjemplo de conversión (primera línea):")
    print(f"Lat1 codificada: {df_raw.iloc[0]['lat1_coded']} → {df.iloc[0]['lat1']:.8f}°")
    print(f"Lon1 codificada: {df_raw.iloc[0]['lon1_coded']} → {df.iloc[0]['lon1']:.8f}°")

    return 0


def cmd_adjust(out_dir: Path, model: str) -> int:
    out_file = out_dir / "resultados.csv"
    if not out_file.exists():
        print(f"[ERROR] No existe {out_file}. Ejecuta primero: prepare", file=sys.stderr)
        return 2

    df = pd.read_csv(out_file)
    run_adjustment(df, model=model)
    return 0
