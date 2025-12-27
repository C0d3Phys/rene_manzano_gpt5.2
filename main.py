from pathlib import Path
import argparse
import sys
import pandas as pd

from src.io.reader import read_apoyo_file
from src.pipelines.core import run_apoyo_pipeline
from src.reports.console import print_summary, print_detail
from src.reports.qa import print_qa_resume
from src.exports.kml import build_unique_points, write_kml_points


# Placeholder: aquí irá tu ajuste real
def run_adjustment(df: pd.DataFrame, model: str = "3d") -> None:
    col = "dif_mm_3d" if model.lower() == "3d" else "dif_mm_2d"
    if col not in df.columns:
        raise KeyError(f"No existe '{col}' en el CSV. Columnas: {list(df.columns)}")

    print("\n[ADJUST] CSV cargado correctamente.")
    print(f"[ADJUST] Modelo seleccionado: {model.upper()} (residuo: {col})")
    print("[ADJUST] Aquí va el ajuste por mínimos cuadrados (A, P, l, iteraciones).")


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

    print("\nEjemplo de conversión (primera línea):")
    print(f"Lat1 codificada: {df_raw.iloc[0]['lat1_coded']} → {df.iloc[0]['lat1']:.8f}°")
    print(f"Lon1 codificada: {df_raw.iloc[0]['lon1_coded']} → {df.iloc[0]['lon1']:.8f}°")

    return 0


def cmd_adjust(out_dir: Path, model: str) -> int:
    out_file = out_dir / "resultados.csv"
    if not out_file.exists():
        print(f"[ERROR] No existe {out_file}. Ejecuta primero PREPARE.", file=sys.stderr)
        return 2

    df = pd.read_csv(out_file)
    run_adjustment(df, model=model)
    return 0


def _prompt_menu(default_data: Path, default_out: Path) -> tuple[str, Path, Path, str]:
    """
    Devuelve: (cmd, data_path, out_dir, model)
    cmd: 'prepare' o 'adjust'
    """
    while True:
        print("\n=== MENÚ ===")
        print("1) PREPARE  (decode + ECEF + distancias + QA + CSV + KML)")
        print("2) ADJUST   (cargar output/resultados.csv y ajustar)")
        print("0) SALIR")
        choice = input("Elige una opción [1/2/0]: ").strip()

        if choice == "0":
            return ("exit", default_data, default_out, "3d")

        if choice in ("1", "2"):
            cmd = "prepare" if choice == "1" else "adjust"

            model = input("Modelo [3d/2d] (Enter=3d): ").strip().lower() or "3d"
            if model not in ("3d", "2d"):
                print("Modelo inválido. Usa 3d o 2d.")
                continue

            if cmd == "prepare":
                data_in = input(f"Ruta data.mes (Enter={default_data}): ").strip()
                data_path = Path(data_in) if data_in else default_data
            else:
                data_path = default_data  # no se usa en adjust

            out_in = input(f"Directorio output (Enter={default_out}): ").strip()
            out_dir = Path(out_in) if out_in else default_out

            return (cmd, data_path, out_dir, model)

        print("Opción inválida.")


def build_parser(default_data: Path, default_out: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rene", description="Pipeline y ajuste (LS) para apoyo.")
    sub = parser.add_subparsers(dest="cmd")  # <-- NO required, para permitir menú si no hay args

    p1 = sub.add_parser("prepare", help="Decodifica, calcula distancias/diffs, QA y exporta CSV/KML.")
    p1.add_argument("--data", type=str, default=str(default_data), help="Ruta a data.mes")
    p1.add_argument("--out", type=str, default=str(default_out), help="Directorio de salida")
    p1.add_argument("--model", choices=["3d", "2d"], default="3d", help="Modelo para QA en consola")

    p2 = sub.add_parser("adjust", help="Carga output/resultados.csv y ejecuta el ajuste por mínimos cuadrados.")
    p2.add_argument("--out", type=str, default=str(default_out), help="Directorio donde está resultados.csv")
    p2.add_argument("--model", choices=["3d", "2d"], default="3d", help="Modelo a ajustar")

    return parser


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    default_data = base_dir / "data" / "data.mes"
    default_out = base_dir / "output"

    parser = build_parser(default_data, default_out)
    args = parser.parse_args()

    # Si no hay subcomando, abrir menú
    if args.cmd is None:
        cmd, data_path, out_dir, model = _prompt_menu(default_data, default_out)
        if cmd == "exit":
            print("Saliendo.")
            return 0
        if cmd == "prepare":
            return cmd_prepare(data_file=data_path, out_dir=out_dir, model=model)
        if cmd == "adjust":
            return cmd_adjust(out_dir=out_dir, model=model)
        return 0

    # Modo CLI normal (con args)
    if args.cmd == "prepare":
        return cmd_prepare(data_file=Path(args.data), out_dir=Path(args.out), model=args.model)

    if args.cmd == "adjust":
        return cmd_adjust(out_dir=Path(args.out), model=args.model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
