from pathlib import Path
import argparse

from src.cli.handlers import cmd_prepare, cmd_adjust


def _prompt_menu(default_data: Path, default_out: Path) -> tuple[str, Path, Path, str]:
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

            model = (input("Modelo [3d/2d] (Enter=3d): ").strip().lower() or "3d")
            if model not in ("3d", "2d"):
                print("Modelo inválido. Usa 3d o 2d.")
                continue

            data_path = default_data
            if cmd == "prepare":
                data_in = input(f"Ruta data.mes (Enter={default_data}): ").strip()
                data_path = Path(data_in) if data_in else default_data

            out_in = input(f"Directorio output (Enter={default_out}): ").strip()
            out_dir = Path(out_in) if out_in else default_out

            return (cmd, data_path, out_dir, model)

        print("Opción inválida.")


def build_parser(default_data: Path, default_out: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rene", description="Pipeline y ajuste (LS) para apoyo.")
    sub = parser.add_subparsers(dest="cmd")  # sin required para permitir menú

    p1 = sub.add_parser("prepare", help="Prepara: decode/ECEF/distancias/QA/exporta.")
    p1.add_argument("--data", type=str, default=str(default_data))
    p1.add_argument("--out", type=str, default=str(default_out))
    p1.add_argument("--model", choices=["3d", "2d"], default="3d")

    p2 = sub.add_parser("adjust", help="Ajuste: usa output/resultados.csv.")
    p2.add_argument("--out", type=str, default=str(default_out))
    p2.add_argument("--model", choices=["3d", "2d"], default="3d")

    return parser


def run_cli() -> int:
    base_dir = Path(__file__).resolve().parents[2]  # sube desde src/cli/app.py hasta raíz
    default_data = base_dir / "data" / "data.mes"
    default_out = base_dir / "output"

    parser = build_parser(default_data, default_out)
    args = parser.parse_args()

    if args.cmd is None:
        cmd, data_path, out_dir, model = _prompt_menu(default_data, default_out)
        if cmd == "exit":
            print("Saliendo.")
            return 0
        if cmd == "prepare":
            return cmd_prepare(data_file=data_path, out_dir=out_dir, model=model)
        return cmd_adjust(out_dir=out_dir, model=model)

    if args.cmd == "prepare":
        return cmd_prepare(data_file=Path(args.data), out_dir=Path(args.out), model=args.model)

    if args.cmd == "adjust":
        return cmd_adjust(out_dir=Path(args.out), model=args.model)

    return 0
