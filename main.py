from pathlib import Path

from src.io.reader import read_apoyo_file
from src.pipelines.core import run_apoyo_pipeline
from src.reports.console import print_summary, print_detail


def main():
    # Ruta al directorio de datos (al nivel de main.py)
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    out_dir = base_dir / "output"

    data_file = data_dir / "data.mes"
    out_file = out_dir / "resultados.csv"

    # Crear directorio de salida si no existe
    out_dir.mkdir(exist_ok=True)

    # Leer datos
    df_raw = read_apoyo_file(str(data_file))

    # Ejecutar pipeline
    df = run_apoyo_pipeline(df_raw)

    # Reportes en consola
    print_summary(df)
    print_detail(df)

    # Exportar DataFrame final a .data (CSV con otra extensión)
    df.to_csv(
        out_file,
        sep=",",
        index=False,
        float_format="%.6f"
    )

    print(f"\nArchivo generado correctamente: {out_file}")

    # Ejemplo de conversión (sanity check)
    print("\nEjemplo de conversión (primera línea):")
    print(f"Lat1 codificada: {df_raw.iloc[0]['lat1_coded']} → {df.iloc[0]['lat1']:.8f}°")
    print(f"Lon1 codificada: {df_raw.iloc[0]['lon1_coded']} → {df.iloc[0]['lon1']:.8f}°")


if __name__ == "__main__":
    main()
