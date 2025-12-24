from pathlib import Path

from src.io.reader import read_apoyo_file
from src.pipelines.core import run_apoyo_pipeline
from src.reports.console import print_summary, print_detail


def main():
    # Ruta al directorio de datos (al nivel de main.py)
    data_dir = Path(__file__).resolve().parent / "data"
    data_file = data_dir / "data.mes"

    # Leer datos
    df_raw = read_apoyo_file(str(data_file))

    # Ejecutar pipeline
    df = run_apoyo_pipeline(df_raw)

    # Reportes
    print_summary(df)
    print_detail(df)

    # Ejemplo de conversión
    print("\nEjemplo de conversión (primera línea):")
    print(f"Lat1 codificada: {df_raw.iloc[0]['lat1_coded']} → {df.iloc[0]['lat1']:.8f}°")
    print(f"Lon1 codificada: {df_raw.iloc[0]['lon1_coded']} → {df.iloc[0]['lon1']:.8f}°")


if __name__ == "__main__":
    main()
