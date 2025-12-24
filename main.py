from src.io.reader import read_apoyo_file
from src.pipelines.core import run_apoyo_pipeline
from src.reports.console import print_summary, print_detail

def main():
    df_raw = read_apoyo_file("datos_apoyo.txt")
    df = run_apoyo_pipeline(df_raw)
    print_summary(df)
    print_detail(df)

    # Ejemplo de conversión
    print("\nEjemplo de conversión (primera línea):")
    print(f"Lat1 codificada: {df_raw.iloc[0]['lat1_coded']} → {df.iloc[0]['lat1']:.8f}°")
    print(f"Lon1 codificada: {df_raw.iloc[0]['lon1_coded']} → {df.iloc[0]['lon1']:.8f}°")

if __name__ == "__main__":
    main()
