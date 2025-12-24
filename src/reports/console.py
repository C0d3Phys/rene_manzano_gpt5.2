import pandas as pd

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "="*60)
    print("          RESUMEN DE CONTROL GEODÉSICO")
    print("="*60)
    print(f"Líneas procesadas       : {len(df)}")
    print(f"Desviación máxima       : {df['dif_mm'].abs().max():.4f} mm")
    print(f"Desviación media        : {df['dif_mm'].abs().mean():.4f} mm")
    print(f"Desviación típica       : {df['dif_mm'].std():.4f} mm")
    print(f"Precisión media (ppm)   : {df['dif_ppm'].abs().mean():.2f} ppm")
    print("="*60)

def print_detail(df: pd.DataFrame) -> None:
    print("\nDetalle de diferencias:")
    print(
        df[["punto1", "punto2", "distancia", "dist_calc", "dif_mm", "dif_ppm"]]
        .round({"distancia": 4, "dist_calc": 4, "dif_mm": 4, "dif_ppm": 2})
    )
