import pandas as pd
import numpy as np

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("          RESUMEN DE CONTROL GEODÉSICO")
    print("=" * 60)
    print(f"Líneas procesadas       : {len(df)}")

    # Compatibilidad: si existe el esquema viejo
    if "dif_mm" in df.columns and "dif_ppm" in df.columns:
        print(f"Desviación máxima       : {df['dif_mm'].abs().max():.4f} mm")
        print(f"Desviación media        : {df['dif_mm'].abs().mean():.4f} mm")
        print(f"Desviación típica       : {df['dif_mm'].std():.4f} mm")
        print(f"Precisión media (ppm)   : {df['dif_ppm'].abs().mean():.2f} ppm")
        print("=" * 60)
        return

    # Esquema nuevo 2D/3D
    req = {"dif_mm_2d", "dif_ppm_2d", "dif_mm_3d", "dif_ppm_3d"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas para resumen: {missing}. Disponibles: {list(df.columns)}")

    print("\n--- Modelo 2D (horizontal) ---")
    print(f"Desviación máxima       : {df['dif_mm_2d'].abs().max():.4f} mm")
    print(f"Desviación media        : {df['dif_mm_2d'].abs().mean():.4f} mm")
    print(f"Desviación típica       : {df['dif_mm_2d'].std():.4f} mm")
    print(f"Precisión media (ppm)   : {df['dif_ppm_2d'].abs().mean():.2f} ppm")

    print("\n--- Modelo 3D (espacial / ECEF) ---")
    print(f"Desviación máxima       : {df['dif_mm_3d'].abs().max():.4f} mm")
    print(f"Desviación media        : {df['dif_mm_3d'].abs().mean():.4f} mm")
    print(f"Desviación típica       : {df['dif_mm_3d'].std():.4f} mm")
    print(f"Precisión media (ppm)   : {df['dif_ppm_3d'].abs().mean():.2f} ppm")

    if "delta_3d_2d" in df.columns:
        print("\n--- Diferencia geométrica (3D - 2D) ---")
        print(f"Media (m)               : {df['delta_3d_2d'].mean():.4f}")
        print(f"Máxima (m)              : {df['delta_3d_2d'].max():.4f}")

    print("=" * 60)


def print_detail(df: pd.DataFrame) -> None:
    print("\nDetalle de diferencias (2D vs 3D):")

    cols = [
        "punto1",
        "punto2",
        "distancia",
        "dist_calc_2d",
        "dif_mm_2d",
        "dist_calc_3d",
        "dif_mm_3d",
        "delta_3d_2d",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas para detalle: {missing}. Disponibles: {list(df.columns)}")

    print(
        df[cols].round(
            {
                "distancia": 4,
                "dist_calc_2d": 4,
                "dif_mm_2d": 2,
                "dist_calc_3d": 4,
                "dif_mm_3d": 2,
                "delta_3d_2d": 4,
            }
        )
    )
