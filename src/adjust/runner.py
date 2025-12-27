import pandas as pd

def run_adjustment(df: pd.DataFrame, model: str = "3d") -> None:
    col = "dif_mm_3d" if model.lower() == "3d" else "dif_mm_2d"
    if col not in df.columns:
        raise KeyError(f"No existe '{col}' en el CSV. Columnas: {list(df.columns)}")

    print("\n[ADJUST] CSV cargado correctamente.")
    print(f"[ADJUST] Modelo seleccionado: {model.upper()} (residuo: {col})")
    print("[ADJUST] Aquí va el ajuste por mínimos cuadrados (A, P, l, iteraciones).")
