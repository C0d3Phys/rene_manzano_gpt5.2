import pandas as pd

DEFAULT_COLUMNS = [
    'lat1_coded', 'lon1_coded', 'h1',
    'lat2_coded', 'lon2_coded', 'h2',
    'distancia', 'punto1', 'punto2'
]

def read_apoyo_file(path: str, columns=DEFAULT_COLUMNS) -> pd.DataFrame:
    """
    Lee archivo de control geodésico con separadores coma o espacios.
    Limpia filas malas y tipa columnas.
    """
    df = pd.read_csv(
        path,
        delimiter=r"\s*,\s*|\s+",
        engine="python",
        comment="#",
        header=None,
        on_bad_lines="skip",
    )

    # Requiere al menos 9 campos
    df = df.dropna(thresh=len(columns)).reset_index(drop=True)

    df = df.apply(pd.to_numeric, errors="coerce")
    df.columns = columns

    # Tipos
    df["punto1"] = df["punto1"].astype("Int64")
    df["punto2"] = df["punto2"].astype("Int64")

    # Filas válidas mínimas
    df = df.dropna(subset=["lat1_coded", "lon1_coded", "lat2_coded", "lon2_coded"])

    return df
