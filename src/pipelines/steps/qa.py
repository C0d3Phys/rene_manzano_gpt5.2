import pandas as pd
from src.pipelines.qa import qa_columns

def add_qa_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta QA sobre residuos 3D y 2D y guarda stats en attrs (sin pisarse)."""
    out = df.copy()

    out, stats3d = qa_columns(out, col_res_mm="dif_mm_3d", suffix="3d")
    out, stats2d = qa_columns(out, col_res_mm="dif_mm_2d", suffix="2d")

    out.attrs["qa_3d"] = stats3d
    out.attrs["qa_2d"] = stats2d
    return out
