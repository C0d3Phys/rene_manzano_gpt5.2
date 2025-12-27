import pandas as pd
from src.pipelines.qa import qa_columns

def print_qa_resume(df: pd.DataFrame, col_res_mm: str = "dif_mm_3d") -> None:
    _, stats = qa_columns(df, col_res_mm=col_res_mm)
    print("\nQA resumen:")
    print(
        f"n={stats['n']} | mean={stats['mean_mm']:.3f} mm | std={stats['std_mm']:.3f} mm "
        f"| p95={stats['p95_mm']:.3f} mm | outliers={stats['count_outlier']}"
    )
