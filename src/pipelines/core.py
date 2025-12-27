import pandas as pd
from src.pipelines.steps.decode import decode_columns
from src.pipelines.steps.ecef import ecef_columns
from src.pipelines.steps.distances import compute_distances
from src.pipelines.steps.qa import add_qa_flags

def run_apoyo_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = decode_columns(df_raw)
    df = ecef_columns(df)
    df = compute_distances(df)
    df = add_qa_flags(df)
    return df
