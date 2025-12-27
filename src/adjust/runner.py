import pandas as pd

from src.adjust.model import extract_adjustment_data, AdjustmentResult
from src.adjust.weights import weights_from_distance
from src.adjust.solver import adjust_distances_free_network


def run_adjustment(
    df: pd.DataFrame,
    model: str = "3d",
    *,
    id1: str = "punto1",
    id2: str = "punto2",
    max_iter: int = 15,
    tol: float = 1e-6,
    sigma0_mm: float = 5.0,
    ppm: float = 1.0,
    verbose: bool = True,
) -> AdjustmentResult:
    data = extract_adjustment_data(df, id1=id1, id2=id2)
    P = weights_from_distance(data.d_obs, sigma0_mm=sigma0_mm, ppm=ppm)

    return adjust_distances_free_network(
        df=df,
        names=data.names,
        coords0=data.coords0,
        pi_name=data.pi,
        pj_name=data.pj,
        d_obs=data.d_obs,
        P=P,
        model=model,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )
