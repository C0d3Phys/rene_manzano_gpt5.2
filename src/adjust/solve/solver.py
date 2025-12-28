from __future__ import annotations
import pandas as pd

from src.adjust.prepare.model import AdjustmentResult, build_index
from src.adjust.solve.iteration import gauss_newton_loop
from src.adjust.stats.stats import finalize_solution


def adjust_distances_free_network(
    df: pd.DataFrame,
    names: list[str],
    coords0,
    pi_name,
    pj_name,
    d_obs,
    P,
    *,
    model: str,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> AdjustmentResult:
    model = model.lower().strip()
    if model not in ("3d", "2d"):
        raise ValueError("model debe ser '3d' o '2d'")

    idx = build_index(names)

    if verbose:
        n = len(names)
        m = len(d_obs)
        print("\n[ADJUST] LS red libre (restricciones internas)")
        print(f"[ADJUST] Modelo {model.upper()} | Obs={m} | Puntos={n} | Params={3*n}")

    coords = gauss_newton_loop(
        coords0=coords0,
        idx=idx,
        pi_name=pi_name,
        pj_name=pj_name,
        d_obs=d_obs,
        P=P,
        model=model,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
    )

    res = finalize_solution(
        df=df,
        names=names,
        coords=coords,
        idx=idx,
        pi_name=pi_name,
        pj_name=pj_name,
        d_obs=d_obs,
        P=P,
        model=model,
    )

    if verbose:
        s = res.stats
        print("\n[ADJUST] Resumen:")
        print(f"  dof       : {s['dof']}")
        print(f"  sigma0^   : {s['sigma0_hat_mm']:.3f} mm")
        print(f"  mean|v|   : {s['mean_abs_v_mm']:.3f} mm")
        print(f"  max |v|   : {s['max_abs_v_mm']:.3f} mm")

    return res
