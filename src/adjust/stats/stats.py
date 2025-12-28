from __future__ import annotations
import numpy as np
import pandas as pd

from src.adjust.solve.constraints import inner_constraints_B
from src.adjust.prepare.observation import assemble_A_mis
from src.adjust.solve.lin_solve import solve_constrained
from src.adjust.prepare.model import AdjustmentResult


def finalize_solution(
    df: pd.DataFrame,
    names: list[str],
    coords: np.ndarray,
    idx: dict[str, int],
    pi_name: np.ndarray,
    pj_name: np.ndarray,
    d_obs: np.ndarray,
    P: np.ndarray,
    *,
    model: str,
) -> AdjustmentResult:
    n = len(names)
    npar = 3 * n
    m = len(d_obs)

    A, mis, d_calc = assemble_A_mis(coords, idx, pi_name, pj_name, d_obs, model)

    AP = A * P[:, None]
    N = A.T @ AP
    u = A.T @ (P * mis)
    B = inner_constraints_B(coords)

    dx_final = solve_constrained(N, u, B)
    v = (A @ dx_final) - mis

    dof = m - (npar - 6)
    dof = int(dof) if dof > 0 else 1
    vTPv = float(np.sum(P * (v**2)))
    sigma0_hat = float(np.sqrt(vTPv / dof))

    stats = {
        "model": model,
        "m_obs": int(m),
        "n_points": int(n),
        "n_params": int(npar),
        "dof": int(dof),
        "sigma0_hat_m": sigma0_hat,
        "sigma0_hat_mm": sigma0_hat * 1000.0,
        "mean_abs_v_mm": float(np.mean(np.abs(v)) * 1000.0),
        "max_abs_v_mm": float(np.max(np.abs(v)) * 1000.0),
    }

    points_out = pd.DataFrame(coords, columns=["X_adj", "Y_adj", "Z_adj"])
    points_out.insert(0, "name", names)

    obs_out = df.copy()
    obs_out["dist_calc_adj"] = d_calc
    obs_out["misclosure_m"] = mis
    obs_out["v_m"] = v
    obs_out["v_mm"] = v * 1000.0
    obs_out["w"] = P

    return AdjustmentResult(points=points_out, observations=obs_out, stats=stats)
