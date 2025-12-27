from __future__ import annotations
import numpy as np
import pandas as pd

from src.adjust.constraints import inner_constraints_B
from src.adjust.model import build_index, AdjustmentResult


def _dist_and_grad(pi: np.ndarray, pj: np.ndarray) -> tuple[float, np.ndarray]:
    dvec = pi - pj
    d = float(np.sqrt(np.dot(dvec, dvec)))
    if d == 0.0 or not np.isfinite(d):
        return np.nan, np.array([np.nan, np.nan, np.nan], dtype=float)
    return d, dvec / d


def _assemble_A_mis(coords: np.ndarray, idx: dict[str, int], pi_name: np.ndarray, pj_name: np.ndarray,
                    d_obs: np.ndarray, model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = coords.shape[0]
    npar = 3 * n
    m = len(d_obs)

    A = np.zeros((m, npar), dtype=float)
    mis = np.zeros((m,), dtype=float)
    d_calc = np.zeros((m,), dtype=float)

    for k in range(m):
        i = idx[pi_name[k]]
        j = idx[pj_name[k]]

        pi = coords[i].copy()
        pj = coords[j].copy()
        if model == "2d":
            pi[2] = 0.0
            pj[2] = 0.0

        d, g = _dist_and_grad(pi, pj)
        if not np.isfinite(d):
            raise ValueError(f"Distancia inválida en obs {k} ({pi_name[k]}-{pj_name[k]})")

        d_calc[k] = d
        mis[k] = d_obs[k] - d  # observado - calculado

        # A = -J, J = ∂d/∂x
        A[k, 3*i:3*i+3] = -g
        A[k, 3*j:3*j+3] =  g

    return A, mis, d_calc


def _solve_constrained(N: np.ndarray, u: np.ndarray, B: np.ndarray) -> np.ndarray:
    Z = np.zeros((6, 6), dtype=float)
    K = np.block([[N, B],
                  [B.T, Z]])
    rhs = np.concatenate([u, np.zeros(6, dtype=float)])
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(K) @ rhs
    return sol[:N.shape[0]]


def adjust_distances_free_network(
    df: pd.DataFrame,
    names: list[str],
    coords0: np.ndarray,
    pi_name: np.ndarray,
    pj_name: np.ndarray,
    d_obs: np.ndarray,
    P: np.ndarray,
    *,
    model: str,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> AdjustmentResult:
    model = model.lower().strip()
    if model not in ("3d", "2d"):
        raise ValueError("model debe ser '3d' o '2d'")

    coords = coords0.copy()
    idx = build_index(names)
    n = len(names)
    npar = 3 * n
    m = len(d_obs)

    if verbose:
        print("\n[ADJUST] LS red libre (restricciones internas)")
        print(f"[ADJUST] Modelo {model.upper()} | Obs={m} | Puntos={n} | Params={npar}")

    for it in range(1, max_iter + 1):
        A, mis, _ = _assemble_A_mis(coords, idx, pi_name, pj_name, d_obs, model)
        AP = A * P[:, None]
        N = A.T @ AP
        u = A.T @ (P * mis)

        B = inner_constraints_B(coords)
        dx = _solve_constrained(N, u, B).reshape(n, 3)

        coords += dx
        max_corr = float(np.max(np.linalg.norm(dx, axis=1)))

        if verbose:
            print(f"[ADJUST] iter {it:02d} | max|dX|={max_corr:.3e} m | mean|mis|={float(np.mean(np.abs(mis))):.3e} m")

        if max_corr < tol:
            if verbose:
                print(f"[ADJUST] Convergió en {it} iteraciones.")
            break

    # finales
    A, mis, d_calc = _assemble_A_mis(coords, idx, pi_name, pj_name, d_obs, model)
    AP = A * P[:, None]
    N = A.T @ AP
    u = A.T @ (P * mis)
    B = inner_constraints_B(coords)
    dx_final = _solve_constrained(N, u, B)
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

    if verbose:
        print("\n[ADJUST] Resumen:")
        print(f"  dof       : {stats['dof']}")
        print(f"  sigma0^   : {stats['sigma0_hat_mm']:.3f} mm")
        print(f"  mean|v|   : {stats['mean_abs_v_mm']:.3f} mm")
        print(f"  max |v|   : {stats['max_abs_v_mm']:.3f} mm")

    return AdjustmentResult(points=points_out, observations=obs_out, stats=stats)
