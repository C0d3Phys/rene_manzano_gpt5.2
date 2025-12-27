from __future__ import annotations
import numpy as np

from src.adjust.constraints import inner_constraints_B
from src.adjust.observation import assemble_A_mis
from src.adjust.lin_solve import solve_constrained


def gauss_newton_loop(
    coords0: np.ndarray,
    idx: dict[str, int],
    pi_name: np.ndarray,
    pj_name: np.ndarray,
    d_obs: np.ndarray,
    P: np.ndarray,
    *,
    model: str,
    max_iter: int,
    tol: float,
    verbose: bool,
) -> np.ndarray:
    coords = coords0.copy()
    n = coords.shape[0]

    for it in range(1, max_iter + 1):
        A, mis, _ = assemble_A_mis(coords, idx, pi_name, pj_name, d_obs, model)

        AP = A * P[:, None]
        N = A.T @ AP
        u = A.T @ (P * mis)

        B = inner_constraints_B(coords)
        dx = solve_constrained(N, u, B).reshape(n, 3)

        coords += dx
        max_corr = float(np.max(np.linalg.norm(dx, axis=1)))

        if verbose:
            print(f"[ADJUST] iter {it:02d} | max|dX|={max_corr:.3e} m | mean|mis|={float(np.mean(np.abs(mis))):.3e} m")

        if max_corr < tol:
            if verbose:
                print(f"[ADJUST] Convergió en {it} iteraciones.")
            break

    return coords
