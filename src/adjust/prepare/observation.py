from __future__ import annotations
import numpy as np

def dist_and_grad(pi: np.ndarray, pj: np.ndarray) -> tuple[float, np.ndarray]:
    dvec = pi - pj
    d = float(np.sqrt(np.dot(dvec, dvec)))
    if d == 0.0 or not np.isfinite(d):
        return np.nan, np.array([np.nan, np.nan, np.nan], dtype=float)
    return d, dvec / d


def assemble_A_mis(
    coords: np.ndarray,
    idx: dict[str, int],
    pi_name: np.ndarray,
    pj_name: np.ndarray,
    d_obs: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        d, g = dist_and_grad(pi, pj)
        if not np.isfinite(d):
            raise ValueError(f"Distancia inválida en obs {k} ({pi_name[k]}-{pj_name[k]})")

        d_calc[k] = d
        mis[k] = d_obs[k] - d

        # A = -J
        A[k, 3*i:3*i+3] = -g
        A[k, 3*j:3*j+3] =  g

    return A, mis, d_calc
