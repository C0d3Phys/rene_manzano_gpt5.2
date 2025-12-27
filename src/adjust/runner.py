# src/adjust/runner.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AdjustmentResult:
    points: pd.DataFrame          # coords ajustadas
    observations: pd.DataFrame     # obs con residuales
    stats: dict                   # resumen numérico


def _unique_points_from_pairs(df: pd.DataFrame, id1: str, id2: str) -> list[str]:
    a = df[id1].astype(str).tolist()
    b = df[id2].astype(str).tolist()
    pts = sorted(set(a) | set(b))
    return pts


def _initial_coords_from_df(df: pd.DataFrame, id1: str, id2: str) -> tuple[list[str], np.ndarray]:
    """
    Construye X0,Y0,Z0 inicial por punto a partir de (X1,Y1,Z1) y (X2,Y2,Z2)
    promediando apariciones.
    Devuelve (point_names, coords0[n,3])
    """
    # tabla larga: (name, X, Y, Z)
    p1 = pd.DataFrame({
        "name": df[id1].astype(str),
        "X": df["X1"].astype(float),
        "Y": df["Y1"].astype(float),
        "Z": df["Z1"].astype(float),
    })
    p2 = pd.DataFrame({
        "name": df[id2].astype(str),
        "X": df["X2"].astype(float),
        "Y": df["Y2"].astype(float),
        "Z": df["Z2"].astype(float),
    })
    pts = pd.concat([p1, p2], ignore_index=True)
    # promedio por nombre (si el mismo punto aparece repetido con pequeñas variaciones)
    g = pts.groupby("name", as_index=False).mean(numeric_only=True)
    names = g["name"].tolist()
    coords0 = g[["X", "Y", "Z"]].to_numpy(dtype=float)
    return names, coords0


def _build_index(names: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}


def _compute_dist_and_partials(pi: np.ndarray, pj: np.ndarray) -> tuple[float, np.ndarray]:
    """
    pi, pj: shape (3,)
    Returns: (d, g) where g = (pi - pj)/d (shape (3,))
    """
    dvec = pi - pj
    d = float(np.sqrt(np.dot(dvec, dvec)))
    if d == 0.0 or not np.isfinite(d):
        return np.nan, np.array([np.nan, np.nan, np.nan], dtype=float)
    g = dvec / d
    return d, g


def _weights_from_distance(d_obs: np.ndarray, sigma0_mm: float, ppm: float) -> np.ndarray:
    """
    sigma(d) = sqrt( (sigma0)^2 + (ppm * d)^2 )
    sigma0_mm en mm. ppm en ppm.
    d_obs en metros.
    Devuelve pesos w = 1/sigma^2 (en 1/m^2)
    """
    sigma0_m = sigma0_mm / 1000.0
    sigma_ppm_m = (ppm * 1e-6) * d_obs
    sigma_m = np.sqrt(sigma0_m**2 + sigma_ppm_m**2)
    w = np.where(sigma_m > 0, 1.0 / (sigma_m**2), 0.0)
    return w


def _inner_constraints_B(coords: np.ndarray) -> np.ndarray:
    """
    Construye matriz B (nparam x 6) para restricciones internas.
    coords: (n,3) = [X,Y,Z] por punto.
    Parámetros apilados: [dX1,dY1,dZ1, dX2,dY2,dZ2, ...]
    Restricciones:
      1) Σ dX = 0
      2) Σ dY = 0
      3) Σ dZ = 0
      4) Σ (Y*dZ - Z*dY) = 0
      5) Σ (Z*dX - X*dZ) = 0
      6) Σ (X*dY - Y*dX) = 0
    """
    n = coords.shape[0]
    npar = 3 * n
    B = np.zeros((npar, 6), dtype=float)

    for i in range(n):
        X, Y, Z = coords[i]
        ix = 3*i
        iy = 3*i + 1
        iz = 3*i + 2

        # traslaciones
        B[ix, 0] = 1.0  # dX en sum dX
        B[iy, 1] = 1.0  # dY
        B[iz, 2] = 1.0  # dZ

        # rotación: Y*dZ - Z*dY
        B[iy, 3] = -Z
        B[iz, 3] =  Y

        # rotación: Z*dX - X*dZ
        B[ix, 4] =  Z
        B[iz, 4] = -X

        # rotación: X*dY - Y*dX
        B[ix, 5] = -Y
        B[iy, 5] =  X

    return B


def run_adjustment(
    df: pd.DataFrame,
    model: str = "3d",
    *,
    id1: str = "punto1",
    id2: str = "punto2",
    max_iter: int = 15,
    tol: float = 1e-6,        # metros (corrección máxima)
    sigma0_mm: float = 5.0,   # constante en mm
    ppm: float = 1.0,         # término proporcional
    verbose: bool = True,
) -> AdjustmentResult:
    """
    Ajuste por mínimos cuadrados de una red de distancias (Gauss-Newton) en ECEF.

    Requiere en df:
      - id1,id2 (punto1,punto2) o los que le pases
      - X1,Y1,Z1, X2,Y2,Z2
      - distancia (observada, metros)

    model="3d": usa distancia 3D ECEF (recomendado en tu caso)
    model="2d": usa solo X,Y (no recomendado para tu dataset por alturas)
    """
    model = model.lower().strip()
    if model not in ("3d", "2d"):
        raise ValueError("model debe ser '3d' o '2d'")

    req = {id1, id2, "distancia", "X1", "Y1", "Z1", "X2", "Y2", "Z2"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}. Disponibles: {list(df.columns)}")

    # Inicial: puntos y coords
    names, coords = _initial_coords_from_df(df, id1=id1, id2=id2)
    idx = _build_index(names)
    n = len(names)
    npar = 3 * n
    m = len(df)

    # Observaciones
    p_i = df[id1].astype(str).to_numpy()
    p_j = df[id2].astype(str).to_numpy()
    d_obs = df["distancia"].to_numpy(dtype=float)

    # Pesos
    w = _weights_from_distance(d_obs, sigma0_mm=sigma0_mm, ppm=ppm)
    # P como vector (diagonal)
    P = w

    if verbose:
        print("\n[ADJUST] Iniciando ajuste LS (red libre con restricciones internas)")
        print(f"[ADJUST] Modelo: {model.upper()} | Observaciones: {m} | Puntos: {n} | Parámetros: {npar}")
        print(f"[ADJUST] Pesos: sigma0={sigma0_mm} mm, ppm={ppm}")

    # Iteración Gauss-Newton
    for it in range(1, max_iter + 1):
        # Construir A (m x npar) y w (misclosure) en metros
        A = np.zeros((m, npar), dtype=float)
        mis = np.zeros((m,), dtype=float)  # d_obs - d_calc

        # Para diagnóstico: d_calc
        d_calc = np.zeros((m,), dtype=float)

        for k in range(m):
            i = idx.get(p_i[k])
            j = idx.get(p_j[k])
            if i is None or j is None:
                raise KeyError(f"Punto no encontrado en índice: {p_i[k]} o {p_j[k]}")

            pi = coords[i].copy()
            pj = coords[j].copy()

            if model == "2d":
                pi[2] = 0.0
                pj[2] = 0.0

            d, g = _compute_dist_and_partials(pi, pj)
            if not np.isfinite(d):
                raise ValueError(f"Distancia inválida para obs {k} ({p_i[k]}-{p_j[k]})")

            d_calc[k] = d
            mis[k] = d_obs[k] - d  # (observado - calculado)

            # Derivadas: ∂d/∂pi = g ; ∂d/∂pj = -g
            # En ecuación w ≈ A dx + v, con w = d_obs - d_calc
            # A lleva -∂d/∂x (porque d_calc aumenta con x). Pero con mis = obs - calc,
            # la linealización queda: mis ≈ -J dx + v, donde J = ∂d/∂x.
            # Entonces A = -J.
            gi = -g
            gj = +g

            # Bloques (X,Y,Z) para punto i
            A[k, 3*i:3*i+3] = gi
            # Bloques para punto j
            A[k, 3*j:3*j+3] = gj

        # N y u
        # N = A^T P A  (P diagonal)
        # u = A^T P mis
        AP = A * P[:, None]
        N = A.T @ AP
        u = A.T @ (P * mis)

        # Restricciones internas
        B = _inner_constraints_B(coords)  # (npar x 6)

        # Resolver sistema aumentado:
        # [ N  B ] [dx] = [u]
        # [B^T 0 ] [k ]   [0]
        Z6 = np.zeros((6, 6), dtype=float)
        K = np.block([[N, B],
                      [B.T, Z6]])
        rhs = np.concatenate([u, np.zeros(6, dtype=float)])

        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            # fallback: pseudo-inversa si algo va mal
            sol = np.linalg.pinv(K) @ rhs

        dx = sol[:npar].reshape(n, 3)
        coords_new = coords + dx

        max_corr = float(np.max(np.linalg.norm(dx, axis=1)))

        if verbose:
            print(f"[ADJUST] iter {it:02d} | max|dX|={max_corr:.6e} m | mean|mis|={float(np.mean(np.abs(mis))):.6e} m")

        coords = coords_new

        if max_corr < tol:
            if verbose:
                print(f"[ADJUST] Convergió en {it} iteraciones.")
            break

    # Residuales finales
    # Recalcular con coords finales
    A = np.zeros((m, npar), dtype=float)
    mis = np.zeros((m,), dtype=float)
    d_calc = np.zeros((m,), dtype=float)

    for k in range(m):
        i = idx[p_i[k]]
        j = idx[p_j[k]]
        pi = coords[i].copy()
        pj = coords[j].copy()
        if model == "2d":
            pi[2] = 0.0
            pj[2] = 0.0
        d, g = _compute_dist_and_partials(pi, pj)
        d_calc[k] = d
        mis[k] = d_obs[k] - d
        gi = -g
        gj = +g
        A[k, 3*i:3*i+3] = gi
        A[k, 3*j:3*j+3] = gj

    AP = A * P[:, None]
    N = A.T @ AP
    u = A.T @ (P * mis)
    B = _inner_constraints_B(coords)
    Z6 = np.zeros((6, 6), dtype=float)
    K = np.block([[N, B],
                  [B.T, Z6]])
    rhs = np.concatenate([u, np.zeros(6, dtype=float)])
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(K) @ rhs
    dx = sol[:npar]

    # residuales: v = A*dx - mis  (porque mis ≈ A dx + v)
    v = (A @ dx) - mis

    # Estadísticos
    # grados de libertad: m - (npar - 6) (porque 6 restricciones internas)
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
        "sigma0_hat": sigma0_hat,     # en metros (porque v en m)
        "sigma0_hat_mm": sigma0_hat * 1000.0,
        "mean_abs_v_mm": float(np.mean(np.abs(v)) * 1000.0),
        "max_abs_v_mm": float(np.max(np.abs(v)) * 1000.0),
    }

    # Salidas DataFrame
    points_out = pd.DataFrame(coords, columns=["X_adj", "Y_adj", "Z_adj"])
    points_out.insert(0, "name", names)

    obs_out = df.copy()
    obs_out["dist_calc_adj"] = d_calc
    obs_out["misclosure_m"] = mis
    obs_out["v_m"] = v
    obs_out["v_mm"] = v * 1000.0
    obs_out["w"] = P  # peso (1/m^2)

    if verbose:
        print("\n[ADJUST] Resumen:")
        print(f"  dof              : {stats['dof']}")
        print(f"  sigma0_hat        : {stats['sigma0_hat_mm']:.3f} mm")
        print(f"  mean |v|          : {stats['mean_abs_v_mm']:.3f} mm")
        print(f"  max  |v|          : {stats['max_abs_v_mm']:.3f} mm")

    return AdjustmentResult(points=points_out, observations=obs_out, stats=stats)
