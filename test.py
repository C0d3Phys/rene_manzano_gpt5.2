"""
Ajuste por ecuaciones de condición (condicional / combinado) para distancias 3D
============================================================================

Este script trabaja DIRECTO con tu np.array `npdata` (dtype=object) con filas tipo:

  [id_i, id_j, Xi(3,), Xj(3,), d_calc, d_obs, (d_calc - d_obs)]

Objetivo
--------
Construir AUTOMÁTICAMENTE:
  1) W : vector de misclosures (d_calc - d_obs)  (m x 1)
  2) B : matriz de coeficientes de condición     (m x (3*npts + m))
         - primeras 3*npts columnas: correcciones de coordenadas ΔX,ΔY,ΔZ por punto
         - últimas m columnas: residuos de distancias v_dk (uno por observación)
           (coeficiente -1 en la fila k)
  3) C : matriz diagonal (covarianza) de tamaño (3*npts + m)
         - varianzas para coords (por punto) y para distancias (por observación)
  4) Resolver:
         K = (B C B^T)^-1 W
         V = - C B^T K
     donde V contiene [Δcoords ; v_dist]

Notas
-----
- Este enfoque permite "combinado": coords con incertidumbre + distancias con incertidumbre.
- Si NO quieres ajustar coords (solo ajustar distancias), entonces este enfoque no aplica tal cual,
  porque aquí las coords forman parte del vector de incógnitas/correcciones.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


# =========================
# Estructuras de resultados
# =========================
@dataclass
class BuildResult:
    npdata: np.ndarray
    order: List[int]
    point_ids: List[int]
    point_col: Dict[int, int]
    m: int
    npts: int
    n: int
    B: np.ndarray
    W: np.ndarray
    C: np.ndarray
    coords0: Dict[int, np.ndarray]  # coordenadas iniciales por punto


@dataclass
class SolveResult:
    K: np.ndarray
    V: np.ndarray
    dXYZ: Dict[int, np.ndarray]     # correcciones por punto
    v_dist: np.ndarray              # residuos de distancias (m x 1)
    coords_adj: Dict[int, np.ndarray]
    closures: List[dict]            # cierres antes/después


# =========================
# Helpers
# =========================
def _vec3(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(3)


def extract_point_ids(npdata: np.ndarray) -> List[int]:
    ids_i = [int(r[0]) for r in npdata]
    ids_j = [int(r[1]) for r in npdata]
    return sorted(set(ids_i) | set(ids_j))


def build_point_index(point_ids: List[int]) -> Dict[int, int]:
    return {pid: 3 * k for k, pid in enumerate(point_ids)}


def build_coords_dict(npdata: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Toma la primera aparición de cada punto en npdata para fijar sus coords iniciales.
    """
    coords = {}
    for r in npdata:
        i = int(r[0]); j = int(r[1])
        Xi = _vec3(r[2]); Xj = _vec3(r[3])
        coords.setdefault(i, Xi)
        coords.setdefault(j, Xj)
    return coords


# =========================
# Construcción B, W, C (modo profesor)
# =========================
def build_W_prof(npdata: np.ndarray) -> np.ndarray:
    """
    Profesor: W = col6 = (d_calc - d_obs)
    """
    m = npdata.shape[0]
    W = np.zeros((m, 1), float)
    for k in range(m):
        W[k, 0] = float(npdata[k, 6])
    return W


def build_B_prof(npdata: np.ndarray, point_col: Dict[int, int]) -> np.ndarray:
    """
    Profesor: u = (Xj - Xi) / d_calc  donde d_calc = col4
    Condición lineal:
        u^T ΔXj - u^T ΔXi - v_k + (d_calc - d_obs) = 0
    => fila k:
        cols(i) = -u
        cols(j) = +u
        col(residuo k) = -1
    """
    m = npdata.shape[0]
    npts = len(point_col)
    n = 3 * npts + m
    B = np.zeros((m, n), float)

    for k in range(m):
        i = int(npdata[k, 0])
        j = int(npdata[k, 1])
        Xi = _vec3(npdata[k, 2])
        Xj = _vec3(npdata[k, 3])
        d_calc = float(npdata[k, 4])  # <- EXACTO como el profe

        u = (Xj - Xi) / d_calc

        ci = point_col[i]
        cj = point_col[j]

        B[k, ci:ci+3] = -u
        B[k, cj:cj+3] = +u

        B[k, 3*npts + k] = -1.0  # residuo de distancia k

    return B


def build_C_prof(npts: int, m: int,
                 var_coord: float = 0.0005,
                 var_dist: float = 0.000004) -> np.ndarray:
    """
    Profesor: C diagonal (24x24 en tu caso):
      - 18 veces 0.0005 (coords)
      - 6  veces 0.000004 (distancias)
    """
    diag = [var_coord] * (3 * npts) + [var_dist] * m
    return np.diag(diag).astype(float)


# =========================
# Solución condicional (consistente con C=covarianza)
# =========================
def solve_conditional(B: np.ndarray, W: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    K = (B C B^T)^-1 W
    V = - C B^T K
    """
    SigmaK = np.linalg.inv(B @ C @ B.T)
    K = SigmaK @ W
    V = -(C @ (B.T @ K))
    return K, V


# =========================
# Post-proceso: coords ajustadas + cierres nuevos
# =========================
def compute_adjusted_coords(point_ids: List[int],
                            coords0: Dict[int, np.ndarray],
                            V: np.ndarray,
                            m: int) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    npts = len(point_ids)
    Vcoords = V[:3*npts].reshape(npts, 3)
    v_dist = V[3*npts:]  # (m x 1)

    dXYZ = {}
    coords_adj = {}
    for pid, d in zip(point_ids, Vcoords):
        dXYZ[pid] = d
        coords_adj[pid] = coords0[pid] + d

    return dXYZ, coords_adj, v_dist


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_closures(npdata: np.ndarray,
                     coords_adj: Dict[int, np.ndarray],
                     v_dist: np.ndarray) -> List[dict]:
    """
    Para cada observación:
      - d_new: distancia con coords ajustadas
      - cierre_raw: d_new - d_obs
      - cierre_corr: d_new - (d_obs + v)   (debe quedar ~0)
    """
    out = []
    for k, r in enumerate(npdata):
        i = int(r[0]); j = int(r[1])
        d_obs = float(r[5])
        d_new = dist(coords_adj[i], coords_adj[j])
        v = float(v_dist[k, 0])

        out.append({
            "i": i, "j": j,
            "d_obs": d_obs,
            "d_new": d_new,
            "cierre_raw_m": d_new - d_obs,
            "v_m": v,
            "cierre_corr_m": d_new - (d_obs + v),
        })
    return out


# =========================
# Pipeline “modo profesor”
# =========================
def run_adjustment_professor(
    data: np.ndarray,
    order: List[int] = None,
    var_coord: float = 0.0005,
    var_dist: float = 0.000004,
    verbose: bool = True,
) -> Tuple[BuildResult, SolveResult]:

    if order is None:
        order = list(range(data.shape[0]))

    npdata = data[order]

    point_ids = extract_point_ids(npdata)
    point_col = build_point_index(point_ids)
    coords0 = build_coords_dict(npdata)

    m = npdata.shape[0]
    npts = len(point_ids)
    n = 3*npts + m

    W = build_W_prof(npdata)
    B = build_B_prof(npdata, point_col)
    C = build_C_prof(npts, m, var_coord=var_coord, var_dist=var_dist)

    K, V = solve_conditional(B, W, C)
    dXYZ, coords_adj, v_dist = compute_adjusted_coords(point_ids, coords0, V, m)
    closures = compute_closures(npdata, coords_adj, v_dist)

    built = BuildResult(
        npdata=npdata, order=order,
        point_ids=point_ids, point_col=point_col,
        m=m, npts=npts, n=n, B=B, W=W, C=C,
        coords0=coords0
    )
    sol = SolveResult(
        K=K, V=V, dXYZ=dXYZ, v_dist=v_dist,
        coords_adj=coords_adj, closures=closures
    )

    if verbose:
        np.set_printoptions(precision=6, suppress=True)
        print("=== MODO PROFESOR (automatizado) ===")
        print("Orden filas usado:", order)
        print("Puntos (orden interno):", point_ids)
        print("B shape:", B.shape, " W shape:", W.shape, " C shape:", C.shape)

        print("\nW = d_calc - d_obs (col6):")
        print(W)

        print("\nK (multiplicadores):")
        print(K)

        print("\nCorrecciones ΔX,ΔY,ΔZ por punto (m):")
        for pid in point_ids:
            d = dXYZ[pid]
            print(f"Pt {pid:>3}: dX={d[0]: .6f} dY={d[1]: .6f} dZ={d[2]: .6f}")

        print("\nCierres nuevos (mm): cierre_raw = d_new-d_obs, cierre_corr = d_new-(d_obs+v)")
        for c in closures:
            print(f"{c['i']:>3}-{c['j']:<3}  raw={c['cierre_raw_m']*1000: .6f}  "
                  f"v={c['v_m']*1000: .6f}  corr={c['cierre_corr_m']*1000: .6f}")

    return built, sol


# =========================
# Ejemplo con tus datos
# =========================
if __name__ == "__main__":
    data = np.array([
        [10.0, 11.0,
         np.array([-922886.72715334, -5951569.11657866,  2099281.85726845]),
         np.array([-923608.04171829, -5951534.04024824,  2099036.70315641]),
         762.6438154315333, 762.566, 0.0778154315332813],

        [4.0, 3.0,
         np.array([-924151.68281468, -5952372.84075829,  2096541.10163531]),
         np.array([-924347.7783152 , -5952308.43195145,  2096629.12382028]),
         224.38771082294144, 224.384, 0.0037108229414570815],

        [11.0, 4.0,
         np.array([-923608.04171829, -5951534.04024824,  2099036.70315641]),
         np.array([-924151.68281468, -5952372.84075829,  2096541.10163531]),
         2688.3375698433133, 2688.1995, 0.13806984331313288],

        [8.0, 4.0,
         np.array([-921992.37525752, -5952223.15849636,  2097964.20772335]),
         np.array([-924151.68281468, -5952372.84075829,  2096541.10163531]),
         2590.4140294107874, 2590.4349, -0.020870589212790946],

        [8.0, 9.0,
         np.array([-921992.37525752, -5952223.15849636,  2097964.20772335]),
         np.array([-921980.7985837 , -5952077.42737261,  2098347.62710959]),
         410.34376509154475, 410.188, 0.15576509154476526],

        [8.0, 11.0,
         np.array([-921992.37525752, -5952223.15849636,  2097964.20772335]),
         np.array([-923608.04171829, -5951534.04024824,  2099036.70315641]),
         2058.035112979308, 2057.4457, 0.5894129793077809],
    ], dtype=object)

    # Orden EXACTO del profesor
    order_prof = [1, 4, 0, 3, 2, 5]

    built, sol = run_adjustment_professor(
        data=data,
        order=order_prof,
        var_coord=0.0005,
        var_dist=0.000004,
        verbose=True
    )
