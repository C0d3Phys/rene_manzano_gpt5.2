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


# ------------------------------------------------------------
# 1) Estructuras de salida para tener todo ordenado
# ------------------------------------------------------------
@dataclass
class BuildResult:
    point_ids: List[int]                 # ids únicos de puntos (orden interno)
    point_col: Dict[int, int]            # id -> columna base en el bloque 3D
    m: int                               # número de observaciones
    npts: int                            # número de puntos
    n: int                               # tamaño total del vector V = 3*npts + m
    B: np.ndarray                        # (m x n)
    W: np.ndarray                        # (m x 1)
    C: np.ndarray                        # (n x n)


@dataclass
class SolveResult:
    K: np.ndarray                        # (m x 1) multiplicadores
    V: np.ndarray                        # (n x 1) [Δcoords ; v_dist]
    V_coords: np.ndarray                 # (npts x 3) correcciones por punto
    v_dist: np.ndarray                   # (m x 1) residuos de distancias
    SigmaK: np.ndarray                   # (m x m) covarianza de K = (B C B^T)^-1


# ------------------------------------------------------------
# 2) Helpers directos sobre tu npdata
# ------------------------------------------------------------
def _as_vec3(x) -> np.ndarray:
    """Convierte a vector (3,) float."""
    return np.asarray(x, dtype=float).reshape(3)


def compute_dcalc_and_u(Xi: np.ndarray, Xj: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calcula d_calc = ||Xj - Xi|| y u = (Xj - Xi) / d_calc.
    """
    dvec = Xj - Xi
    d = float(np.linalg.norm(dvec))
    if d <= 0.0 or not np.isfinite(d):
        raise ValueError(f"Distancia inválida (d={d}). Revisa coordenadas.")
    u = dvec / d
    if not np.all(np.isfinite(u)):
        raise ValueError("Vector unitario u inválido.")
    return d, u


def extract_point_ids(npdata: np.ndarray) -> List[int]:
    """
    Extrae ids únicos de puntos desde las columnas 0 y 1.
    """
    ids_i = [int(r[0]) for r in npdata]
    ids_j = [int(r[1]) for r in npdata]
    ids = sorted(set(ids_i) | set(ids_j))
    return ids


def build_point_index(point_ids: List[int]) -> Dict[int, int]:
    """
    Mapea cada punto a su columna base dentro del bloque de coordenadas:
      col_base = 3 * index_del_punto
    """
    return {pid: 3 * k for k, pid in enumerate(point_ids)}


# ------------------------------------------------------------
# 3) Construcción AUTOMÁTICA de W y B
# ------------------------------------------------------------
def build_W(npdata: np.ndarray, mode: str = "use_col6") -> np.ndarray:
    """
    Construye W (m x 1).

    mode:
      - "use_col6": W = npdata[:,6]  (asume que es d_calc - d_obs)
      - "recompute": recalcula d_calc desde Xi,Xj y hace W = d_calc - d_obs

    Recomendación:
      usa "recompute" si quieres validar coherencia y no depender del precálculo.
    """
    m = npdata.shape[0]
    W = np.zeros((m, 1), dtype=float)

    for k in range(m):
        if mode == "use_col6":
            W[k, 0] = float(npdata[k, 6])
        elif mode == "recompute":
            Xi = _as_vec3(npdata[k, 2])
            Xj = _as_vec3(npdata[k, 3])
            d_obs = float(npdata[k, 5])
            d_calc, _u = compute_dcalc_and_u(Xi, Xj)
            W[k, 0] = d_calc - d_obs
        else:
            raise ValueError("mode debe ser 'use_col6' o 'recompute'.")

    return W


def build_B(npdata: np.ndarray, point_col: Dict[int, int]) -> np.ndarray:
    """
    Construye B (m x (3*npts + m)) automáticamente.

    Para cada observación k entre i-j:
      condición linealizada:
        u^T ΔXj - u^T ΔXi - v_k + (d_calc - d_obs) = 0

    => En la fila k:
      - en columnas del punto i:  -u
      - en columnas del punto j:  +u
      - en la columna del residuo v_k (parte final): -1
    """
    m = npdata.shape[0]
    npts = len(point_col)
    n = 3 * npts + m

    B = np.zeros((m, n), dtype=float)

    for k in range(m):
        i = int(npdata[k, 0])
        j = int(npdata[k, 1])
        Xi = _as_vec3(npdata[k, 2])
        Xj = _as_vec3(npdata[k, 3])

        # u se puede obtener con el d_calc real (recomendado)
        d_calc, u = compute_dcalc_and_u(Xi, Xj)

        ci = point_col[i]       # columna base de i
        cj = point_col[j]       # columna base de j

        # Bloques 3D
        B[k, ci:ci+3] = -u
        B[k, cj:cj+3] = +u

        # Residuo de la distancia k: v_k
        B[k, 3*npts + k] = -1.0

    return B


# ------------------------------------------------------------
# 4) Construcción de C (covarianza diagonal) y/o pesos
# ------------------------------------------------------------
def build_C(
    npts: int,
    m: int,
    var_coord: float = 0.0005,
    var_dist: float = 0.000004,
    per_component: bool = True
) -> np.ndarray:
    """
    Construye C diagonal de tamaño (3*npts + m).

    Parámetros
    ----------
    var_coord:
      varianza para cada componente de coordenada (X,Y,Z) [m^2]
      (si quieres usar sigmas, aquí debe ir sigma^2)

    var_dist:
      varianza para cada distancia [m^2]

    per_component:
      True -> se asigna var_coord a cada componente X,Y,Z (18 entradas si npts=6)
      (esto es lo usual)

    Devuelve:
      C (n x n), covarianza diagonal
    """
    n = 3 * npts + m
    diag = np.zeros(n, dtype=float)

    # Bloque coordenadas
    if per_component:
        diag[:3*npts] = var_coord
    else:
        # (no recomendado) misma var para el bloque completo, lo dejo por compatibilidad
        diag[:3*npts] = var_coord

    # Bloque distancias
    diag[3*npts:] = var_dist

    return np.diag(diag)


def covariance_to_weight(C: np.ndarray) -> np.ndarray:
    """
    Convierte covarianza C a pesos P = C^{-1}.
    """
    return np.linalg.inv(C)


# ------------------------------------------------------------
# 5) Solución condicional (forma CONSISTENTE con C = covarianza)
# ------------------------------------------------------------
def solve_conditional(B: np.ndarray, W: np.ndarray, C: np.ndarray) -> SolveResult:
    """
    Resuelve:

      K = (B C B^T)^-1 W
      V = - C B^T K

    Devuelve además separaciones útiles:
      V_coords: (npts x 3)
      v_dist  : (m x 1)
    """
    m, n = B.shape
    # SigmaK = (B C B^T)^-1
    SigmaK = np.linalg.inv(B @ C @ B.T)

    K = SigmaK @ W
    V = -(C @ (B.T @ K))

    npts = (n - m) // 3
    V_coords = V[:3*npts].reshape(npts, 3)
    v_dist = V[3*npts:]  # (m x 1)

    return SolveResult(K=K, V=V, V_coords=V_coords, v_dist=v_dist, SigmaK=SigmaK)


# ------------------------------------------------------------
# 6) Pipeline completo: desde npdata -> B,W,C -> solve
# ------------------------------------------------------------
def build_all(
    npdata: np.ndarray,
    w_mode: str = "use_col6",
    var_coord: float = 0.0005,
    var_dist: float = 0.000004,
) -> BuildResult:
    """
    Construye todo en orden.
    """
    point_ids = extract_point_ids(npdata)
    point_col = build_point_index(point_ids)

    m = npdata.shape[0]
    npts = len(point_ids)
    n = 3 * npts + m

    W = build_W(npdata, mode=w_mode)
    B = build_B(npdata, point_col=point_col)
    C = build_C(npts=npts, m=m, var_coord=var_coord, var_dist=var_dist)

    return BuildResult(
        point_ids=point_ids,
        point_col=point_col,
        m=m,
        npts=npts,
        n=n,
        B=B,
        W=W,
        C=C,
    )


def run_adjustment(
    npdata: np.ndarray,
    w_mode: str = "use_col6",
    var_coord: float = 0.0005,
    var_dist: float = 0.000004,
    verbose: bool = True,
) -> Tuple[BuildResult, SolveResult]:
    """
    Ejecuta el proceso completo:
      - build_all
      - solve_conditional
      - reporte opcional
    """
    built = build_all(
        npdata=npdata,
        w_mode=w_mode,
        var_coord=var_coord,
        var_dist=var_dist,
    )

    sol = solve_conditional(B=built.B, W=built.W, C=built.C)

    if verbose:
        np.set_printoptions(precision=6, suppress=True)
        print("==== Dimensiones ====")
        print(f"m (observaciones)         : {built.m}")
        print(f"npts (puntos únicos)      : {built.npts} -> {built.point_ids}")
        print(f"n (3*npts + m)            : {built.n}")
        print("B shape:", built.B.shape)
        print("W shape:", built.W.shape)
        print("C shape:", built.C.shape)

        print("\n==== W (d_calc - d_obs) ====")
        print(built.W)

        print("\n==== K (multiplicadores) ====")
        print(sol.K)

        print("\n==== V_coords (ΔX,ΔY,ΔZ por punto, en el orden point_ids) ====")
        for pid, dxyz in zip(built.point_ids, sol.V_coords):
            print(f"Punto {pid:>3}: dX={dxyz[0]: .6f}  dY={dxyz[1]: .6f}  dZ={dxyz[2]: .6f}")

        print("\n==== v_dist (residuos de distancias, por observación en el orden de npdata) ====")
        print(sol.v_dist)

    return built, sol

# ------------------------------------------------------------
# 7) Ejemplo de uso
# ------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    data = np.array([
        [
            10.0, 11.0,
            np.array([-922886.72715334, -5951569.11657866, 2099281.85726845]),
            np.array([-923608.04171829, -5951534.04024824, 2099036.70315641]),
            762.6438154315333,
            762.566,
            0.0778154315332813
        ],
        [
            4.0, 3.0,
            np.array([-924151.68281468, -5952372.84075829, 2096541.10163531]),
            np.array([-924347.7783152 , -5952308.43195145, 2096629.12382028]),
            224.38771082294144,
            224.384,
            0.0037108229414570815
        ],
        [
            11.0, 4.0,
            np.array([-923608.04171829, -5951534.04024824, 2099036.70315641]),
            np.array([-924151.68281468, -5952372.84075829, 2096541.10163531]),
            2688.3375698433133,
            2688.1995,
            0.13806984331313288
        ],
        [
            8.0, 4.0,
            np.array([-921992.37525752, -5952223.15849636, 2097964.20772335]),
            np.array([-924151.68281468, -5952372.84075829, 2096541.10163531]),
            2590.4140294107874,
            2590.4349,
            -0.020870589212790946
        ],
        [
            8.0, 9.0,
            np.array([-921992.37525752, -5952223.15849636, 2097964.20772335]),
            np.array([-921980.7985837 , -5952077.42737261, 2098347.62710959]),
            410.34376509154475,
            410.188,
            0.15576509154476526
        ],
        [
            8.0, 11.0,
            np.array([-921992.37525752, -5952223.15849636, 2097964.20772335]),
            np.array([-923608.04171829, -5951534.04024824, 2099036.70315641]),
            2058.035112979308,
            2057.4457,
            0.5894129793077809
        ]
    ], dtype=object)

    # Ejecutar ajuste
    built, sol = run_adjustment(
        npdata=data,
        w_mode="recompute",   # recalcula d_calc para validar
        var_coord=0.0005,     # varianza coordenadas
        var_dist=0.000004,    # varianza distancias (~2 mm)
        verbose=True
    )
