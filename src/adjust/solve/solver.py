from __future__ import annotations
import pandas as pd

# Modelos / tipos y utilidades de preparación
from src.adjust.prepare.model import AdjustmentResult, build_index

# Bucle iterativo Gauss–Newton (linealiza, arma normales, aplica restricciones internas, actualiza coords)
from src.adjust.solve.iteration import gauss_newton_loop

# Postproceso estadístico: residuos, sigma0_hat, dof, tablas de salida, etc.
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
    """
    Ajuste por mínimos cuadrados de una red de distancias (trilateración) en modo RED LIBRE
    usando restricciones internas (free network / inner constraints).

    Este wrapper hace 3 cosas:
      1) valida y prepara índices (name -> fila en coords)
      2) ejecuta el loop Gauss–Newton hasta converger
      3) construye el resultado final (tablas + estadísticas)

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original de observaciones (se usa para reportes y para reconstruir tablas).
        Debe contener al menos lo que tu pipeline ya usa (id1/id2, distancia, etc.).
    names : list[str]
        Lista de puntos únicos (orden de referencia para coords0 y el vector de parámetros).
    coords0 : array-like (n,3)
        Coordenadas iniciales (aproximación) para comenzar la iteración.
    pi_name, pj_name : array-like (m,)
        Nombres (strings) del punto i y j por cada observación de distancia.
    d_obs : array-like (m,)
        Distancias observadas (metros).
    P : array-like
        Matriz de pesos P (típicamente diagonal) o representación equivalente según tu solver.
        (En LS: N = Aᵀ P A, u = Aᵀ P mis)
    model : str (keyword-only)
        "3d" o "2d". En "2d" se ignora Z (distancia planimétrica).
    max_iter : int (keyword-only)
        Máximo número de iteraciones Gauss–Newton.
    tol : float (keyword-only)
        Tolerancia de convergencia (típicamente norma de dx, cambio relativo, etc.).
    verbose : bool (keyword-only)
        Si True, imprime un resumen del proceso y estadísticas finales.

    Retorna
    -------
    AdjustmentResult
        - points: DataFrame con coordenadas ajustadas y (posiblemente) sigmas/correcciones
        - observations: DataFrame con distancias calculadas, residuos, etc.
        - stats: dict con indicadores globales (dof, sigma0_hat, etc.)
    """

    # ------------------------------------------------------------
    # Normalización / validación del modo (2D o 3D)
    # ------------------------------------------------------------
    model = model.lower().strip()
    if model not in ("3d", "2d"):
        raise ValueError("model debe ser '3d' o '2d'")

    # ------------------------------------------------------------
    # Índice name -> fila en coords
    # ------------------------------------------------------------
    # Esto permite acceder a coords[i] rápido al armar A y mis:
    #   i = idx[pi_name[k]]
    idx = build_index(names)

    # ------------------------------------------------------------
    # Mensajes informativos (opcional)
    # ------------------------------------------------------------
    if verbose:
        n = len(names)     # número de puntos
        m = len(d_obs)     # número de observaciones
        print("\n[ADJUST] LS red libre (restricciones internas)")
        print(f"[ADJUST] Modelo {model.upper()} | Obs={m} | Puntos={n} | Params={3*n}")

    # ------------------------------------------------------------
    # Iteración Gauss–Newton (núcleo del ajuste no lineal)
    # ------------------------------------------------------------
    # Internamente, típicamente hace:
    #   - Armar A, mis, d_calc con coords actuales
    #   - Formar N = Aᵀ P A, u = Aᵀ P mis
    #   - Aplicar restricciones internas (p.ej. KKT con B) para resolver dx
    #   - Actualizar coords <- coords + reshape(dx)
    #   - Chequear convergencia (tol) y repetir hasta max_iter
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

    # ------------------------------------------------------------
    # Postproceso: tablas + estadísticas a posteriori
    # ------------------------------------------------------------
    # Aquí normalmente se calcula:
    #   - distancias calculadas finales
    #   - residuos v = d_obs - d_calc (o según convención)
    #   - sigma0_hat (a posteriori)
    #   - dof (grados de libertad)
    #   - métricas como mean|v|, max|v|, etc.
    #   - DataFrame de puntos y de observaciones con columnas útiles
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

    # ------------------------------------------------------------
    # Resumen final (opcional)
    # ------------------------------------------------------------
    if verbose:
        s = res.stats
        print("\n[ADJUST] Resumen:")
        print(f"  dof       : {s['dof']}")
        print(f"  sigma0^   : {s['sigma0_hat_mm']:.3f} mm")
        print(f"  mean|v|   : {s['mean_abs_v_mm']:.3f} mm")
        print(f"  max |v|   : {s['max_abs_v_mm']:.3f} mm")

    return res
