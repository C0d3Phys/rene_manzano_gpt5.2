from __future__ import annotations
import numpy as np


def dist_and_grad(pi: np.ndarray, pj: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Calcula:
      - d: distancia euclídea entre dos puntos 3D pi y pj
      - g: gradiente unitario (vector dirección) usado para armar el Jacobiano

    Entradas
    --------
    pi, pj : np.ndarray shape (3,)
        Coordenadas [X, Y, Z] de los puntos i y j.

    Salidas
    -------
    d : float
        Distancia ||pi - pj||.
    g : np.ndarray shape (3,)
        Vector unitario en la dirección (pi - pj) / ||pi - pj||.

    Nota
    ----
    Si d = 0 o d no es finito, devuelve NaN para indicar que la derivada no es válida
    (división por cero / geometría degenerada).
    """
    # Vector diferencia (pi - pj)
    dvec = pi - pj

    # d = sqrt(dvec · dvec) = norma Euclídea
    d = float(np.sqrt(np.dot(dvec, dvec)))

    # Caso degenerado: puntos coincidentes o valor numérico inválido
    if d == 0.0 or not np.isfinite(d):
        return np.nan, np.array([np.nan, np.nan, np.nan], dtype=float)

    # Gradiente unitario: ∂d/∂pi = (pi - pj) / d
    # (y ∂d/∂pj = -(pi - pj)/d )
    return d, dvec / d


def assemble_A_mis(
    coords: np.ndarray,
    idx: dict[str, int],
    pi_name: np.ndarray,
    pj_name: np.ndarray,
    d_obs: np.ndarray,
    model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Arma los elementos básicos para un ajuste por mínimos cuadrados con observaciones de DISTANCIA:

      - A    : matriz de diseño (m x (3n))   [en el código se arma como A = -J]
      - mis  : vector de desajustes (misclosure) (m,)
      - d_calc: distancias calculadas con coords actuales (m,)

    Donde:
      m = número de observaciones (len(d_obs))
      n = número de puntos (coords.shape[0])
      3n = número de parámetros si cada punto tiene X,Y,Z

    Modelo de observación (no lineal)
    --------------------------------
      d_obs ≈ d_calc(coords) + v

    Linealización (Gauss-Newton)
    ----------------------------
      d_obs - d_calc(x0) ≈ J * dx + v

    Aquí:
      mis = d_obs - d_calc(x0)

    y se arma A = -J para que la ecuación quede en la forma típica:
      A * dx ≈ mis
    (equivalente, solo cambia el signo según la convención del solver)

    Entradas
    --------
    coords : np.ndarray shape (n,3)
        Coordenadas actuales (iteración actual) de todos los puntos.
    idx : dict[str,int]
        Mapa nombre->índice para ubicar rápidamente coords[i].
    pi_name, pj_name : np.ndarray shape (m,)
        Nombres de puntos i y j para cada observación k.
    d_obs : np.ndarray shape (m,)
        Distancias observadas (metros).
    model : str
        Si "2d": fuerza Z=0 para que la distancia sea planimétrica.
        Si no: usa 3D.

    Salidas
    -------
    A : np.ndarray shape (m, 3n)
        Matriz de diseño (por bloques de 3 columnas por punto).
    mis : np.ndarray shape (m,)
        Vector misclosure = d_obs - d_calc.
    d_calc : np.ndarray shape (m,)
        Distancias calculadas con coords actuales.
    """
    # n = número de puntos
    n = coords.shape[0]

    # npar = número de parámetros del ajuste: (X,Y,Z) por punto
    npar = 3 * n

    # m = número de observaciones
    m = len(d_obs)

    # Matriz de diseño A (m x 3n)
    A = np.zeros((m, npar), dtype=float)

    # Vector de misclosures: mis[k] = d_obs[k] - d_calc[k]
    mis = np.zeros((m,), dtype=float)

    # Vector de distancias calculadas
    d_calc = np.zeros((m,), dtype=float)

    # Recorre cada observación k: distancia entre pi_name[k] y pj_name[k]
    for k in range(m):
        # Índices de los puntos i y j en la matriz coords
        i = idx[pi_name[k]]
        j = idx[pj_name[k]]

        # Copias locales (evita modificar coords original)
        pi = coords[i].copy()
        pj = coords[j].copy()

        # Si el modelo es 2D, anula Z para que no influya en la distancia
        if model == "2d":
            pi[2] = 0.0
            pj[2] = 0.0

        # Distancia calculada y gradiente unitario
        d, g = dist_and_grad(pi, pj)

        # Si la geometría es degenerada (d=0 o NaN/inf), aborta con error explícito
        if not np.isfinite(d):
            raise ValueError(f"Distancia inválida en obs {k} ({pi_name[k]}-{pj_name[k]})")

        # Guarda distancia calculada
        d_calc[k] = d

        # Misclosure: observado - calculado
        mis[k] = d_obs[k] - d

        # ------------------------------------------------------------
        # Construcción del Jacobiano (derivadas parciales)
        # ------------------------------------------------------------
        # Para d = ||pi - pj||:
        #   ∂d/∂pi =  (pi - pj)/d = g
        #   ∂d/∂pj = -(pi - pj)/d = -g
        #
        # Si la ecuación linealizada es:
        #   mis ≈ J * dx
        #
        # Aquí se arma A = -J (convención del solver):
        #   A = -J  =>  A*dx ≈ mis
        #
        # Entonces:
        #   bloque(i) = -∂d/∂pi = -g
        #   bloque(j) = -∂d/∂pj = +g
        #
        # Cada punto ocupa 3 columnas: [X_i, Y_i, Z_i]
        A[k, 3 * i : 3 * i + 3] = -g
        A[k, 3 * j : 3 * j + 3] =  g

    return A, mis, d_calc
