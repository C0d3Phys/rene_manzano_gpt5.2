import numpy as np


def solve_constrained(N: np.ndarray, u: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Resuelve un sistema normal singular usando restricciones internas (red libre).

    Sistema original (singular):
        N · dx = u

    donde:
        N = Aᵀ P A   (singular, rango deficiente)
        u = Aᵀ P mis

    La singularidad aparece porque la red es libre:
        - 3 traslaciones
        - 3 rotaciones
    (6 grados de libertad no observables)

    Se resuelve ampliando el sistema con la matriz de restricciones internas B:

        | N   B | | dx | = | u |
        | Bᵀ  0 | | λ  |   | 0 |

    donde:
        dx = correcciones a los parámetros
        λ  = multiplicadores de Lagrange
        B  = matriz (npar x 6) del espacio nulo

    Entradas
    --------
    N : np.ndarray (npar x npar)
        Matriz normal singular.
    u : np.ndarray (npar,)
        Vector del lado derecho (Aᵀ P mis).
    B : np.ndarray (npar x 6)
        Matriz de restricciones internas (traslaciones + rotaciones).

    Retorna
    -------
    dx : np.ndarray (npar,)
        Correcciones estimadas que cumplen:
            Bᵀ dx = 0
    """

    # ------------------------------------------------------------
    # Matriz cero para el bloque inferior derecho
    # ------------------------------------------------------------
    # Dimensión 6x6 porque hay 6 restricciones internas (3T + 3R)
    Z = np.zeros((6, 6), dtype=float)

    # ------------------------------------------------------------
    # Construcción de la matriz ampliada K
    # ------------------------------------------------------------
    # K = | N   B |
    #     | Bᵀ  0 |
    #
    # Esta matriz es no singular si B describe correctamente
    # el espacio nulo de N.
    K = np.block([
        [N,   B],
        [B.T, Z]
    ])

    # ------------------------------------------------------------
    # Vector del lado derecho ampliado
    # ------------------------------------------------------------
    # | u |
    # | 0 |
    #
    # El bloque inferior fuerza las restricciones internas:
    #   Bᵀ dx = 0
    rhs = np.concatenate([
        u,
        np.zeros(6, dtype=float)
    ])

    # ------------------------------------------------------------
    # Resolución del sistema ampliado
    # ------------------------------------------------------------
    try:
        # Intenta resolver directamente (más rápido y exacto)
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        # Si K está mal condicionada o casi singular,
        # se usa pseudo-inversa (Moore–Penrose)
        sol = np.linalg.pinv(K) @ rhs

    # ------------------------------------------------------------
    # La solución completa es:
    #   sol = [ dx , λ ]
    #
    # Solo nos interesan las correcciones dx
    return sol[:N.shape[0]]
