import numpy as np


def inner_constraints_B(coords: np.ndarray) -> np.ndarray:
    """
    Construye la matriz B de restricciones internas para una red libre 3D.

    En un ajuste libre (free network), el sistema normal es singular porque
    la red puede:
      - trasladarse en X, Y, Z
      - rotar alrededor de X, Y, Z

    La matriz B define el subespacio nulo (null space) de esos 6 movimientos rígidos.

    Parámetros
    ----------
    coords : np.ndarray shape (n,3)
        Coordenadas actuales (aproximadas) de los puntos de la red.

    Retorna
    -------
    B : np.ndarray shape (npar, 6)
        Matriz de restricciones internas, donde:
          npar = 3*n
          6 = 3 traslaciones + 3 rotaciones

    El vector de parámetros está ordenado como:
        [dX1, dY1, dZ1, dX2, dY2, dZ2, ..., dXn, dYn, dZn]
    """
    # Número de puntos
    n = coords.shape[0]

    # Número total de parámetros (3 por punto)
    npar = 3 * n

    # Matriz B: (npar x 6)
    # Cada columna representa un movimiento rígido independiente
    B = np.zeros((npar, 6), dtype=float)

    # Recorre cada punto de la red
    for i in range(n):
        # Coordenadas del punto i
        X, Y, Z = coords[i]

        # Índices del bloque (dXi, dYi, dZi) dentro del vector de parámetros
        ix = 3 * i
        iy = ix + 1
        iz = ix + 2

        # ------------------------------------------------------------
        # 1–3) TRASLACIONES
        # ------------------------------------------------------------
        # Traslación rígida de toda la red:
        #   dX = cte, dY = cte, dZ = cte
        #
        # Cada punto se mueve igual en cada eje

        # Traslación en X
        B[ix, 0] = 1.0

        # Traslación en Y
        B[iy, 1] = 1.0

        # Traslación en Z
        B[iz, 2] = 1.0

        # ------------------------------------------------------------
        # 4–6) ROTACIONES
        # ------------------------------------------------------------
        # Rotaciones infinitesimales (ángulos pequeños) alrededor de
        # los ejes X, Y, Z que NO cambian las distancias internas.
        #
        # Para un vector r = (X, Y, Z):
        #   d r = ω × r
        #
        # con ω = (ωx, ωy, ωz)

        # Rotación alrededor del eje X (ωx):
        #   dY = -ωx * Z
        #   dZ =  ωx * Y
        B[iy, 3] = -Z
        B[iz, 3] =  Y

        # Rotación alrededor del eje Y (ωy):
        #   dX =  ωy * Z
        #   dZ = -ωy * X
        B[ix, 4] =  Z
        B[iz, 4] = -X

        # Rotación alrededor del eje Z (ωz):
        #   dX = -ωz * Y
        #   dY =  ωz * X
        B[ix, 5] = -Y
        B[iy, 5] =  X

    return B
