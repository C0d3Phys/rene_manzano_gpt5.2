import numpy as np

def inner_constraints_B(coords: np.ndarray) -> np.ndarray:
    """
    B: (npar x 6) restricciones internas para red libre.
    params: [dX1 dY1 dZ1 dX2 dY2 dZ2 ...]
    """
    n = coords.shape[0]
    npar = 3 * n
    B = np.zeros((npar, 6), dtype=float)

    for i in range(n):
        X, Y, Z = coords[i]
        ix = 3*i
        iy = ix + 1
        iz = ix + 2

        # traslación
        B[ix, 0] = 1.0
        B[iy, 1] = 1.0
        B[iz, 2] = 1.0

        # rotaciones
        B[iy, 3] = -Z; B[iz, 3] =  Y  # Y*dZ - Z*dY
        B[ix, 4] =  Z; B[iz, 4] = -X  # Z*dX - X*dZ
        B[ix, 5] = -Y; B[iy, 5] =  X  # X*dY - Y*dX

    return B
