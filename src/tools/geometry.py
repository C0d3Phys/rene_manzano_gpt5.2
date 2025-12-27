# src/tools/geometry.py
import numpy as np

def distance_2d_np(x1, y1, x2, y2):
    """
    Distancia Euclídea 2D (vectorizada).
    Acepta escalares, arrays o pandas Series.
    """
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    dx = x2 - x1
    dy = y2 - y1
    return np.sqrt(dx*dx + dy*dy)


def distance_3d_np(x1, y1, z1, x2, y2, z2):
    """
    Distancia Euclídea 3D (vectorizada).
    Acepta escalares, arrays o pandas Series.
    """
    x1 = np.asarray(x1, dtype=float)
    y1 = np.asarray(y1, dtype=float)
    z1 = np.asarray(z1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    y2 = np.asarray(y2, dtype=float)
    z2 = np.asarray(z2, dtype=float)

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    return np.sqrt(dx*dx + dy*dy + dz*dz)
