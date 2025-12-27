# src/tools/geometry.py
import math
import numpy as np
from typing import Tuple

def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Distancia Euclídea 2D."""
    return math.hypot(x2 - x1, y2 - y1)

def distance_3d(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> float:
    """Distancia Euclídea 3D."""
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    return math.sqrt(dx*dx + dy*dy + dz*dz)

# Alias compatibles con tu estilo anterior
def dist2p(x1: float, y1: float, x2: float, y2: float) -> float:
    """Alias: distancia 2D entre dos puntos."""
    return distance_2d(x1, y1, x2, y2)

def distance_3d_np(x1, y1, z1, x2, y2, z2):
    """
    Vectorizado: distancia Euclídea 3D
    Acepta escalares/arrays/Series y devuelve np.ndarray.
    """
    x1 = np.asarray(x1, dtype=float); y1 = np.asarray(y1, dtype=float); z1 = np.asarray(z1, dtype=float)
    x2 = np.asarray(x2, dtype=float); y2 = np.asarray(y2, dtype=float); z2 = np.asarray(z2, dtype=float)

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    return np.sqrt(dx*dx + dy*dy + dz*dz)

