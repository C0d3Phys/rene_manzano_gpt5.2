# src/tools/angles.py
import math
from typing import Tuple


# -------------------------------
# Conversión grados <-> radianes
# -------------------------------

def deg_to_rad(deg: float) -> float:
    """Convierte grados decimales a radianes."""
    return math.radians(deg)

def rad_to_deg(rad: float) -> float:
    """Convierte radianes a grados decimales."""
    return math.degrees(rad)

# Alias para compatibilidad con tu código actual
def grados2rad(angulo: float) -> float:
    """Alias de deg_to_rad para compatibilidad."""
    return deg_to_rad(angulo)


# -------------------------------
# DMS/GMS <-> grados decimales
# -------------------------------

def dms_to_deg(deg: float, minute: float = 0.0, second: float = 0.0) -> float:
    """
    Convierte DMS (Degrees, Minutes, Seconds) a grados decimales.
    Maneja correctamente signos (Sur/Oeste negativos).

    Reglas:
      - El signo se toma únicamente del parámetro 'deg'.
      - minute y second se usan en valor absoluto.

    Ejemplos:
        dms_to_deg(40, 25, 46.8)   ->  40.429666...
        dms_to_deg(-33, 26, 45.9)  -> -33.446083...
    """
    frac = abs(minute) / 60.0 + abs(second) / 3600.0
    return deg + frac if deg >= 0 else deg - frac

# Alias con tu nombre original
def gms2dec(grados: float, minutos: float = 0.0, segundos: float = 0.0) -> float:
    """Alias de dms_to_deg para compatibilidad."""
    return dms_to_deg(grados, minutos, segundos)


def deg_to_dms(deg: float) -> Tuple[int, int, float]:
    """
    Convierte grados decimales a (D, M, S).
    El signo se mantiene en D (grados).
    """
    sign = -1 if deg < 0 else 1
    x = abs(deg)
    d = int(x)
    m_float = (x - d) * 60.0
    m = int(m_float)
    s = (m_float - m) * 60.0
    return sign * d, m, s


# -------------------------------
# Normalización / wrapping
# -------------------------------

def wrap_pi(rad: float) -> float:
    """Envuelve un ángulo en radianes a [-pi, pi)."""
    return (rad + math.pi) % (2.0 * math.pi) - math.pi

def wrap_2pi(rad: float) -> float:
    """Envuelve un ángulo en radianes a [0, 2pi)."""
    return rad % (2.0 * math.pi)

def wrap_deg(deg: float) -> float:
    """Envuelve un ángulo en grados a [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0

def wrap_lon_deg(lon_deg: float) -> float:
    """
    Normaliza longitud en grados a [-180, 180).
    (Alias semántico de wrap_deg para longitudes.)
    """
    return wrap_deg(lon_deg)
