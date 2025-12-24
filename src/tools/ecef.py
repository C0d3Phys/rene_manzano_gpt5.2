# src/tools/ecef.py
#from __future__ import annotations
import math
from dataclasses import dataclass

from .ellipsoids import WGS84, Ellipsoid
from .angles import grados2rad, rad2grados  # si no tienes rad2grados, usa math.degrees


def calc_N(phi_degrees: float, ell: Ellipsoid = WGS84) -> float:
    """
    Radio de curvatura en el primer vertical N(φ) para un elipsoide (default WGS84).
    φ en grados.
    """
    phi = math.radians(phi_degrees)
    e2 = ell.e2
    return ell.a / math.sqrt(1.0 - e2 * (math.sin(phi) ** 2))


def geo_to_ecef(lat_deg: float, lon_deg: float, h: float, ell: Ellipsoid = WGS84) -> tuple[float, float, float]:
    """
    Geodésicas -> ECEF (X,Y,Z) [m]
    lat, lon en grados decimales.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)

    a = ell.a
    e2 = ell.e2

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

    X = (N + h) * cos_lat * cos_lon
    Y = (N + h) * cos_lat * sin_lon
    Z = (N * (1.0 - e2) + h) * sin_lat

    return X, Y, Z


def ecef_to_geo(X: float, Y: float, Z: float, ell: Ellipsoid = WGS84) -> tuple[float, float, float]:
    """
    ECEF -> Geodésicas (lat_deg, lon_deg, h) con un método iterativo robusto.

    - lon: atan2(Y, X) (cuadrantes OK)
    - lat/h: iteración sobre φ usando:
        h = p/cosφ - N
        φ = atan2(Z, p*(1 - e2*N/(N+h)))

    Convergencia controlada por tolerancias separadas (ángulo y altura).
    """
    a = ell.a
    b = ell.b
    e2 = ell.e2

    # p = distancia al eje Z
    p = math.hypot(X, Y)

    # Longitud (si p=0, lon indefinida -> 0)
    lon = 0.0 if p == 0.0 else math.atan2(Y, X)

    # Caso exactamente en el eje polar
    if p == 0.0:
        lat = math.pi / 2.0 if Z >= 0.0 else -math.pi / 2.0
        h = abs(Z) - b
        return math.degrees(lat), math.degrees(lon), h

    # Aproximación inicial (Bowring)
    # theta = atan2(Z*a, p*b)
    theta = math.atan2(Z * a, p * b)
    sin_t = math.sin(theta)
    cos_t = math.cos(theta)

    # lat inicial
    lat = math.atan2(Z + (e2 * b) * (sin_t ** 3), p - (e2 * a) * (cos_t ** 3))

    # Iteración
    tol_lat = 1e-13   # rad (≈ 0.02 mm en superficie)
    tol_h   = 1e-6    # m  (1 micra; puedes subir a 1e-4 para 0.1 mm)
    max_iter = 20

    h = 0.0
    for _ in range(max_iter):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

        h_new = p / math.cos(lat) - N
        lat_new = math.atan2(Z, p * (1.0 - e2 * N / (N + h_new)))

        if abs(lat_new - lat) < tol_lat and abs(h_new - h) < tol_h:
            lat = lat_new
            h = h_new
            break

        lat = lat_new
        h = h_new

    return math.degrees(lat), math.degrees(lon), h
