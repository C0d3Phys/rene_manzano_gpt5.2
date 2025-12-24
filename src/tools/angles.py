import math

def grados2rad(angulo: float) -> float:
    """
    Convierte un ángulo de grados decimales a radianes.
    
    Esta función es esencial en cálculos geodésicos porque las funciones
    trigonométricas de Python (math.sin, math.cos, math.tan, etc.) esperan
    argumentos en radianes, mientras que las latitudes y longitudes suelen
    manejarse en grados decimales (ej. 40.4168° para Madrid).
    
    Fórmula:
        radianes = grados × π / 180
    
    Parámetros:
        angulo (float): Ángulo en grados decimales.
                        Rango típico para latitud: -90.0 a +90.0
                        Rango típico para longitud: -180.0 a +180.0
    
    Retorna:
        float: El ángulo equivalente en radianes.
    
    Ejemplos:
        >>> grados2rad(0)
        0.0
        >>> grados2rad(90)
        1.5707963267948966  # ≈ π/2
        >>> grados2rad(45)
        0.7853981633974483  # ≈ π/4
        >>> grados2rad(180)
        3.141592653589793   # ≈ π
    
    Nota:
        Esta función es simple pero crítica para evitar errores comunes
        en conversiones de coordenadas geográficas. Siempre úsala antes de
        aplicar sin() o cos() a latitudes/longitudes en grados.
    """
    return angulo * math.pi / 180

def gms2dec(grados: float, minutos: float = 0.0, segundos: float = 0.0) -> float:
    """
    Convierte GMS (grados, minutos, segundos) a grados decimales.
    Maneja correctamente grados negativos (Sur/Oeste).
    
    Ejemplos:
        gms2dec(40, 25, 46.8)      →  40.429666... (Norte/Este)
        gms2dec(-33, 26, 45.9)     → -33.446083... (Sur/Oeste)
    """
    fraccion = abs(minutos) / 60.0 + abs(segundos) / 3600.0
    if grados >= 0:
        return grados + fraccion
    else:
        return grados - fraccion