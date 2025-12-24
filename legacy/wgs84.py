import math

# Semieje mayor (radio ecuatorial) en metros.
# Representa la distancia desde el centro de la Tierra hasta el ecuador.
# Valor oficial definido por el estándar WGS84 (NGA, 1984).
a = 6378137.0

# Semieje menor (radio polar) en metros.
# Representa la distancia desde el centro de la Tierra hasta los polos.
# Valor oficial definido por el estándar WGS84, con precisión para aplicaciones de alta exactitud.
# Nota: WGS84 es casi idéntico al GRS80, con diferencias mínimas (~0.1 mm en b).
b = 6356752.314245

def calc_N(phi_degrees: float, a: float = 6378137.0, b: float = 6356752.314245) -> float:
    """
    Calcula el radio de curvatura en el primer vertical N(φ) para el elipsoide WGS84.
    
    N(φ) representa la distancia desde el eje de rotación de la Tierra hasta la superficie del elipsoide,
    en la dirección perpendicular al plano meridiano (este-oeste) en un punto de latitud geodésica φ.
    
    Fórmula estándar utilizada:
    
        N(φ) = a / √(1 - e² sin² φ)
    
    donde:
        - a  = semieje mayor (radio ecuatorial) = 6378137.0 m
        - b  = semieje menor (radio polar)     = 6356752.314245 m
        - e² = primera excentricidad al cuadrado = (a² - b²) / a²
        - φ  = latitud geodésica en grados
    
    Esta fórmula es numéricamente más estable y precisa que la forma alternativa
    a² / √(a² cos² φ + b² sin² φ), especialmente cerca de los polos, porque evita
    la resta de números grandes en el denominador y utiliza directamente la excentricidad.
    
    Parámetros:
        phi_degrees (float): Latitud geodésica en grados decimales (-90 a +90).
        a (float, opcional): Semieje mayor en metros. Valor por defecto WGS84.
        b (float, opcional): Semieje menor en metros. Valor por defecto WGS84.
    
    Retorna:
        float: Radio de curvatura N en metros.
    
    Ejemplo:
        >>> calc_N(0)      # Ecuador
        6378137.0
        >>> calc_N(90)     # Polo
        6399593.625758493
    """
    phi_rad = math.radians(phi_degrees)                # Convertir a radianes
    e2 = (a**2 - b**2) / a**2                           # Primera excentricidad al cuadrado
    denominator = math.sqrt(1 - e2 * math.sin(phi_rad)**2)
    return a / denominator

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

def conversiongeo2cart(phi_grados: float, lam_grados: float, h: float,
                      a: float = 6378137.0, b: float = 6356752.314245) -> tuple[float, float, float]:
    """
    Convierte coordenadas geodésicas (latitud, longitud, altura) a coordenadas cartesianas ECEF
    (Earth-Centered, Earth-Fixed) usando el elipsoide WGS84.
    
    Fórmulas estándar:
        X = (N + h) ⋅ cos φ ⋅ cos λ
        Y = (N + h) ⋅ cos φ ⋅ sin λ
        Z = (N ⋅ (1 - e²) + h) ⋅ sin φ
    
    donde:
        - N(φ) = a / √(1 - e² sin² φ)     → radio de curvatura prime vertical
        - e² = (a² - b²) / a²             → primera excentricidad al cuadrado
        - φ = latitud geodésica en radianes
        - λ = longitud en radianes
        - h = altura elipsoidal en metros (puede ser negativa)
    
    Nota importante:
        El término (b²/a²) es igual a (1 - e²), por lo que Z también se puede escribir como:
        Z = (N ⋅ (1 - e²) + h) ⋅ sin φ
    
    Parámetros:
        phi_grados (float): Latitud geodésica en grados decimales (-90 a +90).
        lam_grados (float): Longitud en grados decimales (-180 a +180).
        h (float): Altura elipsoidal sobre el elipsoide en metros.
                   Puede ser negativa (ej. Mar Muerto ≈ -430 m).
        a (float): Semieje mayor (por defecto WGS84: 6378137.0 m).
        b (float): Semieje menor (por defecto WGS84: 6356752.314245 m).
    
    Retorna:
        tuple[float, float, float]: Coordenadas (X, Y, Z) en metros en sistema ECEF.
    
    Ejemplos:
        >>> conversiongeo2cart(0, 0, 0)          # Ecuador, Greenwich, sobre el elipsoide
        (6378137.0, 0.0, 0.0)
        >>> conversiongeo2cart(90, 0, 0)         # Polo Norte
        (0.0, 0.0, 6356752.314245)
        >>> conversiongeo2cart(40.4168, -3.7038, 650)  # Madrid aprox.
        (≈4510203, ≈-364978, ≈4150000)
    """
    # Convertir ángulos a radianes (latitud y longitud sí, altura NO)
    phi = grados2rad(phi_grados)
    lam = grados2rad(lam_grados)
    
    # Calcular N(φ) - radio de curvatura prime vertical
    # Usamos la fórmula estándar para máxima precisión
    e2 = (a**2 - b**2) / a**2                    # Primera excentricidad al cuadrado
    N = a / math.sqrt(1 - e2 * math.sin(phi)**2)
    
    # Cálculo de coordenadas cartesianas ECEF
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    cos_lam = math.cos(lam)
    sin_lam = math.sin(lam)
    
    X = (N + h) * cos_phi * cos_lam
    Y = (N + h) * cos_phi * sin_lam
    Z = (N * (1 - e2) + h) * sin_phi      # Forma más precisa y común que (N * (b²/a²) + h)
    
    return X, Y, Z

def conversioncart2geo(X: float, Y: float, Z: float,
                      a: float = 6378137.0, b: float = 6356752.314245) -> tuple[float, float, float]:
    """
    Convierte coordenadas cartesianas ECEF (X, Y, Z) a coordenadas geodésicas
    (latitud φ, longitud λ, altura elipsoidal h) usando el elipsoide WGS84.
    
    Método: Iteración de Bowring mejorada (alta precisión, converge en pocas iteraciones).
    Precisión típica: mejor que 1 mm en altura y 10^-12 grados en latitud/longitud.
    
    Parámetros:
        X, Y, Z (float): Coordenadas cartesianas en metros (sistema ECEF).
        a (float): Semieje mayor (por defecto WGS84: 6378137.0 m).
        b (float): Semieje menor (por defecto WGS84: 6356752.314245 m).
    
    Retorna:
        tuple[float, float, float]:
            phi_grados (latitud en grados decimales, -90 a +90)
            lam_grados (longitud en grados decimales, -180 a +180)
            h (altura elipsoidal en metros, puede ser negativa)
    
    Ejemplos:
        >>> conversioncart2geo(6378137.0, 0.0, 0.0)
        (0.0, 0.0, 0.0)                  # Ecuador, sobre el elipsoide
        >>> conversioncart2geo(0.0, 0.0, 6356752.314245)
        (90.0, 0.0, 0.0)                 # Polo Norte
    """
    # Tolerancia para convergencia (muy estricta, pero segura)
    TOLERANCIA = 1e-12
    
    # Primera excentricidad al cuadrado
    e2 = 1 - (b**2 / a**2)
    
    # Distancia proyectada en el plano ecuatorial
    p = math.sqrt(X**2 + Y**2)
    
    # Longitud (con math.atan2 para manejar todos los cuadrantes correctamente)
    if p < 1e-10:  # Casi en el eje polar
        lam_grados = 0.0
    else:
        lam_grados = math.degrees(math.atan2(Y, X))
    
    # Caso especial: cerca de los polos (evita división por cero)
    if p < 1.0:  # Umbral arbitrario pero seguro
        if abs(Z) > abs(p):
            # Aproximación inicial cerca de polos
            h = abs(Z) - b
            phi_grados = 90.0 if Z >= 0 else -90.0
        else:
            h = p - a
            phi_grados = 0.0
        return phi_grados, lam_grados, h
    
    # Aproximación inicial de Bowring (muy buena)
    theta = math.atan2(Z * a, p * b)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    
    phi = math.atan2(Z + e2 * a * sin_theta**3,
                     p - e2 * a * cos_theta**3)
    
    # Iteración para refinar latitud y altura
    iteracion = 0
    while iteracion < 20:  # Máximo 20 iteraciones (nunca se alcanza en práctica)
        sin_phi = math.sin(phi)
        N = a / math.sqrt(1 - e2 * sin_phi**2)      # Radio de curvatura prime vertical
        h_prev = h if iteracion > 0 else 0.0
        phi_prev = phi
        
        h = p / math.cos(phi) - N
        phi = math.atan2(Z, p * (1 - e2 * N / (N + h)))
        
        # Condición de convergencia
        if abs(phi - phi_prev) < TOLERANCIA and abs(h - h_prev) < TOLERANCIA:
            break
        
        iteracion += 1
    
    phi_grados = math.degrees(phi)
    
    return phi_grados, lam_grados, h

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