import numpy as np


def weights_from_distance(
    d_obs: np.ndarray,
    *,
    sigma0_mm: float,
    ppm: float
) -> np.ndarray:
    """
    Calcula los pesos de observaciones de distancia a partir de un modelo de precisión típico
    en topografía / geodesia.

    Modelo estocástico de la distancia
    ----------------------------------
        σ(d) = sqrt( σ0² + (ppm · d)² )

    donde:
        σ0  : término constante del instrumento (ruido base), en milímetros
        ppm : término proporcional a la distancia (partes por millón)
        d   : distancia observada, en metros

    Entradas
    --------
    d_obs : np.ndarray (m,)
        Vector de distancias observadas (en metros).
    sigma0_mm : float
        Precisión constante del instrumento en milímetros (ej. 1–5 mm).
    ppm : float
        Precisión proporcional en ppm (ej. 1–3 ppm).

    Salida
    ------
    w : np.ndarray (m,)
        Vector de pesos estadísticos:
            w = 1 / σ(d)²
        con unidades 1/m², listo para construir la matriz P = diag(w).
    """

    # ------------------------------------------------------------
    # Conversión del término constante de mm a metros
    # ------------------------------------------------------------
    # σ0 [mm] -> σ0 [m]
    sigma0_m = sigma0_mm / 1000.0

    # ------------------------------------------------------------
    # Término proporcional (ppm)
    # ------------------------------------------------------------
    # ppm está en partes por millón (1e-6)
    # σ_ppm(d) = ppm * 1e-6 * d
    sigma_ppm_m = (ppm * 1e-6) * d_obs

    # ------------------------------------------------------------
    # Desviación estándar total de cada observación
    # ------------------------------------------------------------
    # σ(d) = sqrt( σ0² + σ_ppm(d)² )
    sigma_m = np.sqrt(sigma0_m**2 + sigma_ppm_m**2)

    # ------------------------------------------------------------
    # Pesos estadísticos
    # ------------------------------------------------------------
    # En mínimos cuadrados: w = 1 / σ²
    #
    # np.where evita divisiones por cero en caso degenerado
    # (aunque en práctica sigma_m > 0 siempre debería cumplirse)
    return np.where(sigma_m > 0, 1.0 / (sigma_m**2), 0.0)
