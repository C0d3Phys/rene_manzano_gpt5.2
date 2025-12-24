# src/tools/parsers.py
def ddmmssss_to_deg(value: float) -> float:
    """
    Decodifica coordenadas en formato DD.MMssssss a grados decimales reales.

    Formato asumido:
        DD.MMssssss
        - DD = grados
        - MM = minutos
        - ssssss = segundos (con decimales)

    Ejemplos:
        19.192543   -> 19 + 19/60 + 25.43/3600
        -99.114174  -> -(99 + 11/60 + 41.174/3600)
    """
    sign = -1.0 if value < 0 else 1.0
    v = abs(value)

    deg = int(v)
    mmssss = (v - deg) * 100.0

    minutes = int(mmssss)
    seconds = (mmssss - minutes) * 100.0

    return sign * (deg + minutes / 60.0 + seconds / 3600.0)
