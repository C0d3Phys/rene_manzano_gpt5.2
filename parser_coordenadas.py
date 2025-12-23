# parser_coordenadas.py
import pandas as pd
import numpy as np
import math

# Parámetros WGS84
a = 6378137.0
b = 6356752.314245

def grados2rad(angulo: float) -> float:
    return angulo * math.pi / 180

def calc_N(phi_grados: float) -> float:
    phi = grados2rad(phi_grados)
    e2 = (a**2 - b**2) / a**2
    return a / math.sqrt(1 - e2 * math.sin(phi)**2)

def conversiongeo2cart(phi_grados: float, lam_grados: float, h: float = 0.0) -> tuple[float, float, float]:
    phi = grados2rad(phi_grados)
    lam = grados2rad(lam_grados)
    N = calc_N(phi_grados)
    e2 = (a**2 - b**2) / a**2
    
    X = (N + h) * math.cos(phi) * math.cos(lam)
    Y = (N + h) * math.cos(phi) * math.sin(lam)
    Z = (N * (1 - e2) + h) * math.sin(phi)
    return X, Y, Z

def dist2p(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p2 - p1)

def leer_datos_apoyo(ruta_archivo: str = "datos_apoyo.txt") -> pd.DataFrame:
    """
    Lee el archivo datos_apoyo.txt con coordenadas ya en grados decimales.
    """
    df = pd.read_csv(
        ruta_archivo,
        delimiter=r"\s*,\s*|\s+",      # separa por coma o espacios
        engine="python",
        comment='#',
        header=None,
        on_bad_lines='skip'
    )
    
    # Nos quedamos con filas que tengan al menos 9 columnas
    df = df.dropna(thresh=9).reset_index(drop=True)
    
    # Convertimos todo a numérico directamente
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Asignamos nombres de columnas
    df.columns = ['lat1', 'lon1', 'h1', 'lat2', 'lon2', 'h2', 'distancia', 'punto1', 'punto2']
    
    # Convertimos códigos de punto a enteros (quitando posibles puntos)
    df['punto1'] = df['punto1'].astype('Int64')
    df['punto2'] = df['punto2'].astype('Int64')
    
    # Eliminamos filas con coordenadas inválidas
    df = df.dropna(subset=['lat1', 'lon1', 'lat2', 'lon2'])
    
    print(f"Archivo '{ruta_archivo}' cargado correctamente.")
    print(f"{len(df)} líneas de control válidas.")
    return df

def procesar_control_geodesico(df: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa el control con coordenadas ya en grados decimales.
    """
    # Usamos directamente las columnas lat1, lon1, etc. (ya son decimales)
    
    # Punto 1 → Cartesianas
    df[['X1', 'Y1', 'Z1']] = df.apply(
        lambda r: conversiongeo2cart(r['lat1'], r['lon1'], r['h1']),
        axis=1, result_type='expand'
    )
    
    # Punto 2 → Cartesianas
    df[['X2', 'Y2', 'Z2']] = df.apply(
        lambda r: conversiongeo2cart(r['lat2'], r['lon2'], r['h2']),
        axis=1, result_type='expand'
    )
    
    # Distancia calculada 3D
    df['dist_calc'] = df.apply(
        lambda r: dist2p(np.array([r['X1'], r['Y1'], r['Z1']]),
                         np.array([r['X2'], r['Y2'], r['Z2']])), axis=1
    )
    
    # Diferencias
    df['dif_m'] = df['dist_calc'] - df['distancia']
    df['dif_mm'] = df['dif_m'] * 1000
    df['dif_ppm'] = np.where(df['distancia'] != 0, (df['dif_m'] / df['distancia']) * 1_000_000, 0)
    
    return df

# ================================
# PRUEBA AL EJECUTAR EL ARCHIVO
# ================================
if __name__ == "__main__":
    df_raw = leer_datos_apoyo()
    df = procesar_control_geodesico(df_raw.copy())
    
    print("\n" + "="*60)
    print("          RESUMEN DE CONTROL GEODÉSICO")
    print("="*60)
    print(f"Líneas procesadas       : {len(df)}")
    print(f"Desviación máxima       : {df['dif_mm'].abs().max():.4f} mm")
    print(f"Desviación media        : {df['dif_mm'].abs().mean():.4f} mm")
    print(f"Desviación típica       : {df['dif_mm'].std():.4f} mm")
    print(f"Precisión media (ppm)   : {df['dif_ppm'].abs().mean():.2f} ppm")
    print("="*60)
    
    print("\nDetalle de diferencias:")
    print(df[['punto1', 'punto2', 'distancia', 'dist_calc', 'dif_mm', 'dif_ppm']]
          .round({'distancia': 4, 'dist_calc': 4, 'dif_mm': 4, 'dif_ppm': 2}))