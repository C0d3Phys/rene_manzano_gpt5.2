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

def convertir_dd_mmssssss(valor: float) -> float:
    """
    Convierte coordenada en formato DD.MMssssss a grados decimales.
    
    Ejemplos:
        19.192543   → 19 + 19/60 + 25.43/3600 ≈ 19.32373055°
        -99.114174  → -(99 + 11/60 + 41.174/3600) ≈ -99.19477055°
    """
    # Manejar signo negativo
    signo = -1 if valor < 0 else 1
    valor_abs = abs(valor)
    
    # Separar parte entera (grados + minutos como decimal)
    grados_enteros = int(valor_abs)                   # DD
    minutos_decimal = valor_abs - grados_enteros      # .MMssssss
    
    # Convertir minutos_decimal a minutos y segundos
    minutos = int(minutos_decimal * 100)              # MM (enteros)
    segundos = (minutos_decimal * 100 - minutos) * 10000 / 100  # ssssss → segundos con decimal
    
    # Cálculo final
    decimal = grados_enteros + minutos / 60.0 + segundos / 3600.0
    
    return signo * decimal

def leer_datos_apoyo(ruta_archivo: str = "datos_apoyo.txt") -> pd.DataFrame:
    df = pd.read_csv(
        ruta_archivo,
        delimiter=r"\s*,\s*|\s+",
        engine="python",
        comment='#',
        header=None,
        on_bad_lines='skip'
    )
    
    df = df.dropna(thresh=9).reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    df.columns = ['lat1_coded', 'lon1_coded', 'h1', 'lat2_coded', 'lon2_coded', 'h2', 'distancia', 'punto1', 'punto2']
    df['punto1'] = df['punto1'].astype('Int64')
    df['punto2'] = df['punto2'].astype('Int64')
    
    df = df.dropna(subset=['lat1_coded', 'lon1_coded', 'lat2_coded', 'lon2_coded'])
    
    print(f"Archivo '{ruta_archivo}' cargado correctamente.")
    print(f"{len(df)} líneas de control válidas.")
    return df

def procesar_control_geodesico(df: pd.DataFrame) -> pd.DataFrame:
    # Convertir coordenadas codificadas a grados decimales
    df['lat1'] = df['lat1_coded'].apply(convertir_dd_mmssssss)
    df['lon1'] = df['lon1_coded'].apply(convertir_dd_mmssssss)
    df['lat2'] = df['lat2_coded'].apply(convertir_dd_mmssssss)
    df['lon2'] = df['lon2_coded'].apply(convertir_dd_mmssssss)
    
    # Convertir a ECEF
    df[['X1', 'Y1', 'Z1']] = df.apply(
        lambda r: conversiongeo2cart(r['lat1'], r['lon1'], r['h1']),
        axis=1, result_type='expand'
    )
    df[['X2', 'Y2', 'Z2']] = df.apply(
        lambda r: conversiongeo2cart(r['lat2'], r['lon2'], r['h2']),
        axis=1, result_type='expand'
    )
    
    # Distancia calculada
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
# EJECUCIÓN DIRECTA
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
    
    # Opcional: mostrar coordenadas convertidas
    print("\nEjemplo de conversión (primera línea):")
    print(f"Lat1 codificada: {df_raw.iloc[0]['lat1_coded']} → {df.iloc[0]['lat1']:.8f}°")
    print(f"Lon1 codificada: {df_raw.iloc[0]['lon1_coded']} → {df.iloc[0]['lon1']:.8f}°")