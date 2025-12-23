# parser_coordenadas.py
import pandas as pd
import numpy as np

def leer_datos_apoyo(ruta_archivo: str = "datos_apoyo.txt") -> pd.DataFrame:
    """
    Lee el archivo datos_apoyo.txt y devuelve un DataFrame limpio.
    
    Formato esperado (separado por comas o espacios):
    lat1, lon1, h1, lat2, lon2, h2, distancia, punto1, punto2
    
    Ignora líneas de comentario (#) y líneas vacías.
    """
    # Usamos pandas porque maneja mejor archivos irregulares
    df = pd.read_csv(
        ruta_archivo,
        delimiter=r"\s*,\s*|\s+",          # separa por coma o por espacios múltiples
        engine="python",                   # necesario para regex en delimiter
        comment='#',                       # ignora líneas que empiecen con #
        header=None,                       # no hay encabezado
        on_bad_lines='skip',               # salta líneas mal formadas
        dtype=str                          # lee todo como string primero para limpiar
    )
    
    # Nos quedamos solo con filas que tengan al menos 9 columnas
    df = df.dropna(thresh=9)
    
    # Convertimos las columnas numéricas
    columnas_numericas = [0,1,2,3,4,5,6]  # lat1, lon1, h1, lat2, lon2, h2, dist
    for col in columnas_numericas:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Columnas de códigos de punto (7 y 8) como enteros o strings limpios
    df[7] = pd.to_numeric(df[7].str.replace(r'\.', '', regex=True), errors='coerce').astype('Int64')
    df[8] = pd.to_numeric(df[8].str.replace(r'\.', '', regex=True), errors='coerce').astype('Int64')
    
    # Asignamos nombres claros
    df.columns = ['lat1', 'lon1', 'h1', 'lat2', 'lon2', 'h2', 'distancia', 'punto1', 'punto2']
    
    # Eliminamos filas con NaN en coordenadas esenciales
    df = df.dropna(subset=['lat1', 'lon1', 'lat2', 'lon2'])
    
    print(f"Archivo '{ruta_archivo}' cargado correctamente.")
    print(f"{len(df)} líneas de control válidas.")
    
    return df

# Prueba rápida (descomenta para probar directamente)
if __name__ == "__main__":
    datos = leer_datos_apoyo()
    print(datos)
    print("\nPrimeras filas:")
    print(datos.head())