from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


# =============================================================================
# ESTRUCTURAS DE DATOS (Data classes)
# =============================================================================

@dataclass
class AdjustmentResult:
    """
    Contenedor del resultado final de un ajuste (mínimos cuadrados, etc.).

    - points: DataFrame con puntos ajustados (coordenadas finales, correcciones,
      sigmas, etc. según lo que construya el solver).
    - observations: DataFrame con observaciones y sus resultados (calculadas,
      residuos, etc.).
    - stats: diccionario con estadísticas globales (sigma0, dof, RMS, etc.).
    """
    points: pd.DataFrame
    observations: pd.DataFrame
    stats: dict


@dataclass
class AdjustmentData:
    """
    Paquete mínimo de datos para iniciar un ajuste basado en distancias entre puntos.

    - names: lista de nombres únicos de puntos (tamaño n).
    - coords0: coordenadas iniciales (aproximadas) de cada punto, matriz (n, 3).
              Ordenadas exactamente como 'names'.
    - pi: vector (m,) con el nombre del punto i (origen) para cada observación.
    - pj: vector (m,) con el nombre del punto j (destino) para cada observación.
    - d_obs: vector (m,) con las distancias observadas (en metros).
    """
    names: list[str]
    coords0: np.ndarray     # (n, 3) inicial
    pi: np.ndarray          # (m,) names i
    pj: np.ndarray          # (m,) names j
    d_obs: np.ndarray       # (m,) metros


# =============================================================================
# FUNCIONES DE PREPARACIÓN / EXTRACCIÓN
# =============================================================================

def initial_coords_from_df(df: pd.DataFrame, id1: str, id2: str) -> tuple[list[str], np.ndarray]:
    """
    Construye coordenadas iniciales (aproximación inicial) para cada punto.

    La idea:
    - Cada fila del df representa una observación de distancia entre dos puntos (id1, id2).
    - En esa fila también vienen coordenadas "X1,Y1,Z1" (del punto id1) y "X2,Y2,Z2" (del punto id2).
    - Un mismo punto puede aparecer en muchas filas; entonces:
        1) juntamos todos los registros de puntos (lado 1 y lado 2),
        2) agrupamos por nombre del punto,
        3) promediamos coordenadas para obtener una coordenada inicial estable.

    Retorna:
    - names: lista de nombres únicos de puntos (ordenados por el groupby result).
    - coords0: matriz (n,3) con [X,Y,Z] inicial para cada punto en el mismo orden de names.
    """
    # --- Construye tabla de puntos del lado 1: (id1, X1, Y1, Z1) ---
    p1 = pd.DataFrame({
        "name": df[id1].astype(str),     # nombre del punto i (string)
        "X": df["X1"].astype(float),     # X del punto i
        "Y": df["Y1"].astype(float),     # Y del punto i
        "Z": df["Z1"].astype(float),     # Z del punto i
    })

    # --- Construye tabla de puntos del lado 2: (id2, X2, Y2, Z2) ---
    p2 = pd.DataFrame({
        "name": df[id2].astype(str),     # nombre del punto j (string)
        "X": df["X2"].astype(float),     # X del punto j
        "Y": df["Y2"].astype(float),     # Y del punto j
        "Z": df["Z2"].astype(float),     # Z del punto j
    })

    # Une ambas tablas en una sola lista de puntos (duplicados incluidos)
    pts = pd.concat([p1, p2], ignore_index=True)

    # Agrupa por nombre y promedia coordenadas numéricas
    # numeric_only=True asegura que solo promedie X,Y,Z
    g = pts.groupby("name", as_index=False).mean(numeric_only=True)

    # Extrae la lista de nombres únicos (n puntos)
    names = g["name"].tolist()

    # Extrae coordenadas iniciales como numpy array (n,3)
    coords0 = g[["X", "Y", "Z"]].to_numpy(dtype=float)

    return names, coords0


def build_index(names: list[str]) -> dict[str, int]:
    """
    Construye un índice name -> i, donde i es la fila en coords0.

    Ejemplo:
        names = ["A","B","C"]  ->  {"A":0,"B":1,"C":2}

    Esto es útil para:
    - mapear rápidamente nombres a posiciones en vectores/matrices
    - construir matrices de diseño A (jacobianos) sin búsquedas O(n)
    """
    return {n: i for i, n in enumerate(names)}


def extract_adjustment_data(df: pd.DataFrame, *, id1: str, id2: str) -> AdjustmentData:
    """
    Extrae y valida datos desde un DataFrame crudo hacia un AdjustmentData listo para un solver.

    Requiere:
    - columnas de identificación: id1, id2 (por ejemplo "punto1", "punto2")
    - distancia observada: "distancia"
    - coordenadas asociadas a cada extremo: X1,Y1,Z1 y X2,Y2,Z2

    Flujo:
    1) valida columnas mínimas requeridas
    2) calcula coordenadas iniciales por promedio (initial_coords_from_df)
    3) arma vectores pi, pj (nombres por observación) y d_obs (distancias)
    4) empaqueta en AdjustmentData
    """
    # Conjunto de columnas mínimas necesarias
    req = {id1, id2, "distancia", "X1", "Y1", "Z1", "X2", "Y2", "Z2"}

    # Detecta columnas faltantes para fallar con mensaje claro
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(
            f"Faltan columnas requeridas: {missing}. Disponibles: {list(df.columns)}"
        )

    # Calcula nombres únicos y coordenadas iniciales (n,3)
    names, coords0 = initial_coords_from_df(df, id1=id1, id2=id2)

    # Vector de nombres del punto i por observación (m,)
    pi = df[id1].astype(str).to_numpy()

    # Vector de nombres del punto j por observación (m,)
    pj = df[id2].astype(str).to_numpy()

    # Vector de distancias observadas (m,) en float
    d_obs = df["distancia"].to_numpy(dtype=float)

    # Empaqueta todo listo para el ajuste
    return AdjustmentData(names=names, coords0=coords0, pi=pi, pj=pj, d_obs=d_obs)
