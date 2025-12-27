from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AdjustmentResult:
    points: pd.DataFrame
    observations: pd.DataFrame
    stats: dict


@dataclass
class AdjustmentData:
    names: list[str]
    coords0: np.ndarray     # (n,3) inicial
    pi: np.ndarray          # (m,) names i
    pj: np.ndarray          # (m,) names j
    d_obs: np.ndarray       # (m,) metros


def initial_coords_from_df(df: pd.DataFrame, id1: str, id2: str) -> tuple[list[str], np.ndarray]:
    p1 = pd.DataFrame({
        "name": df[id1].astype(str),
        "X": df["X1"].astype(float),
        "Y": df["Y1"].astype(float),
        "Z": df["Z1"].astype(float),
    })
    p2 = pd.DataFrame({
        "name": df[id2].astype(str),
        "X": df["X2"].astype(float),
        "Y": df["Y2"].astype(float),
        "Z": df["Z2"].astype(float),
    })
    pts = pd.concat([p1, p2], ignore_index=True)
    g = pts.groupby("name", as_index=False).mean(numeric_only=True)
    names = g["name"].tolist()
    coords0 = g[["X", "Y", "Z"]].to_numpy(dtype=float)
    return names, coords0


def build_index(names: list[str]) -> dict[str, int]:
    return {n: i for i, n in enumerate(names)}


def extract_adjustment_data(df: pd.DataFrame, *, id1: str, id2: str) -> AdjustmentData:
    req = {id1, id2, "distancia", "X1", "Y1", "Z1", "X2", "Y2", "Z2"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}. Disponibles: {list(df.columns)}")

    names, coords0 = initial_coords_from_df(df, id1=id1, id2=id2)
    pi = df[id1].astype(str).to_numpy()
    pj = df[id2].astype(str).to_numpy()
    d_obs = df["distancia"].to_numpy(dtype=float)

    return AdjustmentData(names=names, coords0=coords0, pi=pi, pj=pj, d_obs=d_obs)
