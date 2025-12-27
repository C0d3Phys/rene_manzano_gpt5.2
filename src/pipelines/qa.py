import numpy as np
import pandas as pd

def _nanmean(x): return float(np.nanmean(x))
def _nanstd(x):  return float(np.nanstd(x, ddof=1))

def _median(x):  return float(np.nanmedian(x))

def _mad(x):
    """
    MAD = median(|x - median(x)|)
    """
    med = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - med)))

def qa_columns(
    df: pd.DataFrame,
    col_res_mm: str = "dif_mm",
    # Umbrales prácticos (ajústalos según tu control):
    suspect_mm: float = 5.0,
    outlier_mm: float = 10.0,
    z_suspect: float = 3.0,
    z_outlier: float = 5.0,
    robust_z_suspect: float = 3.5,
    robust_z_outlier: float = 6.0,
) -> tuple[pd.DataFrame, dict]:
    """
    QA geodésico sobre residuos en mm.
    Devuelve (df_out, stats_dict).
    """
    out = df.copy()

    v = out[col_res_mm].to_numpy(dtype=float)

    # Estadística clásica
    mu = _nanmean(v)
    sig = _nanstd(v)
    if not np.isfinite(sig) or sig == 0.0:
        z = np.full_like(v, np.nan, dtype=float)
    else:
        z = (v - mu) / sig

    # Estadística robusta (MAD escalado)
    med = _median(v)
    mad = _mad(v)
    # Escala: sigma_rob ≈ 1.4826 * MAD (asumiendo normal)
    sigma_rob = 1.4826 * mad if (np.isfinite(mad) and mad != 0.0) else np.nan
    if not np.isfinite(sigma_rob) or sigma_rob == 0.0:
        rz = np.full_like(v, np.nan, dtype=float)
    else:
        rz = (v - med) / sigma_rob

    out["z_mm"] = z
    out["robust_z_mm"] = rz

    abs_v = np.abs(v)
    abs_z = np.abs(z)
    abs_rz = np.abs(rz)

    # Reglas de marcado (combinadas: magnitud mm + z)
    is_outlier = (
        (abs_v >= outlier_mm) |
        (abs_z >= z_outlier) |
        (abs_rz >= robust_z_outlier)
    )

    is_suspect = (
        ~is_outlier &
        (
            (abs_v >= suspect_mm) |
            (abs_z >= z_suspect) |
            (abs_rz >= robust_z_suspect)
        )
    )

    flag = np.where(is_outlier, "OUTLIER", np.where(is_suspect, "SUSPECT", "OK"))

    out["is_outlier"] = is_outlier
    out["is_suspect"] = is_suspect
    out["flag_qa"] = flag

    # Resumen global útil para impresión o log
    stats = {
        "n": int(np.isfinite(v).sum()),
        "mean_mm": mu,
        "std_mm": sig,
        "median_mm": med,
        "mad_mm": mad,
        "sigma_rob_mm": float(sigma_rob) if np.isfinite(sigma_rob) else np.nan,
        "p50_mm": float(np.nanpercentile(v, 50)),
        "p95_mm": float(np.nanpercentile(v, 95)),
        "p99_mm": float(np.nanpercentile(v, 99)),
        "count_ok": int(np.sum(flag == "OK")),
        "count_suspect": int(np.sum(flag == "SUSPECT")),
        "count_outlier": int(np.sum(flag == "OUTLIER")),
    }

    return out, stats
