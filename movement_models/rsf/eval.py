from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..sampling import _get_sampling_points, _sample_env_layer

def fixed_width_Boyce(pred, rsf, domain, n_bg_points=100_000, n_bins=20, seed=42, pred_col="rsf_pred"):
    try:
        rsf_crs = rsf.rio.crs
    except Exception as e:
        raise ValueError("rsf has no .rio CRS; set it via rsf = rsf.rio.write_crs('EPSG:....')") from e
    if rsf_crs is None:
        raise ValueError("rsf.rio.crs is None; set it via rsf = rsf.rio.write_crs('EPSG:....')")

    if domain.crs != rsf_crs:
        domain = domain.to_crs(rsf_crs)

    bg_points = _get_sampling_points(domain, n_bg_points, seed=seed)
    bg_samples = _sample_env_layer(bg_points, rsf)

    b = bg_samples["rsf"].to_numpy(dtype=float)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return np.nan, pd.DataFrame({"q_mid": [], "rsf_mid": [], "pe": []})
    bg = np.log(b + 1e-12)

    used_mask = pred["used"].astype(bool)
    u = pred.loc[used_mask, pred_col].to_numpy(dtype=float)
    u = u[np.isfinite(u)]
    if u.size == 0:
        return np.nan, pd.DataFrame({"q_mid": [], "rsf_mid": [], "pe": []})
    used = np.log(u + 1e-12)

    q_edges = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(bg, q_edges)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.nan, pd.DataFrame({"q_mid": [], "rsf_mid": [], "pe": []})

    u_cnt, _ = np.histogram(used, bins=edges)
    b_cnt, _ = np.histogram(bg, bins=edges)

    u_prop = u_cnt / used.size
    b_prop = b_cnt / bg.size
    pe = np.divide(u_prop, b_prop, out=np.full_like(u_prop, np.nan, dtype=float), where=(b_prop > 0))

    mids = 0.5 * (edges[:-1] + edges[1:])
    q_mids = (np.arange(len(pe)) + 0.5) / len(pe)

    chart = pd.DataFrame({"q_mid": q_mids, "rsf_mid": mids, "pe": pe})
    B, _ = spearmanr(mids, pe, nan_policy="omit")
    return B, chart


def sliding_window_Boyce(pred: gpd.GeoDataFrame, rsf: xarray.DataArray, domain, window_frac: float = 0.1, step_frac: float = 0.02, seed = 42):

    bg_points = _get_sampling_points(domain, len(pred) * 10, df=None, seed=seed)
    bg_samples = _sample_env_layer(bg_points, rsf)
    bg = np.log(bg_samples["rsf"] + + 1e-12)
    used = np.log(pred.loc[pred["used"] == True, "rsf_pred"].to_numpy(dtype=float) + + 1e-12)

    q_starts = np.arange(0, 1.0 - window_frac + 1e-12, step_frac)
    q_ends = q_starts + window_frac

    rows = []
    bg_n = bg.size
    used_n = used.size

    for qs, qe in zip(q_starts, q_ends):
        lo, hi = np.quantile(bg, [qs, qe])

        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            continue
        
        in_bg = (bg >= lo) & (bg < hi)
        b_cnt = in_bg.sum()
        
        if b_cnt == 0:
            continue
        b_prop = b_cnt / bg_n

        in_used = (used >= lo) & (used < hi)
        u_cnt = in_used.sum()
        u_prop = u_cnt / used_n
        
        pe = u_prop / b_prop
        mid = 0.5 * (lo + hi)
        rows.append((mid, pe))

    boyce_chart = pd.DataFrame(rows, columns = ["rsf_mid", "pe"])
    B, _ = spearmanr(boyce_chart["rsf_mid"], boyce_chart["pe"])
    return B, boyce_chart