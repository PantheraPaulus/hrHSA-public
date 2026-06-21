from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hsa.sampling import sample_available_points, sample_raster_stack


def boyce_quantile_bins(
    pred: pd.DataFrame,
    rsf,
    domain,
    *,
    n_background_points: int = 100_000,
    n_bins: int = 20,
    seed: int = 42,
    pred_col: str = "rsf_pred",
) -> tuple[float, pd.DataFrame]:
    """Continuous Boyce-style index using equal-frequency background bins."""

    bg_points = sample_available_points(domain, n_background_points, seed=seed)
    bg_samples = sample_raster_stack(bg_points, rsf)

    background = np.log(bg_samples["rsf"].to_numpy(dtype=float) + 1e-12)
    used = np.log(pred.loc[pred["used"].astype(bool), pred_col].to_numpy(dtype=float) + 1e-12)
    background = background[np.isfinite(background)]
    used = used[np.isfinite(used)]

    if background.size == 0 or used.size == 0:
        return np.nan, pd.DataFrame(columns=["q_mid", "rsf_mid", "used_n", "bg_n", "pe"])

    edges = np.quantile(background, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.nan, pd.DataFrame(columns=["q_mid", "rsf_mid", "used_n", "bg_n", "pe"])

    used_counts, _ = np.histogram(used, bins=edges)
    bg_counts, _ = np.histogram(background, bins=edges)
    used_prop = used_counts / used.size
    bg_prop = bg_counts / background.size
    pe = np.divide(used_prop, bg_prop, out=np.full_like(used_prop, np.nan, dtype=float), where=bg_prop > 0)

    chart = pd.DataFrame(
        {
            "q_mid": (np.arange(len(pe)) + 0.5) / len(pe),
            "rsf_mid": 0.5 * (edges[:-1] + edges[1:]),
            "used_n": used_counts,
            "bg_n": bg_counts,
            "pe": pe,
        }
    )
    valid = np.isfinite(chart["rsf_mid"]) & np.isfinite(chart["pe"])
    if valid.sum() < 2:
        return np.nan, chart
    boyce, _ = spearmanr(chart.loc[valid, "rsf_mid"], chart.loc[valid, "pe"])
    return float(boyce), chart


def boyce_sliding_window(
    pred: pd.DataFrame,
    rsf,
    domain,
    *,
    n_background_points: int = 100_000,
    window_fraction: float = 0.1,
    step_fraction: float = 0.02,
    seed: int = 42,
    pred_col: str = "rsf_pred",
) -> tuple[float, pd.DataFrame]:
    """Continuous Boyce-style index using overlapping background quantile windows."""

    bg_points = sample_available_points(domain, n_background_points, seed=seed)
    bg_samples = sample_raster_stack(bg_points, rsf)

    background = np.log(bg_samples["rsf"].to_numpy(dtype=float) + 1e-12)
    used = np.log(pred.loc[pred["used"].astype(bool), pred_col].to_numpy(dtype=float) + 1e-12)
    background = background[np.isfinite(background)]
    used = used[np.isfinite(used)]

    rows = []
    for q0 in np.arange(0, 1.0 - window_fraction + 1e-12, step_fraction):
        q1 = q0 + window_fraction
        lo, hi = np.quantile(background, [q0, q1])
        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            continue

        in_bg = (background >= lo) & (background < hi)
        in_used = (used >= lo) & (used < hi)
        if in_bg.sum() == 0:
            continue

        pe = (in_used.sum() / used.size) / (in_bg.sum() / background.size)
        rows.append({"rsf_mid": 0.5 * (lo + hi), "pe": pe})

    chart = pd.DataFrame(rows)
    if len(chart) < 2:
        return np.nan, chart
    boyce, _ = spearmanr(chart["rsf_mid"], chart["pe"])
    return float(boyce), chart
