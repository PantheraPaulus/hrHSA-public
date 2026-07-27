from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
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

def plot_boyce_values(
    summary: pd.DataFrame,
    *,
    id_col: str = "heldout_ID",
    boyce_col: str = "boyce",
    sort: bool = True,
    ax=None,
    figsize: tuple[float, float] = (9, 4),
    title: str = "Boyce validation by held-out individual",
):
    """Plot Boyce index values by fold or held-out individual."""

    df = summary.copy()
    df = df.loc[df[boyce_col].notna()].copy()

    if sort:
        df = df.sort_values(boyce_col)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(df))

    ax.bar(x, df[boyce_col].to_numpy(dtype=float))
    ax.axhline(0, linewidth=1)
    ax.axhline(df[boyce_col].mean(), linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(df[id_col].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Boyce index")
    ax.set_xlabel(id_col)
    ax.set_title(title)

    ax.text(
        0.99,
        0.02,
        f"mean = {df[boyce_col].mean():.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )

    fig.tight_layout()
    return fig, ax

def plot_boyce_curves(
    boyce_bins: pd.DataFrame,
    *,
    id_col: str = "heldout_ID",
    x_col: str = "rsf_mid",
    y_col: str = "pe",
    show_points: bool = True,
    show_reference: bool = True,
    log_x: bool = False,
    ax=None,
    figsize: tuple[float, float] = (7, 5),
    title: str = "Boyce calibration curves",
    alpha: float = 0.45,
    legend: bool | str = "auto",
):
    """Plot Boyce calibration curves.

    Expects output from ``boyce_quantile_bins`` or ``boyce_sliding_window``.
    If multiple held-out individuals are present, one curve is drawn per ID.
    """

    df = boyce_bins.copy()

    missing = [col for col in (x_col, y_col) if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Available columns are: {list(df.columns)}"
        )

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])

    if df.empty:
        raise ValueError("No finite Boyce-bin values available for plotting.")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if id_col in df.columns:
        groups = df.groupby(id_col)
    else:
        groups = [(None, df)]

    for individual_id, g in groups:
        g = g.sort_values(x_col)

        label = str(individual_id) if individual_id is not None else None

        ax.plot(
            g[x_col].to_numpy(dtype=float),
            g[y_col].to_numpy(dtype=float),
            linewidth=1,
            alpha=alpha,
            label=label,
        )

        if show_points:
            ax.scatter(
                g[x_col].to_numpy(dtype=float),
                g[y_col].to_numpy(dtype=float),
                s=20,
                alpha=alpha,
            )

    if show_reference:
        ax.axhline(1, linestyle="--", linewidth=1)

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel("Log RSF score" if x_col == "rsf_mid" else x_col)
    ax.set_ylabel("Predicted/expected ratio" if y_col == "pe" else y_col)
    ax.set_title(title)

    if legend == "auto":
        legend = id_col in df.columns and df[id_col].nunique() <= 12

    if legend:
        ax.legend(title=id_col, bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    return fig, ax