from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd

from hsa.types import FoldSplit, FeatureSpec
from hsa.sampling import get_availability_domain, sample_available_points, sample_raster_stack
from hsa.rsf.model import fit_rsf, predict_rsf_points
from hsa.rsf.surface import predict_rsf_surface
from hsa.rsf.validation import boyce_quantile_bins


def assign_week_folds(reloc: gpd.GeoDataFrame, *, k: int = 5, seed: int = 42) -> gpd.GeoDataFrame:
    """Assign whole ISO weeks to cross-validation folds."""

    g = reloc.copy()
    iso = g["Timestamp"].dt.isocalendar()
    g["year_week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    weeks = g["year_week"].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(weeks)
    g["fold"] = g["year_week"].map({week: i % k for i, week in enumerate(weeks)}).astype(int)
    return g


def thin_by_time(reloc: gpd.GeoDataFrame, *, min_dt: str = "12h") -> gpd.GeoDataFrame:
    """Thin records so retained fixes are at least ``min_dt`` apart."""

    g = reloc.sort_values("Timestamp").reset_index(drop=True).copy()
    min_delta = pd.Timedelta(min_dt)
    keep = np.zeros(len(g), dtype=bool)
    last_time = None
    for i, timestamp in enumerate(g["Timestamp"]):
        if last_time is None or pd.Timestamp(timestamp) - last_time >= min_delta:
            keep[i] = True
            last_time = pd.Timestamp(timestamp)
    return g.loc[keep].reset_index(drop=True)


def get_fold_split(
    reloc_folds: gpd.GeoDataFrame,
    fold_id: int,
    *,
    thin_train_dt: str = "12h",
    thin_test_dt: Optional[str] = None,
) -> FoldSplit:
    """Return train/test split for one fold, with optional temporal thinning."""

    train = reloc_folds.loc[reloc_folds["fold"] != fold_id].copy()
    test = reloc_folds.loc[reloc_folds["fold"] == fold_id].copy()
    train_thin = thin_by_time(train, min_dt=thin_train_dt)
    test_thin = test if thin_test_dt is None else thin_by_time(test, min_dt=thin_test_dt)
    return FoldSplit(train=train, test=test, train_thin=train_thin, test_thin=test_thin)


def cross_validate_rsf_temporal(
    reloc: gpd.GeoDataFrame,
    env,
    spec: FeatureSpec,
    *,
    k_folds: int = 5,
    sampling_factor_train: int = 10,
    n_background_boyce: int = 100_000,
    n_bins: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Temporal/week-blocked RSF validation.

    This is intentionally conservative and should later be complemented by
    leave-one-individual-out validation for multi-animal studies.
    """

    domain = get_availability_domain(reloc)
    reloc_folds = assign_week_folds(reloc, k=k_folds, seed=seed)
    rows = []

    for fold in range(k_folds):
        split = get_fold_split(reloc_folds, fold)
        train_samples = sample_available_points(
            domain,
            len(split.train_thin) * sampling_factor_train,
            used=split.train_thin,
            seed=seed + fold,
        )
        train_df = sample_raster_stack(train_samples, env)
        model, scaler, fitted_spec, meta = fit_rsf(train_df, spec)
        rsf = predict_rsf_surface(env, model, scaler, fitted_spec, meta)

        test = split.test.copy()
        test["used"] = True
        test_df = sample_raster_stack(test, env).replace([np.inf, -np.inf], np.nan)
        test_pred = predict_rsf_points(test_df.dropna(subset=spec.linear), model, scaler, fitted_spec, meta)

        boyce, _ = boyce_quantile_bins(
            test_pred,
            rsf,
            domain,
            n_background_points=n_background_boyce,
            n_bins=n_bins,
            seed=seed + 200 * fold,
        )
        rows.append({"fold": fold, "boyce": boyce, "n_train": len(split.train_thin), "n_test": len(split.test)})

    return pd.DataFrame(rows)
