from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import dataclass

from movement_models.types import FeatureSpec, _FoldSplit
from movement_models.sampling import _get_availability_domain, _get_sampling_points, _sample_env_layer
from movement_models.rsf.model import fit_rsf, predict_rsf_points
from movement_models.rsf.surface import get_rsf_surface
from movement_models.rsf.eval import fixed_width_Boyce

def _assign_week_folds(reloc: gpd.GeoDataFrame, k: int = 5, seed: int = 42) -> gpd.GeoDataFrame:
    g = reloc.copy()
    iso = g["Timestamp"].dt.isocalendar() 
    g["iso_year"] = iso["year"].astype(int)
    g["iso_week"] = iso["week"].astype(int)
    g["year_week"] = g["iso_year"].astype(str) + "-W" + g["iso_week"].astype(str).str.zfill(2)
    
    weeks = g["year_week"].dropna().unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(weeks)

    week_to_fold = {w: (i % k) for i, w in enumerate(weeks)}
    g["fold"] = g["year_week"].map(week_to_fold).astype(int)
    return g

def _thin_by_time(reloc: gpd.GeoDataFrame, min_dt: str = "12H") -> gpd.GeoDataFrame:
    g = reloc.copy()
    g = g.sort_values("Timestamp").reset_index(drop=True)

    min_dt = pd.Timedelta(min_dt)

    keep = np.zeros(len(g), dtype=bool)
    last_t = None
    for i, t in enumerate(g["Timestamp"].values):
        t = pd.Timestamp(t)
        if last_t is None or (t - last_t) >= min_dt:
            keep[i] = True
            last_t = t

    return g.loc[keep].reset_index(drop=True)

def _get_fold_split(
    reloc_folds: gpd.GeoDataFrame,
    fold_id: int,
    thin_train_dt: str = "12H",
    thin_test_dt: Optional[str] = None
    ) -> _FoldSplit:
    train = reloc_folds.loc[reloc_folds["fold"] != fold_id].copy()
    test  = reloc_folds.loc[reloc_folds["fold"] == fold_id].copy()

    train_thin = _thin_by_time(train, min_dt=thin_train_dt)

    if thin_test_dt is None:
        test_thin = test.copy()
    else:
        test_thin = _thin_by_time(test, min_dt=thin_test_dt)

    return _FoldSplit(train=train, test=test, train_thin=train_thin, test_thin=test_thin)

def cv_model(
    obs: pd.DataFrame,
    env: xarray.DataArray,
    k_folds: int = 5,
    subset_predictors = ["ndvi"],
    sampling_factor_train: int = 10,
    n_bg_boyce: int = 100_000,
    n_bins: int = 20,
    seed: int = 42,
    verbose: bool = False,
):
    domain = _get_availability_domain(obs)
    cov = env.sel(band=subset_predictors)
    obs_folds = _assign_week_folds(obs, k=k_folds, seed=seed)
    rows = []

    for k in range(k_folds):
        if verbose: print(f"Working on fold: {k}")

        if verbose: print("Splitting the dataset...")
        subset = _get_fold_split(obs_folds, k)
        train = subset.train_thin
        test = subset.test

        if verbose: print("I'm getting the sample points - used + available combined in one dataframe.")
        train_samples = _get_sampling_points(domain, len(train) * sampling_factor_train, df=train, seed=seed + k)

        if verbose: print("I'm extracting the values of the covariate at the locations of the samples.")
        sampled = _sample_env_layer(train_samples, cov)

        if verbose: print("I'm fitting the model.")
        spec = FeatureSpec(linear=subset_predictors, add_const=True)
        m, scaler, _ = fit_rsf(sampled, spec)

        if verbose: print("I'm calculating the rsf surface.")
        rsf = get_rsf_surface(cov, m, scaler, spec, crs=obs.crs)
        
        if verbose: print("Testing on the held out piece.")
        test = test.copy()
        test["used"] = True

        if verbose: print("Sampling the covariate for used locations in the test set.")
        test_sampled = _sample_env_layer(test, cov)

        if verbose: print("Predicting the rsf for these locations...")
        test_sampled = test_sampled.replace([np.inf, -np.inf], np.nan).dropna(subset=subset_predictors)
        test_evaluated = predict_rsf_points(test_sampled, m, scaler, spec)

        if verbose: print("Calculating boyce...")
        boyce, _ = fixed_width_Boyce(
            pred=test_evaluated,
            rsf=rsf,
            domain=domain,
            n_bg_points=n_bg_boyce,
            n_bins=n_bins,
            seed=seed + 200 * k,
        )

        rows.append({"k": k, "boyce": boyce})
        if verbose: print(boyce)

    return pd.DataFrame(rows)