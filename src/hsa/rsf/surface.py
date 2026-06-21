from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler

from hsa.types import FeatureSpec


def split_multiscale_name(name: str) -> tuple[str, str | None]:
    """Split names such as ``ndvi_mean_90m`` into base variable and scale suffix."""

    base, sep, suffix = name.rpartition("_")
    if sep and suffix.endswith("m"):
        return base, suffix
    return name, None


def predict_rsf_surface(
    env: xr.DataArray,
    model,
    scaler: StandardScaler,
    spec: FeatureSpec,
    meta: dict[str, Any],
) -> xr.DataArray:
    """Project a fitted RSF model onto a raster stack with matching band names."""

    predictors = [*spec.linear, *spec.categorical]
    if not predictors:
        raise ValueError("FeatureSpec must contain at least one linear or categorical predictor.")

    coef = model.params
    eta = xr.zeros_like(env.sel(band=predictors[0])) + float(coef.get("const", 0.0))
    valid_mask = xr.ones_like(eta, dtype=bool)

    z: dict[str, xr.DataArray] = {}
    for index, variable in enumerate(spec.linear):
        layer = env.sel(band=variable)
        z[variable] = (layer - float(scaler.mean_[index])) / float(scaler.scale_[index])
        if variable in coef.index:
            eta = eta + z[variable] * float(coef[variable])

    for variable in spec.quadratic:
        name = f"{variable}__sq"
        if name in coef.index:
            eta = eta + (z[variable] ** 2) * float(coef[name])

    for left, right in spec.interactions:
        name = f"{left}__x__{right}"
        if name in coef.index:
            eta = eta + (z[left] * z[right]) * float(coef[name])

    for variable in spec.categorical:
        layer = env.sel(band=variable)
        info = meta["categorical"][variable]
        keep_levels = info["keep_levels"]
        valid_mask = valid_mask & layer.isin(keep_levels)

        for level in keep_levels:
            if level == info["reference"]:
                continue
            name = f"{variable}_{level}"
            if name in coef.index:
                eta = eta + xr.where(layer == level, float(coef[name]), 0.0)

    rsf = np.exp(eta.where(valid_mask)).rename("rsf").expand_dims(band=["rsf"])
    try:
        crs = env.rio.crs
        if crs is not None:
            rsf = rsf.rio.write_crs(crs)
    except Exception:
        pass
    return rsf


def build_prediction_stack(
    env_by_scale: dict[str, xr.DataArray],
    predictors: list[str],
    *,
    target_scale: str,
) -> xr.DataArray:
    """Build a prediction stack whose bands exactly match multiscale predictor names.

    This starter implementation assumes all requested scales already exist in
    ``env_by_scale`` and share a compatible grid. Reprojection/resampling logic
    from the pangolin notebook should be added here when needed.
    """

    if target_scale not in env_by_scale:
        raise KeyError(f"target_scale={target_scale!r} not found. Available: {list(env_by_scale)}")

    layers: list[xr.DataArray] = []
    for predictor in predictors:
        base, scale = split_multiscale_name(predictor)
        if scale is None:
            raise ValueError(f"Predictor {predictor!r} has no scale suffix, e.g. '_30m'.")
        if scale not in env_by_scale:
            raise KeyError(f"Scale {scale!r} required by {predictor!r} not found in env_by_scale.")

        layer = env_by_scale[scale].sel(band=base).rename(predictor).expand_dims(band=[predictor])
        layers.append(layer)

    return xr.concat(layers, dim="band").transpose("band", "y", "x")


def predict_rsf_surface_multiscale(
    env_by_scale: dict[str, xr.DataArray],
    *,
    target_scale: str,
    model,
    scaler: StandardScaler,
    spec: FeatureSpec,
    meta: dict[str, Any],
) -> xr.DataArray:
    """Project a fitted multiscale RSF model onto raster stacks."""

    predictors = [*spec.linear, *spec.categorical]
    stack = build_prediction_stack(env_by_scale, predictors, target_scale=target_scale)
    return predict_rsf_surface(stack, model, scaler, spec, meta)
