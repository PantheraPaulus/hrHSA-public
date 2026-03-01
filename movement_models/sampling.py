from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

def _get_availability_domain(used: gpd.GeoDataFrame, estimator: str = "MCP", thres: float = 0.95) -> gpd.GeoDataFrame:
    if estimator != "MCP":
        raise ValueError(f"Unsupported estimator: {estimator}")
    orig_crs = used.crs

    if orig_crs is None:
        raise ValueError("used.crs is None; set a CRS before calling _get_availability_domain")

    used_m = used
    if orig_crs.is_geographic:
        used_m = used.to_crs(used.estimate_utm_crs())

    centroid = used_m.geometry.unary_union.centroid
    tmp = used_m.copy()
    tmp["dist_to_centroid"] = tmp.geometry.distance(centroid)

    points_within = tmp.loc[tmp["dist_to_centroid"] <= tmp["dist_to_centroid"].quantile(thres)]
    hull = points_within.geometry.unary_union.convex_hull

    domain_m = gpd.GeoDataFrame(geometry=[hull], crs=used_m.crs)

    return domain_m.to_crs(orig_crs)


def _get_sampling_points(domain, n, df = None, seed: int = 42) -> pd.DataFrame:

    available = gpd.GeoDataFrame(geometry = domain.sample_points(n, rng = seed).explode(), crs = domain.crs)
    
    if df is not None:
        samples = df[["Timestamp", "geometry"]].copy()
        samples["used"] = True
        samples = pd.concat([samples, available], ignore_index=True)
        samples["used"] = samples["used"].fillna(False)
        return gpd.GeoDataFrame(samples, geometry="geometry", crs=domain.crs)

    available["Timestamp"] = None
    available["used"] = False
    return available


def _choose_chunk_size_points(env, *, client=None, frac_of_worker_mem=0.10, per_worker_budget_mb=None, k=10,
    min_points=1_000, max_points=200_000):

    if client is not None:
        info = client.scheduler_info()["workers"]
        limits_mb = [w["memory_limit"] / 1024**2 for w in info.values()]
        base_budget_mb = min(limits_mb) * frac_of_worker_mem
        per_worker_budget_mb = int(max(8, min(512, base_budget_mb)))
    else:
        if per_worker_budget_mb is None:
            per_worker_budget_mb = 16  # conservative fallback
        per_worker_budget_mb = int(per_worker_budget_mb)

    if hasattr(env, "dims") and "band" in env.dims:
        n_bands = int(env.sizes["band"])
        dtype = env.dtype
    else:
        n_bands = len(env.data_vars)
        dtype = next(iter(env.data_vars.values())).dtype

    bytes_per_value = np.dtype(dtype).itemsize
    budget_bytes = per_worker_budget_mb * 1024 * 1024

    bytes_per_point = n_bands * bytes_per_value
    n_points = budget_bytes // (k * bytes_per_point)

    return int(min(max_points, max(min_points, n_points)))

def _sample_env_layer(
    samples: gpd.GeoDataFrame,
    env: xr.DataArray,
    chunk_size_points="auto",
    *,
    client=None,
    per_worker_budget_mb=16,
    k=10,
    min_points=1_000,
    max_points=200_000,
):
    try:
        env_crs = env.rio.crs
    except Exception:
        env_crs = None

    if env_crs is None:
        raise ValueError("env has no rio CRS; set it with env = env.rio.write_crs('EPSG:...')")

    if samples.crs is None:
        raise ValueError("samples.crs is None; set a CRS on your GeoDataFrame before sampling")

    if samples.crs != env_crs:
        samples = samples.to_crs(env_crs)

    if chunk_size_points == "auto" or chunk_size_points is None:
        chunk_size_points = _choose_chunk_size_points(
            env,
            client=client,
            per_worker_budget_mb=per_worker_budget_mb,
            k=k,
            min_points=min_points,
            max_points=max_points,
        )
    else:
        chunk_size_points = int(chunk_size_points)

    xs = xr.DataArray(samples.geometry.x.to_numpy(), dims="points", name="x")
    ys = xr.DataArray(samples.geometry.y.to_numpy(), dims="points", name="y")

    sampled = env.sel(x=xs, y=ys, method="nearest").chunk({"points": chunk_size_points})

    arr = sampled.data
    arr = arr.compute() if hasattr(arr, "compute") else np.asarray(arr)
    arr = np.asarray(arr, dtype="float32")

    bands = sampled["band"].to_numpy().tolist()
    df = pd.DataFrame(arr.T, columns=bands)
    df["x"] = xs.to_numpy()
    df["y"] = ys.to_numpy()
    df["used"] = samples["used"].to_numpy()
    df["Timestamp"] = samples["Timestamp"].to_numpy()
    return df