from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr


def get_availability_domain(
    used: gpd.GeoDataFrame,
    *,
    estimator: str = "mcp",
    quantile: float = 0.95,
) -> gpd.GeoDataFrame:
    """Create an availability domain from used locations.

    Currently implements a centroid-trimmed MCP. This is intentionally simple;
    project-specific domains such as reserve boundaries, home ranges, or buffers
    can be passed directly to :func:`sample_available_points` instead.
    """

    if estimator.lower() != "mcp":
        raise ValueError(f"Unsupported estimator: {estimator!r}. Currently only 'mcp' is implemented.")
    if used.crs is None:
        raise ValueError("used.crs is None; set a CRS before defining availability.")

    original_crs = used.crs
    used_metric = used if not original_crs.is_geographic else used.to_crs(used.estimate_utm_crs())

    centroid = used_metric.geometry.union_all().centroid
    tmp = used_metric.copy()
    tmp["distance_to_centroid"] = tmp.geometry.distance(centroid)
    keep = tmp["distance_to_centroid"] <= tmp["distance_to_centroid"].quantile(quantile)
    hull = tmp.loc[keep].geometry.union_all().convex_hull

    domain = gpd.GeoDataFrame(geometry=[hull], crs=used_metric.crs)
    return domain.to_crs(original_crs)


def sample_available_points(
    domain: gpd.GeoDataFrame,
    n: int,
    used: gpd.GeoDataFrame | None = None,
    *,
    seed: int = 42,
    timestamp_col: str = "Timestamp",
) -> gpd.GeoDataFrame:
    """Sample available points and optionally append used points."""

    available = gpd.GeoDataFrame(
        geometry=domain.sample_points(n, rng=seed).explode(index_parts=True),
        crs=domain.crs,
    ).reset_index(drop=True)
    available[timestamp_col] = None
    available["used"] = False

    if used is None:
        return available

    used_min = used[[timestamp_col, "geometry"]].copy()
    used_min["used"] = True

    samples = pd.concat([used_min, available], ignore_index=True)
    return gpd.GeoDataFrame(samples, geometry="geometry", crs=domain.crs)


def infer_resolution(env: xr.DataArray) -> tuple[float, float]:
    """Infer absolute x/y resolution from a regular xarray grid."""

    x = env["x"].values
    y = env["y"].values
    return float(np.abs(np.diff(x)).mean()), float(np.abs(np.diff(y)).mean())


def aggregate_env(env: xr.DataArray, target_resolution: float, *, reducer: str = "mean") -> xr.DataArray:
    """Aggregate an environmental stack to an integer multiple of its native resolution."""

    dx, dy = infer_resolution(env)
    fx = target_resolution / dx
    fy = target_resolution / dy

    if not float(fx).is_integer() or not float(fy).is_integer():
        raise ValueError(
            f"target_resolution={target_resolution} is not an integer multiple of "
            f"native resolution ({dx:.3f}, {dy:.3f})."
        )

    fx = int(round(fx))
    fy = int(round(fy))
    if fx == 1 and fy == 1:
        return env

    coarsened = env.coarsen(x=fx, y=fy, boundary="trim")
    reducers = {
        "mean": coarsened.mean,
        "median": coarsened.median,
        "max": coarsened.max,
        "min": coarsened.min,
    }
    if reducer not in reducers:
        raise ValueError(f"Unsupported reducer: {reducer!r}")

    out = reducers[reducer]()
    return out.transpose(*[d for d in ("band", "y", "x") if d in out.dims])


def sample_raster_stack(
    samples: gpd.GeoDataFrame,
    env: xr.DataArray,
    *,
    target_resolution: float | None = None,
    reducer: str = "mean",
    dtype: str = "float32",
) -> pd.DataFrame:
    """Sample an xarray raster stack at point locations using nearest-neighbour lookup."""

    if target_resolution is not None:
        env = aggregate_env(env, target_resolution=target_resolution, reducer=reducer)

    try:
        env_crs = env.rio.crs
    except Exception:
        env_crs = None
    if env_crs is not None and samples.crs is not None and samples.crs != env_crs:
        samples = samples.to_crs(env_crs)

    env = env.transpose("band", "y", "x")
    xs = xr.DataArray(samples.geometry.x.to_numpy(), dims="points", name="x")
    ys = xr.DataArray(samples.geometry.y.to_numpy(), dims="points", name="y")
    sampled = env.sel(x=xs, y=ys, method="nearest").transpose("band", "points")

    arr = sampled.data
    arr = arr.compute() if hasattr(arr, "compute") else np.asarray(arr)
    arr = np.asarray(arr, dtype=dtype)

    bands = [str(band) for band in sampled["band"].values]
    out = pd.DataFrame(arr.T, columns=bands, index=samples.index)
    out["x"] = xs.values
    out["y"] = ys.values

    for column in ("used", "Timestamp"):
        if column in samples.columns:
            out[column] = samples[column].to_numpy()

    return out


def sample_raster_stack_multiscale(
    samples: gpd.GeoDataFrame,
    env: xr.DataArray,
    *,
    target_resolutions: list[float] | tuple[float, ...],
    reducer: str = "mean",
    dtype: str = "float32",
) -> tuple[pd.DataFrame, dict[str, xr.DataArray]]:
    """Sample the same stack at multiple spatial resolutions.

    Returns the sampled dataframe and the aggregated stacks used for prediction.
    Column names are suffixed with ``_30m``, ``_90m`` and so forth.
    """

    frames: list[pd.DataFrame] = []
    env_dict: dict[str, xr.DataArray] = {}

    for resolution in target_resolutions:
        suffix = f"{int(resolution)}m" if float(resolution).is_integer() else f"{resolution}m"
        env_res = aggregate_env(env, target_resolution=float(resolution), reducer=reducer)
        env_dict[suffix] = env_res
        sampled = sample_raster_stack(samples, env_res, dtype=dtype)
        value_columns = [c for c in sampled.columns if c not in {"x", "y", "used", "Timestamp"}]
        sampled = sampled.rename(columns={c: f"{c}_{suffix}" for c in value_columns})
        frames.append(sampled.drop(columns=[c for c in ("x", "y", "used", "Timestamp") if c in sampled.columns]))

    out = pd.concat(frames, axis=1)
    out["x"] = samples.geometry.x.to_numpy()
    out["y"] = samples.geometry.y.to_numpy()
    for column in ("used", "Timestamp"):
        if column in samples.columns:
            out[column] = samples[column].to_numpy()

    return out, env_dict
