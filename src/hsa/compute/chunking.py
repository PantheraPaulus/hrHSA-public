from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hsa.sampling import sample_raster_stack


def estimate_raster_bytes(
    env: xr.DataArray,
    *,
    dtype: str | np.dtype | None = None,
) -> int:
    """Estimate the dense in-memory size of an xarray raster stack in bytes."""

    itemsize = np.dtype(dtype or env.dtype).itemsize
    n_cells = 1
    for dim in env.dims:
        n_cells *= int(env.sizes[dim])
    return int(n_cells * itemsize)


def suggest_xy_chunks(
    env: xr.DataArray,
    *,
    target_chunk_mb: int = 128,
    min_xy: int = 256,
    max_xy: int = 4096,
    dtype: str | np.dtype | None = None,
) -> dict[str, int]:
    """Suggest square-ish x/y chunks for a ``band, y, x`` raster stack.

    The heuristic keeps all bands in one chunk and chooses x/y chunks so one
    block is approximately ``target_chunk_mb``. This is a sensible default for
    many RSF prediction and raster-sampling workflows, but users should still
    benchmark on real HPC storage.
    """

    if "x" not in env.dims or "y" not in env.dims:
        raise ValueError("suggest_xy_chunks expects dimensions named 'x' and 'y'.")

    n_bands = int(env.sizes.get("band", 1))
    itemsize = np.dtype(dtype or env.dtype).itemsize
    target_bytes = target_chunk_mb * 1024**2
    cells_per_xy_chunk = max(1, target_bytes // max(1, n_bands * itemsize))
    side = int(math.sqrt(cells_per_xy_chunk))
    side = max(min_xy, min(max_xy, side))

    return {
        "band": -1 if "band" in env.dims else 1,
        "y": min(side, int(env.sizes["y"])),
        "x": min(side, int(env.sizes["x"])),
    }


def rechunk_raster(
    env: xr.DataArray,
    *,
    chunks: dict[str, int] | None = None,
    target_chunk_mb: int = 128,
) -> xr.DataArray:
    """Return a Dask-chunked raster stack."""

    if chunks is None:
        chunks = suggest_xy_chunks(env, target_chunk_mb=target_chunk_mb)
    return env.chunk(chunks)


def suggest_point_batch_size(
    env: xr.DataArray,
    *,
    target_batch_mb: int = 64,
    min_points: int = 1_000,
    max_points: int = 250_000,
    safety_factor: float = 4.0,
) -> int:
    """Suggest a point batch size for raster-stack sampling.

    ``safety_factor`` accounts for dataframe overhead and intermediate arrays.
    """

    n_bands = int(env.sizes.get("band", 1))
    bytes_per_value = np.dtype(env.dtype).itemsize
    bytes_per_point = max(1, int(n_bands * bytes_per_value * safety_factor))
    target_bytes = target_batch_mb * 1024**2
    n_points = target_bytes // bytes_per_point
    return int(min(max_points, max(min_points, n_points)))


def iter_point_batches(
    samples: gpd.GeoDataFrame,
    *,
    batch_size: int,
) -> Iterable[gpd.GeoDataFrame]:
    """Yield slices of a GeoDataFrame for batch processing."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(samples), batch_size):
        yield samples.iloc[start : start + batch_size]


def sample_raster_stack_batched(
    samples: gpd.GeoDataFrame,
    env: xr.DataArray,
    *,
    batch_size: int | None = None,
    target_batch_mb: int = 64,
    **sample_kwargs: Any,
) -> pd.DataFrame:
    """Sample a raster stack in point batches and concatenate the result."""

    if batch_size is None:
        batch_size = suggest_point_batch_size(env, target_batch_mb=target_batch_mb)

    frames = [sample_raster_stack(batch, env, **sample_kwargs) for batch in iter_point_batches(samples, batch_size=batch_size)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def persist_if_dask(obj, *, client=None):
    """Persist a Dask-backed object when a client is available."""

    if not hasattr(obj, "persist"):
        return obj
    if client is None:
        return obj.persist()
    return client.persist(obj)
