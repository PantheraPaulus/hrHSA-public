from __future__ import annotations

from pathlib import Path

import xarray as xr

from hsa.compute.chunking import rechunk_raster, suggest_xy_chunks


def write_raster_stack_zarr(
    env: xr.DataArray,
    path: str | Path,
    *,
    name: str = "env",
    mode: str = "w",
    chunks: dict[str, int] | None = None,
    target_chunk_mb: int = 128,
    consolidated: bool = True,
    compute: bool = True,
):
    """Write a raster stack to Zarr using HPC-friendly chunks.

    Returns the object returned by ``xarray.Dataset.to_zarr``. With
    ``compute=False`` this can be a delayed object suitable for explicit Dask
    execution.
    """

    if chunks is None:
        chunks = suggest_xy_chunks(env, target_chunk_mb=target_chunk_mb)
    env = rechunk_raster(env, chunks=chunks)
    ds = env.to_dataset(name=name)
    return ds.to_zarr(path, mode=mode, consolidated=consolidated, compute=compute)


def open_raster_stack_zarr(
    path: str | Path,
    *,
    name: str = "env",
    chunks: dict[str, int] | str | None = "auto",
    consolidated: bool | None = None,
) -> xr.DataArray:
    """Open a raster stack from Zarr."""

    ds = xr.open_zarr(path, chunks=chunks, consolidated=consolidated)
    if name not in ds:
        raise KeyError(f"Variable {name!r} not found in Zarr store. Available variables: {list(ds.data_vars)}")
    return ds[name]


def write_table_parquet(df, path: str | Path, *, index: bool = False, **kwargs) -> None:
    """Write a dataframe to Parquet.

    This helper keeps tabular intermediate output consistent across examples.
    It requires either ``pyarrow`` or ``fastparquet`` in the environment.
    """

    df.to_parquet(path, index=index, **kwargs)


def read_table_parquet(path: str | Path, **kwargs):
    """Read a dataframe from Parquet."""

    import pandas as pd

    return pd.read_parquet(path, **kwargs)
