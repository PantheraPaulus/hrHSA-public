"""Compute backends for local and HPC-scale workflows."""

from hsa.compute.chunking import estimate_raster_bytes, iter_point_batches, persist_if_dask, rechunk_raster, sample_raster_stack_batched, suggest_point_batch_size, suggest_xy_chunks
from hsa.compute.dask import configure_dask_memory, get_or_create_client, initialize_earth_engine_on_workers, make_local_dask_client, make_slurm_cluster, run_on_workers, shutdown_default_client
from hsa.compute.io import open_raster_stack_zarr, read_table_parquet, write_raster_stack_zarr, write_table_parquet

__all__ = [
    "configure_dask_memory",
    "estimate_raster_bytes",
    "get_or_create_client",
    "initialize_earth_engine_on_workers",
    "iter_point_batches",
    "make_local_dask_client",
    "make_slurm_cluster",
    "open_raster_stack_zarr",
    "persist_if_dask",
    "read_table_parquet",
    "rechunk_raster",
    "run_on_workers",
    "sample_raster_stack_batched",
    "shutdown_default_client",
    "suggest_point_batch_size",
    "suggest_xy_chunks",
    "write_raster_stack_zarr",
    "write_table_parquet",
]
