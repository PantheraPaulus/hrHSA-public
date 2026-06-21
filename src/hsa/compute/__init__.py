"""Compute backends for local and HPC-scale workflows."""

from hsa.compute.dask import (
    configure_dask_memory,
    get_or_create_client,
    initialize_earth_engine_on_workers,
    make_local_dask_client,
    make_slurm_cluster,
    shutdown_default_client,
)

__all__ = [
    "configure_dask_memory",
    "get_or_create_client",
    "initialize_earth_engine_on_workers",
    "make_local_dask_client",
    "make_slurm_cluster",
    "shutdown_default_client",
]
