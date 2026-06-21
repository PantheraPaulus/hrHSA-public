# HPC workflows with HSA

HSA is intended to scale from notebooks to workstation-scale Dask and SLURM-backed Dask clusters.

Use `hsa.compute.make_local_dask_client` for local notebook or workstation work.

Use `hsa.compute.make_slurm_cluster` for SLURM-backed Dask clusters via `dask-jobqueue`.

Use `hsa.compute.initialize_earth_engine_on_workers` when Earth Engine calls happen inside Dask tasks, because Earth Engine initialization is process-local.

For large raster stacks, use Zarr through `open_raster_stack_zarr` and `write_raster_stack_zarr`.

For large sampled point tables, use Parquet through `write_table_parquet` and `read_table_parquet`.

For large point sampling operations, use `sample_raster_stack_batched` so point extraction is split into memory-safe batches.

See `examples/hpc_slurm_rsf.py` for a SLURM-oriented workflow scaffold.
