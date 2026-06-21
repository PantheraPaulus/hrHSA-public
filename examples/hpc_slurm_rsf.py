"""HPC / SLURM RSF workflow scaffold.

This example shows the intended scaling pattern. It is not meant to be run
unchanged: replace paths, SLURM parameters, and model specification with values
for the target cluster and study system.
"""

from pathlib import Path

import geopandas as gpd
import xarray as xr
from dask.distributed import Client

from hsa import FeatureSpec
from hsa.compute import (
    initialize_earth_engine_on_workers,
    make_slurm_cluster,
    open_raster_stack_zarr,
    persist_if_dask,
    sample_raster_stack_batched,
    suggest_xy_chunks,
    write_table_parquet,
)
from hsa.rsf import fit_rsf, predict_rsf_surface_multiscale
from hsa.sampling import sample_available_points


PROJECT = "your-ee-project"
WORKDIR = Path("/path/to/hpc/workdir")
ENV_ZARR = WORKDIR / "env_32733.zarr"
RELOCATIONS = WORKDIR / "relocations.gpkg"
DOMAIN = WORKDIR / "availability_domain.gpkg"
SAMPLED_PARQUET = WORKDIR / "sampled_points.parquet"


def main():
    cluster = make_slurm_cluster(
        queue="general",
        cores=4,
        processes=4,
        memory="32GB",
        walltime="04:00:00",
        local_directory="$TMPDIR",
        scale_jobs=10,
    )
    client = Client(cluster)
    initialize_earth_engine_on_workers(client, project=PROJECT)

    # Open and persist the environmental raster stack.
    env = open_raster_stack_zarr(ENV_ZARR, name="env", chunks="auto")
    env = env.chunk(suggest_xy_chunks(env, target_chunk_mb=256))
    env = persist_if_dask(env, client=client)

    # If the Zarr stack already contains multiscale variables as separate stores,
    # open them into this dict. Here we use one scale as a placeholder.
    env_by_scale: dict[str, xr.DataArray] = {"30m": env}

    reloc = gpd.read_file(RELOCATIONS)
    domain = gpd.read_file(DOMAIN)
    samples = sample_available_points(domain, n=len(reloc) * 100, used=reloc, seed=42)

    sampled = sample_raster_stack_batched(
        samples,
        env,
        target_batch_mb=128,
    )
    write_table_parquet(sampled, SAMPLED_PARQUET)

    spec = FeatureSpec(linear=["ndvi_mean_30m"], add_const=True)
    model, scaler, spec, meta = fit_rsf(sampled, spec)

    rsf = predict_rsf_surface_multiscale(
        env_by_scale,
        target_scale="30m",
        model=model,
        scaler=scaler,
        spec=spec,
        meta=meta,
    )

    # Example export. Consider writing to Zarr or Cloud Optimized GeoTIFF depending
    # on cluster storage and downstream GIS needs.
    rsf.to_dataset(name="rsf").to_zarr(WORKDIR / "rsf_30m.zarr", mode="w")
    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
