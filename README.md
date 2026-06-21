<p align="center">
  <img src="logo/logo_rsf2.png" width="600">
</p>

# HSA

Python package skeleton for **Habitat / Resource Selection Analysis** and related movement-modelling workflows.

The package is intended to provide a standardized workflow to perform Resource Selection Analysis, optionally extended toward Step Selection Analysis and simulation on larger telemetry and environmental datasets.

## Current structure

```text
src/hsa/
├── sampling.py          # availability domains, available points, raster sampling
├── features.py          # design-matrix construction from FeatureSpec
├── types.py             # shared dataclasses
├── rsf/                 # RSF fitting, prediction, validation, CV, selection
├── movement/            # steps, turn angles, movement kernels
├── remote_sensing/      # optional Earth Engine / predictor-stack helpers
├── compute/             # Dask utilities for local and HPC workflows
└── simulation/          # future SSF / mechanistic movement simulation
```

## Development install

```bash
conda env create -f environment.yml
conda activate hsa
pip install -e .
```

For a pip-based HPC install with the optional Dask/SLURM tools:

```bash
pip install -e ".[hpc,earthengine]"
```

## Minimal RSF pattern

```python
from hsa import FeatureSpec
from hsa.sampling import sample_available_points, sample_raster_stack
from hsa.rsf import fit_rsf, predict_rsf_surface

spec = FeatureSpec(linear=["ndvi_mean_30m"], add_const=True)

# samples = sample_available_points(domain, n=10_000, used=relocations)
# df = sample_raster_stack(samples, env)
# model, scaler, spec, meta = fit_rsf(df, spec)
# rsf = predict_rsf_surface(env, model, scaler, spec, meta)
```

## Dask / HPC pattern

Local workstation or notebook:

```python
from hsa.compute import make_local_dask_client

client = make_local_dask_client(n_workers=8, local_directory="/tmp/hsa-dask")
```

SLURM-backed cluster using `dask-jobqueue`:

```python
from dask.distributed import Client
from hsa.compute import make_slurm_cluster

cluster = make_slurm_cluster(
    queue="general",
    cores=4,
    processes=4,
    memory="32GB",
    walltime="04:00:00",
    scale_jobs=10,
)
client = Client(cluster)
```

Earth Engine must be initialized separately on each worker when EE calls happen inside Dask tasks:

```python
from hsa.compute import initialize_earth_engine_on_workers

initialize_earth_engine_on_workers(client, project="your-ee-project")
```

## Examples

Project-specific workflows should live in `examples/`, especially the Okonjima pangolin RSF workflow. The package itself should stay general and should not contain local file paths, Earth Engine project names, reserve-specific exclusions, or project-specific predictor choices.
