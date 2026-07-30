# Implementation

## Design Principles

**hrHSA** is implemented as a scientific software package rather than as a collection of study-specific scripts. Its architecture is guided by four principles:

1. **Reproducibility** — every transformation between the original data and the final inference should be explicit and recoverable.
2. **Modularity** — data preparation, feature construction, statistical inference, validation and prediction should remain separable.
3. **Efficiency** — computational resources should be used deliberately, with particular attention to memory access, spatial chunking and repeated raster operations.
4. **Scalability** — the same analytical workflow should run on a notebook, a multicore workstation or a SLURM-managed computing cluster.

The resulting workflow separates scientific decisions from computational execution:

```text
telemetry and environmental data
        ↓
spatial standardisation
        ↓
definition and sampling of availability
        ↓
environmental feature extraction
        ↓
model-matrix construction
        ↓
statistical inference
        ↓
validation and diagnostics
        ↓
spatial prediction or movement simulation
```

Each stage exposes a limited interface and may be inspected, tested or replaced independently.

## Package Architecture

The source code follows the conventional Python `src` layout and is divided according to analytical responsibility:

```text
src/hsa/
├── sampling.py          # availability domains and raster sampling
├── features.py          # reproducible construction of model matrices
├── types.py             # shared specifications and data containers
├── rsf/                 # fitting, prediction, validation and cross-validation
├── movement/            # trajectories, steps, angles and movement kernels
├── remote_sensing/      # optional Earth Observation interfaces
├── compute/             # chunking, scalable I/O and distributed execution
└── simulation/          # forward movement simulation
```

The package itself remains independent of particular species, reserves, local file systems or Earth Engine projects. Study-specific assumptions belong in configuration files, examples or downstream analyses. This distinction allows empirical workflows to remain fully documented without embedding them in the reusable statistical core.

Optional functionality is separated into dependency groups. A minimal installation provides the numerical and geospatial foundations of the package, while Earth Engine, distributed computation and HPC support may be installed only where required:

```bash
pip install -e .
pip install -e ".[earthengine]"
pip install -e ".[hpc,io]"
```

Optional libraries are imported lazily inside the functions that require them. Importing **hrHSA** therefore neither initialises external services nor imposes a distributed-computing environment upon smaller analyses.

## Reproducible Model Construction

A fitted model is defined not only by its coefficients, but also by the transformations applied before estimation. **hrHSA** represents these transformations through a declarative `FeatureSpec`:

```python
from hsa import FeatureSpec

spec = FeatureSpec(
    linear=["ndvi", "distance_to_water"],
    quadratic=["distance_to_water"],
    interactions=[("ndvi", "distance_to_water")],
    categorical=["landcover"],
    add_const=True,
)
```

Continuous predictors are standardised during fitting. Quadratic and interaction terms are then constructed on the standardised scale. Categorical levels, reference categories and final column order are stored as model metadata.

```python
model, scaler, spec, meta = fit_rsf(samples, spec)
```

The model, fitted scaler, feature specification and metadata jointly constitute the fitted analysis. Passing the same objects to validation and prediction ensures that raster surfaces and held-out observations receive precisely the transformations used during model fitting.

Randomised operations accept explicit seeds. Coordinate reference systems are checked before spatial operations, and functions involving distances require projected coordinates with meaningful linear units. Invalid inputs are rejected early rather than being allowed to produce numerically plausible but ecologically incorrect results.

## Geospatial Data and Scalable I/O

Telemetry locations are represented as `GeoDataFrame` objects, while environmental fields are stored as labelled `xarray` arrays. A conventional environmental stack has the dimensions

```text
band × y × x
```

and retains its coordinates, projection, spatial resolution and band names throughout the workflow.

Large raster stacks may be stored as **Zarr**, preserving multidimensional structure and supporting parallel access:

```python
from hsa.compute import write_raster_stack_zarr

write_raster_stack_zarr(
    env,
    "environment.zarr",
    target_chunk_mb=128,
)
```

Large sampled point tables may be stored as **Parquet**:

```python
from hsa.compute import write_table_parquet

write_table_parquet(samples, "samples.parquet")
```

These intermediate products separate expensive environmental processing from statistical fitting. An analysis can therefore be repeated with a different model specification without downloading, transforming and sampling the complete environmental dataset again.

## Chunking and Memory Management

High-resolution analyses commonly exceed available memory long before they exceed available computing power. **hrHSA** therefore partitions raster and point operations into manageable blocks.

For a raster containing $B$ bands, spatial chunk dimensions $n_x$ and $n_y$, and elements requiring $d$ bytes, the approximate memory occupied by one chunk is

$$
M_{\mathrm{chunk}}
=
B n_x n_y d.
$$

The package provides heuristics for selecting approximately square spatial chunks under a specified memory target. Point extraction is similarly divided into batches whose size depends upon the number of environmental bands and their numeric precision.

These heuristics provide conservative starting values rather than universal optima. Suitable chunk sizes depend upon the storage system, network throughput, worker memory and downstream operation. Computationally demanding analyses should therefore be benchmarked under realistic conditions.

## Distributed and HPC Execution

Distributed execution is provided through **Dask**. The statistical and geospatial functions remain independent of the scheduler, allowing the same analysis to operate under different computational backends.

A local client may be created for notebook or workstation analyses:

```python
from hsa.compute import make_local_dask_client

client = make_local_dask_client(
    n_workers=8,
    threads_per_worker=1,
    local_directory="/tmp/hsa-dask",
)
```

On a SLURM-managed cluster, the computational backend may be replaced without altering the scientific workflow:

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

Worker-memory thresholds are configured explicitly so that intermediate data may spill to local storage before workers exhaust their available memory. Large raster stacks can be persisted across workers when they are accessed repeatedly, and temporary data can be directed to node-local scratch storage.

External services require explicit worker initialisation. Google Earth Engine, for example, maintains process-local state and must be initialised separately on each distributed worker. **hrHSA** exposes this operation directly rather than relying upon hidden state inherited from the client process.

## Statistical Validation

Validation is part of the implementation rather than an optional final step. The package supports temporally blocked cross-validation and leave-one-individual-out validation, together with Boyce-index and calibration diagnostics.

Temporal blocks reduce leakage between neighbouring relocations, while leave-one-individual-out validation tests whether a model transfers to animals not represented during fitting. In the latter case, training availability is generated separately within each training animal's domain, and the held-out animal is evaluated relative to its own availability.

Validation returns fold-specific coefficients, predictions, surfaces, Boyce diagnostics, calibration tables and sample sizes. Retaining these outputs permits individual failures and unstable coefficients to be examined instead of concealing them within a single pooled score.

Movement modules provide trajectory preparation, step-length and turning-angle calculation, and estimation of empirical movement distributions by individual. These components supply the movement information required by subsequent SSF and iSSF workflows.

## Bayesian Inference and Simulation Validation

The package architecture is intended to support hierarchical Bayesian inference and large stochastic simulation ensembles. Feature construction, environmental sampling and validation are kept independent of the fitting backend so that likelihood-based estimation may be supplemented by PyMC- or JAX-based models without rebuilding the complete geospatial workflow.

Gradient-based algorithms such as Hamiltonian Monte Carlo and the No-U-Turn Sampler are particularly suitable for the high-dimensional and correlated parameter spaces produced by hierarchical habitat-selection models (Hoffman & Gelman, 2014; Betancourt, 2017). Independent chains can be distributed across processes or nodes, while automatic differentiation and just-in-time compilation may accelerate repeated likelihood evaluation.

Forward simulations provide a second level of parallelism. Independent trajectories or posterior simulations may be distributed among workers, while the candidate steps within each trajectory can be evaluated through vectorised array operations.

The present package provides the geospatial, movement and distributed-computing foundations for these extensions. Fully integrated Bayesian fitting and habitat-biased forward simulation remain developing components and are kept distinct from the stable RSF implementation.

## Documentation and Quality Assurance

The documentation is built with **Sphinx** and **MyST Markdown**, providing mathematical notation, automatically generated API references and versioned narrative documentation. Package configuration is maintained in `pyproject.toml`, with explicit core, optional and development dependencies.

The current test suite verifies the public import surface and provides a foundation for continued quality assurance. Release hardening should additionally include:

- unit tests for spatial and statistical functions;
- regression tests for stable numerical outputs;
- synthetic parameter-recovery experiments;
- end-to-end workflow tests;
- and benchmarks of runtime, memory use and scaling efficiency.

Scientific verification must accompany software verification. A function may execute correctly while implementing an inappropriate availability domain, temporal scale or ecological model.

## Reproducibility

A reproducible **hrHSA** analysis should preserve:

- package and dependency versions;
- coordinate reference systems and raster grains;
- the definition of availability;
- random seeds and sampling ratios;
- the feature specification and preprocessing metadata;
- fitted models and diagnostic outputs;
- cross-validation partitions;
- and the environmental products, or immutable references, from which predictors were derived.

Reproducibility is therefore treated as part of the analytical design rather than as an administrative addition made after the analysis is complete.

## Summary

**hrHSA** combines an accessible Python interface with a modular architecture for high-resolution telemetry and environmental data. Declarative model specifications, labelled geospatial arrays, memory-aware batching, chunked storage, structured validation and interchangeable local or SLURM-backed execution allow the same scientific workflow to operate across different computational scales.

The purpose of this architecture is not parallelism for its own sake. It is to ensure that increasing spatial resolution, study extent and model complexity does not require a corresponding loss of transparency, reproducibility or ecological rigour.

## Principal References

Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. *arXiv*, 1701.02434.

Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn Sampler: adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, **15**, 1593–1623.

Hoyer, S. & Hamman, J. J. (2017). xarray: N-D labelled arrays and datasets in Python. *Journal of Open Research Software*, **5**, 10.

Rocklin, M. (2015). Dask: parallel computation with blocked algorithms and task scheduling. *Proceedings of the 14th Python in Science Conference*, 130–136.

Sandve, G. K., Nekrutenko, A., Taylor, J. & Hovig, E. (2013). Ten simple rules for reproducible computational research. *PLoS Computational Biology*, **9**, e1003285.

Wilkinson, M. D. et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data*, **3**, 160018.

Wilson, G. et al. (2017). Good enough practices in scientific computing. *PLoS Computational Biology*, **13**, e1005510.