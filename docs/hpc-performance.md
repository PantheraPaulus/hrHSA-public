# HPC execution and performance engineering

High-resolution habitat-selection analysis is often limited by **memory traffic,
storage access and repeated geospatial work** before it is limited by the number
of floating-point operations in the statistical model. hrHSA therefore treats HPC
execution as an algorithmic problem rather than as a request to add parallelism to
otherwise unchanged code.

This page describes the performance architecture introduced for KONWIHR-scale
workloads, how to use it, and how performance should be evaluated on a workstation
or SLURM cluster.

## Design principles

The accelerated implementation follows four rules.

1. **Reference and accelerated kernels remain separate.** The existing serial
   implementations remain the numerical reference. Accelerated kernels are tested
   against them rather than replacing them silently.
2. **Move data less.** Raster chunks should be read once per group of points,
   prediction should avoid unnecessary intermediate arrays, and expensive
   environmental extraction should be cached before repeated model fitting.
3. **Parallelize independent scientific units.** LOIO folds, temporal folds,
   simulations and Bayesian chains are preferable distributed tasks because they
   are large, independent units with low scheduler overhead.
4. **Measure the complete system.** Runtime alone is insufficient. Benchmark logs
   should include memory, throughput, worker count, chunk size, software revision
   and statistical equivalence.

The central workflow is therefore

```text
telemetry + environmental raster
            |
            v
    availability generation
            |
            v
  chunk-aware environmental extraction
            |
            v
 per-individual prepared Parquet partitions
            |
       +----+---------------------+
       |                          |
       v                          v
 pooled fitting             independent CV folds
       |                          |
       +------------+-------------+
                    v
          fused raster prediction
```

## Installation

HPC dependencies remain optional:

```bash
pip install -e ".[hpc,io]"
```

Bayesian benchmarks additionally require

```bash
pip install -e ".[bayesian]"
```

Optional NUTS implementations such as `nutpie`, `numpyro` and `blackjax` must be
installed in the environment when they are benchmarked. hrHSA does not make them
hard dependencies.

## Execution configuration

Scheduler choices are represented explicitly by `ExecutionConfig`:

```python
from hsa.compute import ExecutionConfig

compute = ExecutionConfig(
    backend="local",
    n_workers=8,
    threads_per_worker=1,
    chunk_mb=256,
    point_batch_mb=128,
    local_directory="/tmp/hrhsa-dask",
)
```

Supported execution modes are:

- `serial`: reference execution without a distributed scheduler;
- `local`: create a local Dask cluster;
- `distributed`: use a Dask client already owned by the notebook or batch job;
- `slurm`: create a `dask-jobqueue` SLURM cluster from `slurm_options`.

For mixed Python/GDAL/NumPy workloads, **one thread per Dask worker process** is a
safe starting point. Native BLAS/OpenMP libraries should also normally be limited
to one thread per process in the batch environment:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

This avoids accidental oversubscription such as 32 Dask processes each launching
32 BLAS threads.

## Storage-aware raster chunk planning

The original `suggest_xy_chunks()` helper remains available. The higher-level
planner additionally considers the source chunk layout:

```python
from hsa.compute import plan_raster_chunks

chunks = plan_raster_chunks(
    env,
    workload="point_sampling",
    target_chunk_mb=256,
    align_storage=True,
)
```

The basic memory model is

$$
M_{\mathrm{chunk}}
\approx B n_x n_y d,
$$

where $B$ is the number of selected bands, $n_x$ and $n_y$ are spatial chunk
dimensions and $d$ is bytes per value.

The important qualifier is **selected bands**. A model using six predictors should
not automatically read all 100 bands in a remote-sensing archive. hrHSA therefore
selects model bands before planning chunks and keeps the selected band dimension in
one chunk where possible.

Chunk alignment is a heuristic rather than a universal optimum. Parallel file
systems, local NVMe scratch and object storage have different latency and throughput
characteristics. The performance notebook includes a chunk-size sweep so the final
value is empirical.

## Chunk-aware static raster extraction

The reference sampler uses labelled Xarray nearest-neighbour indexing:

```python
from hsa.sampling import sample_raster_stack

reference = sample_raster_stack(points, env)
```

The accelerated sampler changes the execution order:

```python
from hsa.compute import sample_raster_stack_chunked

accelerated = sample_raster_stack_chunked(
    points,
    env,
    target_chunk_mb=256,
    client=client,
)
```

Internally, points are converted to nearest integer raster indices, mapped to their
spatial chunk, grouped by chunk, and all requested values from a touched chunk are
gathered together. Conceptually:

```text
point coordinates
      |
      v
nearest row / column
      |
      v
(row chunk, column chunk)
      |
      v
group all points touching the same chunk
      |
      v
load chunk once -> NumPy gather -> restore original row order
```

This is particularly advantageous for large, spatially clustered point sets and
chunked Zarr storage because the unit of computation becomes the same order as the
unit of storage.

Numerical equivalence can be checked directly during development:

```python
from hsa.compute import compare_sampling_engines

compare_sampling_engines(points, env, bands=["ndvi", "slope"])
```

The accelerated sampler preserves `used`, `Timestamp` and explicitly requested ID
columns.

## Prepare once, reuse across fitting and validation

The largest end-to-end improvement for multi-individual RSF workflows comes from
eliminating repeated geospatial extraction during cross-validation.

Create a prepared dataset once:

```python
prepared = analysis.prepare(
    "/scratch/project/hrhsa/prepared_lions",
    sampling_factor=20,
    execution=compute,
    client=client,
    engine="chunked",
)

prepared.describe()
```

The output has the form

```text
prepared_lions/
├── metadata.json
├── manifest.csv
└── individuals/
    ├── 00000-NPL28.parquet
    ├── 00001-NPL34.parquet
    └── ...
```

Each partition contains the sampled environmental predictors and the used/available
indicator for one individual. The preparation metadata records predictors, random
seed, sampling factor, thinning rule, chunk target and sampling engine.

A pooled frequentist fit can then bypass geospatial extraction:

```python
fit = analysis.fit(prepared=prepared)
```

Bayesian RSFs can reuse the same prepared table:

```python
fit_bayes = bayes_analysis.fit(
    prepared=prepared,
    nuts_sampler="nutpie",
)
```

Prepared datasets are intended as **analytical products**, not opaque caches. They
should be versioned by configuration or stored under immutable run directories when
multiple availability definitions are compared.

### Scientific safeguards

A prepared LOIO run refuses to continue when its sampling factor differs from the
validation scheme. It also requires the cached thinning rule to match both training
and held-out thinning. This is intentionally strict: performance optimization must
not silently alter the scientific fold definition.

For workflows that require different thinning rules for train and test, use the
reference LOIO path until a cache containing separate training and evaluation roles
is prepared explicitly.

## Distributed leave-one-individual-out validation

The standard object-oriented validation API is unchanged:

```python
from hsa.rsf import LeaveOneIndividualOut

scheme = LeaveOneIndividualOut(
    sampling_factor_train=20,
    n_background=100_000,
    n_bins=20,
    seed=42,
)
```

Serial reference validation remains:

```python
result = analysis.validate(scheme)
```

Prepared validation reuses cached environmental values:

```python
result = analysis.validate(
    scheme,
    prepared=prepared,
)
```

and independent held-out folds can be submitted to Dask simply by passing an
existing client:

```python
result = analysis.validate(
    scheme,
    prepared=prepared,
    client=client,
    surface_engine="chunked",
    target_chunk_mb=256,
)
```

By default the distributed result does not retain fitted models or complete
fold-specific raster surfaces in the returned diagnostics. This keeps scheduler
communication bounded. They can be retained when needed:

```python
result = analysis.validate(
    scheme,
    prepared=prepared,
    client=client,
    keep_models=True,
    keep_surfaces=True,
)
```

For large validation studies it is usually preferable to persist selected model
objects or final surfaces explicitly rather than returning dozens of large objects
to the scheduler process.

## Fused surface prediction

The reference prediction engine expresses each model term as an Xarray operation.
That is readable and remains the numerical reference:

```python
rsf = fit.predict_surface(engine="reference")
```

The HPC engine evaluates the complete model inside each spatial block:

```python
rsf = fit.predict_surface(
    engine="chunked",
    target_chunk_mb=256,
    align_storage=True,
)
```

Within a block the operation is approximately

```text
read selected predictors
 -> standardize
 -> linear terms
 -> quadratic terms
 -> interactions
 -> categorical masks
 -> exp(eta)
 -> write RSF block
```

Temporary standardized and interaction arrays are discarded with the block instead
of becoming complete raster-sized intermediates. This is primarily a memory-bandwidth
optimization and is expected to matter most for large rasters and multivariate
models.

## Spatiotemporally tiled dynamic covariates

Time-varying SSF predictors often come from remote or chunked datasets such as ERA5.
A temporal batch alone can be inefficient when animals are far apart: the bounding
box for one month may cover a large fraction of a continent.

The accelerated wrapper introduces a joint time-space routing layer:

```python
from hsa.compute import sample_dynamic_covariates_tiled

choices = sample_dynamic_covariates_tiled(
    choices,
    era5,
    variables={
        "u10": "wind_u",
        "v10": "wind_v",
        "skt": "skin_temperature",
    },
    time_col="end_time",
    temporal_tile="M",
    spatial_tile_degrees=2.0,
    method="linear",
)
```

The interpolation itself is unchanged. The new parameter only reduces the spatial
extent of each remote read. Recommended tile sizes should be established with the
I/O benchmark rather than chosen from theory alone.

## Bayesian acceleration

Bayesian computation should be evaluated using **effective samples per second**, not
wall time alone. `BayesianRSF.fit()` exposes PyMC's NUTS implementation explicitly:

```python
fit = analysis.fit(
    prepared=prepared,
    nuts_sampler="nutpie",
    sample_kwargs={
        "draws": 1500,
        "tune": 1500,
        "chains": 4,
        "target_accept": 0.95,
    },
)
```

Candidate backends include the native PyMC sampler and optional `nutpie`, `numpyro`
and `blackjax` implementations supported by the installed PyMC version. The Bayesian
SSF API already forwards arbitrary `sample_kwargs` to `pm.sample`, so the same
comparison can be made there by including `nuts_sampler` in `sample_kwargs`.

For every backend compare at least:

- wall-clock time;
- bulk and tail ESS per second;
- R-hat;
- divergences;
- posterior means and standard deviations;
- posterior predictive or held-out scores.

A faster sampler that materially changes inference is not a valid optimization.

## SLURM execution

A normal SLURM-backed session remains explicit:

```python
from dask.distributed import Client
from hsa.compute import make_slurm_cluster

cluster = make_slurm_cluster(
    queue="general",
    cores=4,
    processes=4,
    memory="32GB",
    walltime="04:00:00",
    local_directory="$TMPDIR",
    scale_jobs=8,
    env_extra=[
        "export OMP_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
    ],
)
client = Client(cluster)
```

Site-specific project/account names, partitions and module commands belong in the
batch script or `slurm_options`, not in the reusable scientific package.

## Benchmark harness

`benchmarks/benchmark_hpc.py` provides a reproducible synthetic command-line
benchmark for point sampling and fused raster prediction.

For example:

```bash
python benchmarks/benchmark_hpc.py \
    --benchmark sampling \
    --raster-size 4096 \
    --bands 6 \
    --points 1000000 \
    --workers 8 \
    --chunk-mb 256 \
    --output benchmarks/results.jsonl
```

Every JSONL record includes:

- benchmark name;
- wall time;
- processed rows/cells;
- throughput;
- worker and thread counts;
- chunk target;
- process peak RSS where supported;
- hostname and Python/platform information;
- git commit SHA when run inside a checkout;
- benchmark-specific metadata.

Read and analyze the log with

```python
from hsa.compute import read_benchmark_records, scaling_table

results = read_benchmark_records("benchmarks/results.jsonl")
strong = scaling_table(results)
```

Two notebooks accompany this implementation:

- `notebooks/hpc/01_hpc_features.ipynb` demonstrates the new execution features;
- `notebooks/hpc/02_performance_evaluation_blueprint.ipynb` is a template for the
  KONWIHR performance campaign.

## Performance evaluation protocol

### 1. Correctness gate

Before accepting an optimization, compare accelerated and reference outputs on
small deterministic data.

For raster extraction use exact/near-exact value comparisons. For fitted models,
compare coefficients, log-likelihood/predictions and validation summaries. For
Bayesian samplers compare posterior summaries within Monte Carlo uncertainty.

### 2. Strong scaling

Keep the scientific workload fixed and vary workers, for example

```text
1, 2, 4, 8, 16, 32, 64 workers
```

Report

$$
S_p = \frac{T_1}{T_p}
$$

and

$$
E_p = \frac{S_p}{p}.
$$

When the smallest feasible run uses more than one worker, normalize worker counts
relative to that baseline and report the fact explicitly.

### 3. Weak scaling

Keep work per worker approximately fixed while increasing workload and resources,
for example:

| Workers | Point queries |
| ---: | ---: |
| 1 | 2 million |
| 2 | 4 million |
| 4 | 8 million |
| 8 | 16 million |
| 16 | 32 million |
| 32 | 64 million |

Weak-scaling efficiency is high when runtime remains approximately constant.

### 4. Chunk-size sweep

At a fixed workload compare, for example,

```text
64, 128, 256, 512 MiB
```

and record runtime, task count, spill volume and storage throughput. The best chunk
size on local NVMe may not be the best size on a shared parallel filesystem.

### 5. Cold and warm storage runs

Separate cold-cache I/O tests from repeated warm-cache computational tests whenever
possible. Otherwise filesystem cache effects can be mistaken for algorithmic
speedup.

### 6. End-to-end workload

Microbenchmarks are necessary but not sufficient. The final KONWIHR evaluation
should include an end-to-end ecological workflow such as:

```text
raw telemetry
 -> availability
 -> environmental extraction
 -> prepared dataset
 -> pooled RSF
 -> LOIO validation
 -> full-resolution prediction
```

Report stage-specific and total runtime. This identifies whether an optimization
changes the dominant bottleneck rather than only improving a small kernel.

## Recommended canonical benchmark ladder

A practical benchmark ladder is:

| Scale | Point queries | Raster scale | Purpose |
| --- | ---: | ---: | --- |
| Tiny | 10,000 | ~1,000 x 1,000 | correctness / CI |
| Medium | 1 million | ~10,000 x 10,000 | workstation |
| Large | 10 million | ~30,000 x 30,000 | single HPC node |
| Extreme | 50-100 million | multi-store / 100k-equivalent | distributed cluster |

The exact raster dimensions may be adapted to available storage. What matters is
that the benchmark definition and git revision are recorded and that the same
workload is reused when comparing software revisions.

## What should be reported for KONWIHR

A strong final performance report should contain:

1. reference versus accelerated correctness tests;
2. sampling throughput versus chunk size;
3. strong scaling of chunk-aware point extraction;
4. strong scaling of prepared LOIO folds;
5. weak scaling of point extraction;
6. peak memory and Dask spill behavior;
7. fused versus reference surface-prediction runtime and memory;
8. dynamic ERA5 sampling with temporal-only versus spatiotemporal tiling;
9. Bayesian ESS/second and diagnostic equivalence across NUTS backends;
10. one full end-to-end real study workflow.

The goal is not to claim perfect linear scaling. The useful scientific result is to
identify where scaling stops, why it stops (storage bandwidth, scheduler overhead,
memory or serial statistical work), and how much larger a scientifically realistic
analysis becomes possible with the optimized architecture.

## Development priorities

The implemented architecture follows the following priority order:

1. **prepared per-individual analytical datasets** to eliminate repeated
   environmental extraction;
2. **chunk-aware static raster extraction** to align work with storage;
3. **distributed LOIO folds** as large independent tasks;
4. **machine-readable benchmark infrastructure**;
5. **fused blockwise surface prediction**;
6. **spatiotemporal tiling for dynamic predictors**;
7. **explicit accelerated Bayesian sampler benchmarking**;
8. further design-matrix optimization only if profiling shows it is material;
9. alternative frequentist optimizers only if model fitting, rather than data
   preparation, becomes a dominant end-to-end cost.

This ordering is intentional. Replacing `statsmodels.Logit` with a more exotic
optimizer is not useful if 95% of an analysis is spent reading and repeatedly
sampling environmental data. Optimization should follow measured bottlenecks.
