# Implementation

**hrHSA** is implemented as a scientific analysis framework with two deliberately separate concerns:

1. the **scientific model** — how availability, environmental predictors, individual variation and movement enter the analysis; and
2. the **execution model** — how the same operations are evaluated efficiently on a notebook, workstation or distributed cluster.

The package is no longer only an RSF toolkit. Development has added stateful frequentist and Bayesian RSF workflows, movement-informed SSFs, proposal-corrected iSSFs, dynamic environmental annotation, validation and heterogeneity diagnostics, prepared analytical datasets, accelerated raster kernels and explicit HPC execution paths.

This page is the architectural overview. Detailed APIs and examples live in the model-specific and performance pages linked throughout.

## 1. Current capability map

The most important implementation distinction is between what is fully available now and what remains a future extension.

| Component | Current implementation |
| --- | --- |
| Frequentist RSF | Implemented through `FrequentistRSF` and lower-level functional kernels |
| Hierarchical Bayesian RSF | Implemented through `BayesianRSF`, including random slopes, LOIO validation and optional shrinkage |
| Frequentist SSF | Implemented as conditional logistic choice models |
| Hierarchical Bayesian SSF | Implemented as individual-level softmax models with partial pooling |
| Frequentist iSSF | Implemented with proposal correction, movement terms and environmental movement modifiers |
| Hierarchical Bayesian iSSF | Implemented by converting a prepared iSSF analysis to the Bayesian hierarchy |
| Dynamic environmental fields | Implemented for SSF/iSSF annotation and diagnostic plotting |
| RSF validation uncertainty | Implemented with blocked bootstrap and contiguous temporal validation |
| SSF validation | Implemented with LOIO, temporal blocking and fitted-model PSIS-LOO |
| iSSF cross-validation | Not yet exposed; the API raises rather than leaking fold-specific scaling/proposal information |
| HPC acceleration | Implemented for raster extraction, prepared RSF data, distributed LOIO and fused surface prediction |
| Habitat-biased forward simulation | Not yet implemented; the current simulation entry point is a placeholder |

The distinction matters because an overview page should describe the package that exists, not planned functionality.

## 2. Design principles

The architecture follows six principles.

### Scientific state is explicit

A fitted model is more than its coefficients. Scaling, feature construction, predictor definitions, availability configuration, environmental support and validation metadata must travel with the fit. hrHSA therefore uses stateful analysis and result objects to keep those pieces together.

### Object interfaces orchestrate public kernels

The high-level classes do not hide the numerical implementation. Functional kernels remain public and independently testable. This supports concise notebooks without making the object layer the only way to reproduce an analysis.

### RSF and SSF remain distinct abstractions

Both frameworks use telemetry and environmental data, but their likelihoods and availability processes differ fundamentally. SSF classes therefore do not inherit from `RSFAnalysis`. Reusing data-handling ideas is preferable to forcing biologically different models into one class hierarchy.

### Expensive geospatial work should be reusable

Environmental extraction can dominate runtime in high-resolution studies. hrHSA can therefore prepare sampled analytical tables once and reuse them across fitting and validation rather than repeatedly resampling the same raster values.

### Reference and accelerated paths coexist

Readable serial/vectorized implementations remain the numerical reference. HPC-oriented kernels are separate and must be checked against the reference implementation. Performance work is not allowed to silently redefine the scientific analysis.

### Invalid scientific shortcuts should fail loudly

Examples include observed SSF endpoints outside environmental support, iSSF validation before fold-specific scaling is available, mismatched prepared-data configurations and distance calculations in inappropriate coordinate systems. The package generally raises instead of silently producing a plausible result from an invalid design.

## 3. Package architecture

The source uses the conventional `src` layout. At a high level:

```text
src/hsa/
├── sampling.py          # spatial availability and reference raster sampling
├── features.py          # reproducible RSF feature construction
├── types.py             # shared specifications
├── rsf/                 # RSF analysis, fits, prediction, validation and HPC paths
├── ssf/                 # SSF/iSSF choice sets, annotation, fitting and diagnostics
├── movement/            # trajectory and movement-kernel utilities
├── diagnostics/         # shared diagnostic helpers
├── compute/             # execution, chunking, prepared data, I/O and benchmarks
├── remote_sensing/      # optional Earth-observation interfaces
└── simulation/          # future habitat-biased forward simulation boundary
```

The model packages contain several layers internally. For example, `hsa.rsf` separates frequentist, Bayesian, validation, object and HPC responsibilities, while `hsa.ssf` separates choice-set construction, environmental annotation, frequentist/Bayesian fitting, iSSF design and dynamic diagnostics.

This organization is intentional: statistical code, spatial data preparation and distributed execution can evolve independently while still being composed through the public analysis objects.

## 4. Object layer and result objects

The RSF object hierarchy is shallow:

```text
RSFAnalysis
├── FrequentistRSF
└── BayesianRSF

RSFFit
├── FrequentistRSFFit
└── BayesianRSFFit
```

Validation strategies are composed with an estimator rather than encoded into increasingly specific subclasses:

```text
LeaveOneIndividualOut
BlockedBootstrap
ContiguousTemporalBlocks
```

This prevents combinations such as estimator × validation method × uncertainty method from becoming separate class hierarchies.

The same design idea is used in SSF/iSSF workflows: analysis objects own the prepared telemetry, choice sets, annotations, model definitions and scaling; fitted objects expose coefficient summaries, diagnostics and derived ecological quantities.

For more detail, see [Object-oriented RSF workflows](rsf_objects.md).

## 5. Reproducible RSF model construction

### Feature specification

Frequentist RSFs use a declarative `FeatureSpec`:

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

Continuous predictors are standardized during fitting. Quadratic and interaction terms are then generated on the fitted standardized scale. Categorical levels, design-column order and scaling metadata are retained with the fit so validation and raster prediction reproduce the original model matrix exactly.

The low-level frequentist kernel remains available:

```python
model, scaler, spec, meta = fit_rsf(samples, spec)
```

The fitted state is therefore conceptually

```text
coefficients
+ scaler
+ feature specification
+ design metadata
+ environmental predictor definition
```

rather than a bare Statsmodels result.

### Frequentist optimization

`fit_rsf()` exposes the Statsmodels optimizer through `method=` and optional fit arguments. The scientific default remains conservative; alternative solvers can be benchmarked but should only replace a project default after convergence, coefficient and likelihood equivalence are demonstrated.

Inference performance is documented separately in [Inference performance benchmarking](inference-performance.md).

## 6. Hierarchical Bayesian RSF implementation

`BayesianRSF` owns the telemetry, environmental stack, predictors, availability configuration, binning definition, random slopes and model options.

The Bayesian workflow:

```text
used/available sample
        ↓
standardize original predictor observations
        ↓
optional ecological-unit binning / aggregation
        ↓
hierarchical PyMC model
        ↓
InferenceData + aggregation/scaling metadata
        ↓
prediction / diagnostics / validation
```

The model supports population intercepts and slopes together with non-centred individual random intercepts and random slopes. The aggregated likelihood is Binomial, reducing repeated identical predictor combinations before MCMC when binning is used.

A typical fit is:

```python
fit = rsf.fit(
    sampling_factor=50,
    sample_kwargs={
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
        "target_accept": 0.9,
    },
)
```

The result retains the PyMC model, `InferenceData`, predictor definitions, scaling/aggregation metadata and environmental stack.

### Shrinkage and collinearity diagnostics

For small biologically pre-specified predictor sets, independent standardized Normal priors remain the default. A regularized horseshoe prior can be requested for larger correlated candidate sets.

The implementation also exposes:

- posterior correlation among population coefficients;
- predictor correlation on the fitted design;
- diagnostics identifying competing correlated effects; and
- summaries of global/local shrinkage scales.

These are regularization and identifiability diagnostics, not automated causal-variable-selection tools.

See [Hierarchical Bayesian RSF workflow](bayesian_rsf.md).

## 7. RSF validation architecture

Validation is a first-class part of the analysis object.

### Leave-one-individual-out

The same `LeaveOneIndividualOut` strategy can be passed to frequentist and Bayesian RSFs. Each held-out animal is evaluated relative to its own availability while the model is fitted without that animal.

Results retain fold-specific summaries rather than only a pooled score. Depending on estimator type they may include:

```text
coefficients
Boyce bins / curves
calibration bins
diagnostics
posterior score matrices
fold metadata
```

### Post-fit uncertainty

Two complementary strategies operate on held-out RSF predictions:

**Blocked bootstrap** resamples whole temporal blocks. Frequentist fits keep the fitted coefficient vector fixed; Bayesian fits can pair each replicate with a posterior draw.

**Contiguous temporal blocks** score real chronological periods and diagnose temporal non-stationarity. They preserve the exact time units that actually contained observations so telemetry gaps remain auditable.

Estimator-independent Boyce and temporal-block helpers are shared rather than duplicated between frequentist and Bayesian modules.

## 8. SSF implementation

The SSF workflow begins with movement-conditioned availability rather than a broad spatial background.

### Choice-set construction

Observed steps are derived from telemetry at an expected sampling interval. Available endpoints are generated from fitted movement distributions and begin at the same step origin as the observed endpoint.

Choice-set construction preserves:

- step length;
- turning angle;
- movement transforms;
- candidate bearing;
- proposal log density; and
- stratum/individual identity.

When an environmental raster defines valid support, available candidates falling outside it are regenerated rather than snapped to the edge. Observed endpoints are never moved. Unsupported observed strata either raise or, when explicitly requested, are removed as complete strata and recorded in diagnostics.

### Environmental annotation

Static predictors are sampled at candidate endpoints. Dynamic predictors are sampled independently for every relevant alternative/time combination rather than broadcast blindly across a stratum.

Vector fields can be converted to model-ready quantities such as:

```text
wind_support
crosswind
abs_crosswind
wind_alignment
```

using geodesic candidate bearings. This matters because atmospheric vectors describe true east/north components whereas a projected raster axis need not align perfectly with geographic direction.

### Frequentist SSF

`FrequentistSSF` fits conditional logistic regression with one chosen endpoint per stratum and no intercept. Predictor scaling is learned on the fitting choice table. Non-finite data are handled at the **whole-stratum** level so the conditional-choice structure is not changed by deleting individual alternatives.

No-pooling individual fits reuse the same model design and can be used to inspect heterogeneity before fitting a hierarchy.

### Bayesian SSF

`BayesianSSF` fits a hierarchical categorical/softmax model with population means, individual deviations and between-individual heterogeneity. Its fitted objects expose separate population, individual and heterogeneity summaries plus posterior trace/forest diagnostics.

Scaling can be shared deliberately between a trusted full fit and a pilot fit, but cross-validation always estimates scaling inside the training partition.

See [Step-selection workflows](ssf.md).

## 9. SSF diagnostics and validation

Development added diagnostics that distinguish weak ecological signal from weak study design.

### Selection opportunity

Within-stratum predictor variation determines how much information a choice set contains about a coefficient. Individual summaries therefore include selection-opportunity diagnostics evaluated under a common pooled reference coefficient when comparing animals.

### Conditional Information Inflation Factor

CIIF is derived from the conditional-logit information matrix and measures overlap among predictor information. It is an identifiability diagnostic rather than a bootstrap or temporal uncertainty interval.

### Predictive validation

SSF validation uses proper conditional log scores, reported relative to uniform choice. Supported validation questions include:

- transfer to a new individual through LOIO;
- temporal stability through blocked temporal validation; and
- new-stratum/known-individual prediction through fitted-model PSIS-LOO.

These schemes deliberately answer different questions.

## 10. iSSF implementation

The iSSF API was developed as a separate layer on top of the choice-set machinery because it requires more explicit control over movement terms and proposal correction.

The workflow is intentionally staged:

```text
telemetry
   ↓
create analysis
   ↓
generate movement-informed candidate steps
   ↓
annotate static / dynamic / vector conditions
   ↓
declare ecological selection + movement model
   ↓
prepare design
   ↓
frequentist or Bayesian fit
```

An environmental raster is therefore not required when the analysis object is created, and the ecological model can be changed without regenerating candidates or resampling already annotated covariates.

### Proposal correction

Candidate generation stores `proposal_logpdf`. The iSSF working utility includes the fixed correction

```text
-proposal_logpdf
```

so the proposal distribution used to draw alternatives is not confused with the ecological movement kernel being estimated.

### Movement model

The default movement basis is:

```text
step_length_km
log_step_length
cos_turn_angle
```

`set_model()` can combine endpoint-selection predictors with environmental movement modifiers. Start-of-step conditions that are constant within a stratum enter through interactions with movement terms, while candidate-varying directional quantities can also contribute main effects.

For example:

```python
issf.set_model(
    selection=["elevation", "slope"],
    modifiers={
        "heat_start": "step_length",
        "vrm_start": "step_length",
        "wind_start_support": ["step_length", "turning"],
    },
)
```

The model specification records whether modifiers are stratum-constant or candidate-varying and expands shorthand movement components consistently.

### Derived movement interpretation

Fitted objects can transform movement coefficients into quantities such as expected displacement, step-length distributions and turning-angle distributions across environmental levels. This is important because a simple interaction on the coefficient scale can imply a nonlinear movement response after transformation back to displacement space.

### Frequentist and Bayesian iSSF

Frequentist fitting supports the optimized conditional engine. A prepared iSSF analysis can then be converted to the hierarchical Bayesian form without rebuilding choice sets or re-annotating the environment:

```python
bayes = issf.to_bayesian()
```

The Bayesian result separates population coefficients, partially pooled individual coefficients and heterogeneity in the same manner as the Bayesian SSF.

### External MCMC boundary

For long external jobs, a Bayesian iSSF analysis can export model-ready numeric design data and metadata without serializing large spatial objects. A subsequently written `InferenceData` NetCDF can be loaded back into the analysis object.

### Current validation boundary

The iSSF public API does not yet expose the existing SSF cross-validation strategies. Correct validation must fit scaling inside each training fold and carry the proposal correction through held-out choice sets. Until that workflow is implemented, `validate()` raises intentionally.

See [Integrated step-selection workflows](issf.md).

## 11. Dynamic environmental diagnostics

Dynamic fields can be inspected before they enter a model. The quick-look utilities support:

- spatial maps at requested times;
- weighted histograms or ECDFs;
- timezone-aware selection of UTC-backed fields;
- area weighting by latitude;
- derived fields;
- bounded study extents; and
- spatial decimation for fast exploratory views.

The same bounded dynamic data machinery helps users verify sign conventions, units, diurnal development and transformations before constructing an SSF/iSSF predictor.

See [Dynamic environmental condition plots](ssf_dynamic_conditions.md).

## 12. Geospatial data model

Telemetry is represented primarily as `GeoDataFrame` objects. Environmental fields use labelled `xarray` arrays/datasets and retain coordinates, dimensions and CRS metadata.

A common static stack is conceptually

```text
band × y × x
```

while dynamic environmental data typically add time.

CRS checks occur before spatial operations. Distance-based operations require meaningful projected units, while atmospheric vector support is derived using geodesic bearings where true directional interpretation matters.

Only bands required by the fitted predictor/model specification should be sampled. This rule is applied in pooled RSF workflows and validation paths to avoid unnecessary raster I/O and excessively large lazy indexing graphs.

## 13. Persistent analytical products

Large multidimensional rasters can be stored as Zarr, while sampled tabular data use Parquet. This separates expensive environmental processing from repeated statistical experiments.

For large RSF workflows, `analysis.prepare()` creates an immutable-style prepared dataset organized by individual:

```text
prepared_analysis/
├── metadata.json
├── manifest.csv
└── individuals/
    ├── 00000-<ID>.parquet
    ├── 00001-<ID>.parquet
    └── ...
```

The metadata records the predictor set and scientific preparation choices such as sampling factor, thinning, seed and extraction engine.

A prepared dataset is not treated as an opaque cache. Compatibility checks prevent a validation run from silently reusing data prepared under a different scientific design.

## 14. Execution configuration

HPC execution is configured separately from the ecological model through `ExecutionConfig`.

Supported modes include:

```text
serial       reference/local execution
local        local Dask cluster
distributed  caller-owned Dask client
slurm        Dask-jobqueue SLURM cluster
```

This allows the same analysis code to move between a laptop, workstation and cluster without embedding scheduler details in the statistical model.

A safe default for mixed Python/NumPy/GDAL workloads is often one thread per worker process, with BLAS/OpenMP thread counts controlled explicitly to avoid oversubscription.

## 15. Chunk-aware raster extraction

High-resolution workflows are frequently limited by storage access and memory traffic rather than arithmetic.

The reference sampler uses labelled xarray nearest-neighbour indexing. The accelerated sampler changes the execution order:

```text
point coordinates
      ↓
nearest raster indices
      ↓
map indices to storage/computational chunks
      ↓
group all points touching each chunk
      ↓
load chunk once
      ↓
NumPy gather
      ↓
restore original point order
```

Point-to-chunk grouping is vectorized so multi-million-point samples do not spend most of their time appending Python objects. When a distributed client is used, point-local index payloads are scattered separately from the Dask graph to keep scheduler graphs compact.

The accelerated sampler can use reduced numeric precision for extraction where appropriate, while correctness tests compare it against the reference path before performance conclusions are accepted.

## 16. Chunk planning and memory

For a selected raster subset with $B$ bands, spatial chunk dimensions $n_x,n_y$ and $d$ bytes per value,

$$
M_{\mathrm{chunk}} \approx Bn_xn_yd.
$$

The critical word is **selected**: a six-predictor model should not automatically read every band from a large Earth-observation archive.

The chunk planner considers memory targets and, where useful, source storage chunks. Chunk size remains a performance parameter rather than a universal constant and should be benchmarked under the real filesystem/storage backend.

## 17. Fused RSF surface prediction

The reference prediction path evaluates transformed model terms as transparent xarray operations. The accelerated path evaluates the complete fitted model inside each spatial block:

```text
read predictors
 → standardize
 → linear terms
 → quadratic terms
 → interactions
 → categorical masks
 → exp(eta)
 → output block
```

Intermediate standardized and interaction rasters die with the block instead of becoming full-resolution arrays. This reduces memory traffic and peak memory for large multivariate surfaces.

The reference path remains available for numerical comparison.

## 18. Distributed validation

Independent LOIO folds are natural distributed tasks because they have substantial computational work and little need to communicate with one another.

Prepared RSF validation can therefore:

1. read cached per-individual predictor partitions;
2. fit each held-out fold independently;
3. evaluate the held-out animal against its own domain; and
4. return compact diagnostics rather than shipping every fitted model/surface back to the scheduler by default.

Large model and raster objects are retained only when explicitly requested.

Nested Dask execution is avoided: when a distributed fold itself touches a Dask-backed raster, the inner raster graph is executed locally inside that worker instead of recursively submitting work to the same cluster.

## 19. Dynamic spatiotemporal sampling

Dynamic environmental datasets can be expensive when a temporal batch spans a geographically large bounding box. The accelerated dynamic sampler can partition reads jointly in time and space before applying the same interpolation semantics.

This changes data routing, not ecological meaning. Tile sizes should be determined empirically because local NVMe, parallel filesystems and remote/object stores have different optimal access patterns.

## 20. Bayesian performance engineering

A NUTS chain is sequential, while separate chains, LOIO folds, candidate models and simulation replicates can run independently. Bayesian performance is therefore evaluated differently from raster scaling.

`BayesianRSF.fit()` can expose alternative PyMC-supported NUTS backends such as native PyMC, `nutpie`, `numpyro` or `blackjax` when those optional packages are installed.

Useful comparison metrics include:

- wall time;
- bulk/tail ESS per second;
- R-hat;
- divergences;
- tree depth / NUTS steps;
- posterior agreement; and
- held-out predictive agreement.

Raw draws per second alone are not a sufficient performance metric.

### Observation-level posterior storage

The hierarchical RSF must compute the linear predictor `eta` for its likelihood, but storing every observation-level `eta` draw can dominate posterior memory. hrHSA therefore does not store it as a posterior deterministic by default; it can be enabled explicitly when needed.

This is representative of the package's performance philosophy: avoid retaining large intermediates unless they are scientifically required.

## 21. Benchmark architecture

Performance benchmarking is part of the implementation rather than an informal timing exercise.

The benchmark harness records quantities such as:

```text
wall time
throughput
worker/thread counts
chunk configuration
memory/RSS
hostname/platform
software/git revision
benchmark-specific correctness metadata
```

Separate campaigns measure:

- point extraction and raster prediction;
- workstation reference-versus-accelerated behavior;
- strong/weak scaling;
- chunk-size sensitivity;
- dynamic-field I/O; and
- frequentist/Bayesian statistical inference.

Every accelerated path should pass a correctness gate before speedup is interpreted.

See [HPC execution and performance engineering](hpc-performance.md), [Workstation benchmark](workstation-benchmark.md), [Inference performance benchmarking](inference-performance.md), [CoolMUC-4 tuning](coolmuc4-tuning.md) and [CoolMUC-4 benchmark](coolmuc4-benchmark.md).

## 22. Optional dependencies

The core package installs the numerical/geospatial stack only. Specialized functionality is separated into extras defined in `pyproject.toml`:

```bash
pip install -e .
pip install -e ".[earthengine]"
pip install -e ".[dask]"
pip install -e ".[hpc,io]"
pip install -e ".[bayesian]"
```

Optional NUTS implementations are deliberately not hard dependencies. Install the backend explicitly in the environment where it will be used or benchmarked.

External services are initialized explicitly. Importing `hsa` should not automatically authenticate Earth Engine or create a distributed cluster.

## 23. Reproducibility contract

A reproducible hrHSA analysis should preserve at least:

- package and dependency versions;
- source-data provenance;
- telemetry sampling rules;
- coordinate reference systems;
- availability-domain definitions;
- random seeds and available:used sampling ratios;
- feature/model specification;
- fitted scaling and design metadata;
- prepared-data configuration;
- validation partitions and exact held-out temporal units;
- sampler settings and posterior diagnostics for Bayesian models; and
- execution/benchmark configuration when performance claims are made.

Scientific and computational reproducibility are linked. A faster run is not equivalent if it changed predictor precision, availability, thinning, environmental support or the validation fold definition without recording that change.

## 24. Current development boundaries

The package now contains substantially more production functionality than the earlier implementation overview described, but several boundaries remain intentional.

**Implemented and production-facing:** frequentist/Bayesian RSFs, frequentist/Bayesian SSFs, frequentist/Bayesian iSSF fitting, movement-response diagnostics, dynamic annotation, RSF and SSF validation, prepared RSF data and HPC raster/validation kernels.

**Not yet complete:** iSSF cross-validation with fold-specific scaling/proposal handling and habitat-biased forward trajectory simulation.

Maintaining these boundaries in the documentation is important. New functionality should move from this section into the relevant architectural section only when the public API and validation semantics are actually implemented.

## 25. Where to go next

Use this page to understand how the pieces fit together, then continue with the model-specific documentation:

- [Getting started](getting_started.md)
- [Theory](theory.md)
- [Object-oriented RSF workflows](rsf_objects.md)
- [Hierarchical Bayesian RSF workflow](bayesian_rsf.md)
- [Step-selection workflows](ssf.md)
- [Integrated step-selection workflows](issf.md)
- [Dynamic environmental condition plots](ssf_dynamic_conditions.md)
- [HPC execution and performance engineering](hpc-performance.md)
- [API reference](api.md)

The central implementation principle is that increasing ecological realism and computational scale should not require sacrificing inspectability. The package keeps scientific state explicit, separates ecological design from execution, preserves reference kernels beside accelerated ones and refuses performance shortcuts that would silently change the inferential question.
