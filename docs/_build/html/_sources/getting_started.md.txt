# Getting started

This guide introduces the core **frequentist habitat-selection workflow** in hrHSA. Starting from telemetry records and a raster stack of environmental covariates, we will:

1. prepare and project relocation data;
2. define the spatial domain considered available;
3. sample environmental conditions at used and available locations;
4. inspect predictor distributions;
5. compare a biologically defined set of candidate models;
6. fit and project a resource-selection function;
7. evaluate transferability among individuals.

The final section briefly shows how the same used–available design can be extended to hierarchical Bayesian inference.

## Installation

Clone the repository and install hrHSA with its geospatial and distributed-computing dependencies:

```bash
git clone <repository-url>
cd hrHSA
python -m pip install -e ".[hpc]"
```

Import the components used throughout this guide:

```python
import numpy as np
import pandas as pd

from hsa import FeatureSpec
from hsa.movement.geometry import prepare_trajectory_data
from hsa.compute import open_raster_stack_zarr, suggest_xy_chunks
from hsa.sampling import (
    get_availability_domain,
    sample_available_points,
    sample_raster_stack,
)
from hsa.diagnostics import inspect_sampled_use_available
from hsa.rsf import (
    compare_single_predictors,
    evaluate_linear_candidates_up_to_k,
    fit_rsf,
    predict_rsf_surface,
)
from hsa.rsf.cv import leave_one_individual_out_rsf
from hsa.rsf.validation import plot_boyce_curves
```

## 1. Prepare relocation data

hrHSA expects timestamps, individual identifiers, and spatial coordinates. Relocations should be represented in a projected coordinate reference system appropriate for the study area.

```python
raw = pd.read_csv("data/relocations.csv")

reloc = prepare_trajectory_data(
    raw,
    id_col="Individual_ID",
    timestamp_col="Timestamp",
    lon_col="longitude",
    lat_col="latitude",
    source_crs="EPSG:4326",
    target_crs="EPSG:32733",  # replace with the CRS of the raster stack
    round_freq="h",
    drop_duplicate_fixes=True,
)

reloc.head()
```

The function parses timestamps, removes invalid records, projects coordinates, sorts trajectories, and optionally removes duplicate observations within a specified temporal interval.

```{important}
The target CRS must use meaningful planar units and should match the environmental raster stack. Geographic longitude–latitude coordinates should not be used for distance-, area-, or availability-based operations.
```

## 2. Open the environmental raster stack

Environmental covariates can be stored as a Zarr-backed `xarray.DataArray` with dimensions `band`, `y`, and `x`.

```python
env = open_raster_stack_zarr(
    "data/environment.zarr",
    name="env",
    chunks=None,
)

chunks = suggest_xy_chunks(
    env,
    target_chunk_mb=128,
)

env = env.chunk(chunks)

print(chunks)
print(env)
```

`suggest_xy_chunks` chooses approximately square spatial chunks while retaining all predictor bands within each chunk. This provides a defensible starting point for raster sampling and prediction, although chunk sizes should ultimately be benchmarked against the available memory, storage system, and workload.

Confirm that the relocation and raster coordinate systems agree:

```python
assert reloc.crs == env.rio.crs
```

## 3. Define availability

A resource-selection analysis compares observed use with environmental conditions that were accessible but not necessarily used. The definition of availability is therefore part of the ecological model rather than merely a computational preprocessing decision.

Here, separate availability domains are estimated for each individual using a centroid-trimmed 95% minimum convex polygon:

```python
domains = get_availability_domain(
    reloc,
    id_col="Individual_ID",
    estimator="mcp",
    quantile=0.95,
    min_points=5,
)

domains.head()
```

Available locations are then sampled independently within each individual domain:

```python
samples = sample_available_points(
    domains,
    used=reloc,
    id_col="Individual_ID",
    n_per_used=10,
    seed=42,
)

samples["used"].value_counts()
```

The resulting table contains both observed relocations (`used=True`) and sampled available locations (`used=False`).

```{note}
The number of available points controls the numerical approximation of the available environment. It does not represent the number of locations an animal could literally have visited.
```

## 4. Extract environmental covariates

Sample all raster bands at the used and available locations:

```python
sampled = sample_raster_stack(
    samples,
    env,
    id_cols="Individual_ID",
)

sampled.head()
```

Each row now represents one used or available location together with its environmental covariates.

## 5. Inspect used and available environments

Before fitting a model, inspect missingness, predictor distributions, and the degree of separation between used and available samples.

```python
inspection = inspect_sampled_use_available(
    sampled,
    id_col="Individual_ID",
    plot=True,
    plot_top_n=8,
)

inspection["pooled_stats"]
```

The diagnostic summary includes distributional differences such as the Kolmogorov–Smirnov statistic, Wasserstein distance, differences in means and medians, and the common-language effect size.

These summaries are descriptive. They are useful for identifying data problems, implausible contrasts, and potentially informative predictors, but they do not replace a defined ecological hypothesis or multivariable model.

## 6. Evaluate candidate models

Begin with a biologically justified set of candidate predictors:

```python
candidate_predictors = [
    "elevation",
    "slope",
    "ndvi",
    "distance_to_water",
]
```

Compare single-predictor models:

```python
single_predictor_models = compare_single_predictors(
    sampled,
    predictors=candidate_predictors,
)

single_predictor_models[
    [
        "predictor",
        "AIC",
        "delta_aic",
        "akaike_weight",
        "pseudo_r2",
        "converged",
    ]
]
```

Small candidate combinations can then be evaluated:

```python
candidate_models = evaluate_linear_candidates_up_to_k(
    sampled,
    predictor_cols=candidate_predictors,
    max_k=3,
    n_jobs=4,
)

candidate_models.head(10)
```

```{warning}
The number of possible models grows combinatorially. Candidate predictors, transformations, and interactions should be constrained by ecological reasoning before model fitting. AIC ranking quantifies relative support within the supplied candidate set; it does not discover causal structure automatically.
```

## 7. Specify and fit the RSF

`FeatureSpec` defines the model matrix explicitly. Continuous predictors are standardized during fitting, while quadratic terms and interactions are constructed on the standardized scale.

```python
spec = FeatureSpec(
    linear=[
        "distance_to_water",
        "ndvi",
        "slope",
    ],
    quadratic=[
        "distance_to_water",
    ],
    interactions=[
        ("ndvi", "slope"),
    ],
    add_const=True,
)
```

Fit the used–available logistic regression:

```python
model, scaler, fitted_spec, meta = fit_rsf(
    sampled,
    spec,
)

print(model.summary())
```

For continuous predictors, positive coefficients indicate increasing relative selection with increasing predictor values, conditional on the remaining terms. Negative coefficients indicate decreasing relative selection.

Because the model is fitted to a researcher-defined ratio of used and available locations, its intercept and raw logistic probabilities generally do not represent absolute occurrence probabilities. The primary inferential quantities are relative selection coefficients and contrasts in relative selection strength.

## 8. Predict the selection surface

Project the fitted model over the environmental raster stack:

```python
rsf = predict_rsf_surface(
    env,
    model,
    scaler,
    fitted_spec,
    meta,
)

rsf
```

The output is an `xarray.DataArray` containing the relative selection surface:

```python
rsf.sel(band="rsf").plot(
    robust=True,
    figsize=(9, 7),
)
```

The surface is proportional to

[
w(\mathbf{x}) = \exp!\left(
\beta_0 + \boldsymbol{\beta}^{\mathsf T}\mathbf{x}
\right),
]

and should therefore be interpreted comparatively: a location with a score of two is estimated to have twice the relative selection strength of a location with a score of one, under the fitted model and availability design.

## 9. Validate among individuals

Randomly splitting autocorrelated relocations can produce optimistic validation results. For multi-individual datasets, hrHSA therefore supports leave-one-individual-out validation.

```python
(
    cv_summary,
    cv_parameters,
    boyce_bins,
    calibration_bins,
    cv_diagnostics,
) = leave_one_individual_out_rsf(
    reloc,
    env,
    fitted_spec,
    id_col="Individual_ID",
    heldout="all",
    domain_quantile=0.95,
    thin_train_dt="12h",
    sampling_factor_train=10,
    n_background_boyce=100_000,
    seed=42,
)

cv_summary
```

In each fold, the model is trained on all but one individual and evaluated against the relocations and availability domain of the held-out individual. This tests whether population-level selection relationships transfer to animals not used during model fitting.

Plot the fold-specific Boyce curves:

```python
fig, ax = plot_boyce_curves(
    boyce_bins,
    id_col="heldout_ID",
)
```

The Boyce diagnostic evaluates whether observed locations become progressively more frequent than expected under random use as predicted relative selection increases. It should be interpreted alongside fold-specific coefficients, calibration diagnostics, sample sizes, and the ecological structure of the held-out individuals.

## A hierarchical Bayesian extension

The frequentist workflow estimates a common population-level selection relationship. When repeated observations from multiple individuals are available, a hierarchical model can estimate both population-level effects and among-individual variation.

The following abbreviated PyMC model gives each individual its own intercept and slope while partially pooling those effects toward the population mean.

```python
import pymc as pm

predictor = "distance_to_water"

bayes_df = sampled[
    ["used", "Individual_ID", predictor]
].dropna().copy()

x = bayes_df[predictor].to_numpy(dtype=float)
x = (x - x.mean()) / x.std()

y = bayes_df["used"].to_numpy(dtype=int)

id_idx, individual_names = pd.factorize(
    bayes_df["Individual_ID"],
    sort=True,
)

coords = {
    "obs": np.arange(len(bayes_df)),
    "individual": individual_names.tolist(),
}
```

```python
with pm.Model(coords=coords) as hierarchical_rsf:

    # Data containers
    x_data = pm.Data("x", x, dims="obs")
    id_data = pm.Data("id_idx", id_idx, dims="obs")
    y_data = pm.Data("y", y, dims="obs")

    # Population-level intercept and slope
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.5)
    beta = pm.Normal("beta", mu=0.0, sigma=1.0)

    # Among-individual variation
    sigma_alpha = pm.Exponential("sigma_alpha", lam=1.0)
    sigma_beta = pm.Exponential("sigma_beta", lam=1.0)

    # Standardized individual deviations
    z_alpha = pm.Normal(
        "z_alpha",
        mu=0.0,
        sigma=1.0,
        dims="individual",
    )
    z_beta = pm.Normal(
        "z_beta",
        mu=0.0,
        sigma=1.0,
        dims="individual",
    )

    # Non-centered random effects
    alpha_ind = pm.Deterministic(
        "alpha_ind",
        z_alpha * sigma_alpha,
        dims="individual",
    )
    beta_ind = pm.Deterministic(
        "beta_ind",
        z_beta * sigma_beta,
        dims="individual",
    )

    # Total individual effects
    alpha_total = pm.Deterministic(
        "alpha_total",
        alpha + alpha_ind,
        dims="individual",
    )
    beta_total = pm.Deterministic(
        "beta_total",
        beta + beta_ind,
        dims="individual",
    )

    # Linear predictor
    eta = (
        alpha_total[id_data]
        + beta_total[id_data] * x_data
    )

    # Used–available likelihood
    used = pm.Bernoulli(
        "used",
        logit_p=eta,
        observed=y_data,
        dims="obs",
    )

    inference = pm.sample(
        draws=2_000,
        tune=2_000,
        chains=4,
        target_accept=0.95,
        random_seed=42,
    )
```

The population slope `beta` describes the average selection response, while `sigma_beta` quantifies the degree to which that response varies among individuals. Partial pooling regularizes poorly sampled individuals and propagates uncertainty through population- and individual-level estimates.

As in the frequentist used–available model, the intercept depends on the availability-sampling design. Individual slopes and their population distribution are generally the principal ecological targets.

```{note}
The hierarchical model is currently presented as an optional extension rather than a stabilized hrHSA interface. A future Bayesian module will connect design-matrix construction, prior specification, posterior diagnostics, spatial prediction, and posterior predictive validation within the package workflow.
```
