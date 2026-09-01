# hrHSA

**hrHSA** *(high-resolution Habitat Selection Analysis)* is a scalable Python framework for rigorous statistical inference on spatially distributed observations. It integrates positional data with environmental covariates to quantify how heterogeneous spatial fields, individual variation, and changing conditions shape the probability distribution of observed locations. The framework combines resource-selection analysis, spatial point-process models, hierarchical Bayesian inference, and mechanistic simulation, with explicit treatment of availability, observation processes, uncertainty, validation, and prediction.

Developed primarily for wildlife telemetry and spatial ecology, **hrHSA** addresses the more general problem of inferring interactions between actively and/or passively redistributing entities and their spatial context. Its methods therefore extend naturally to applications in geospatial science, epidemiology, environmental modelling, human mobility, and other domains concerned with spatially structured processes. Reproducible, vectorized, out-of-core, and HPC-ready workflows support analyses from covariate extraction and parameter estimation to predictive simulation.

```{toctree}
:maxdepth: 1
:caption: Documentation
getting_started
theory
implementation
case-studies
performance
```
