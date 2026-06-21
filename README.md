<p align="center">
  <img src="logo_rsf2.png" width="600">
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
└── simulation/          # future SSF / mechanistic movement simulation
```

## Development install

```bash
conda env create -f environment.yml
conda activate hsa
pip install -e .
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

## Examples

Project-specific workflows should live in `examples/`, especially the Okonjima pangolin RSF workflow. The package itself should stay general and should not contain local file paths, Earth Engine project names, reserve-specific exclusions, or project-specific predictor choices.
