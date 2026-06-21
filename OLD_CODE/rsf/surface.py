from __future__ import annotations

import numpy as np
import xarray as xr
from sklearn.preprocessing import StandardScaler

from ..types import FeatureSpec

def get_rsf_surface(env, res, scaler, spec: FeatureSpec, *, crs=None):

    base = list(spec.linear)
    env = env.sel(band=base)
    coef = res.params

    eta = xr.zeros_like(env.sel(band=base[0])) + float(coef.get("const", 0.0))

    z = {}
    for i, v in enumerate(base):
        layer = env.sel(band=v)
        z[v] = (layer - float(scaler.mean_[i])) / float(scaler.scale_[i])

        if v in coef.index:
            eta = eta + z[v] * float(coef[v])

    if spec.quadratic:
        for v in spec.quadratic:
            name = f"{v}__sq"
            if name in coef.index:
                eta = eta + (z[v] ** 2) * float(coef[name])

    if spec.interactions:
        for a, b in spec.interactions:
            name = f"{a}__x__{b}"
            if name in coef.index:
                eta = eta + (z[a] * z[b]) * float(coef[name])

    rsf = np.exp(eta).rename("rsf").expand_dims(band=["rsf"])

    try:
        import rioxarray  # noqa: F401

        if crs is not None:
            rsf = rsf.rio.write_crs(crs)
        else:
            env_crs = getattr(getattr(env, "rio", None), "crs", None)
            if env_crs is not None:
                rsf = rsf.rio.write_crs(env_crs)
    except Exception:
        pass

    return rsf