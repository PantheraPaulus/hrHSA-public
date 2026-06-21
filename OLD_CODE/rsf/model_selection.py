import itertools
import pandas as pd
import xarray as xr

from OLD_CODE.types import FeatureSpec
from OLD_CODE.rsf.model import fit_rsf

def eval_all_linear_candidates(df: pd.DataFrame, env: xr.DataArray, subset: list = None):

    if subset is None:
        bands = env.band.values
    else:
        bands = subset
        
    rows = []
    for i in range(1, len(bands)+1):
        for combo in itertools.combinations(bands, i):
            linear = list(combo)
            spec = FeatureSpec(linear = linear, add_const = True)
            res, _, _ = fit_rsf(df, spec)
            rows.append({
                "Variables": linear,
                "AIC": res.aic,
                "BIC": res.bic
            })

    return pd.DataFrame(rows)