from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from ..types import FeatureSpec
from ..design_matrix import _build_design_matrix


def fit_rsf(df: pd.DataFrame, spec: FeatureSpec, *, clean: bool = True):
    cols = ["used"] + spec.linear

    if clean:
        df = (
            df.replace([np.inf, -np.inf], np.nan)
              .dropna(subset=cols)
        )

    y = df["used"].astype(int)
    Xz, scaler = _build_design_matrix(df, spec, scaler=None, fit_scaler=True)

    m = sm.Logit(y, Xz).fit(disp=False)
    return m, scaler, spec


def predict_rsf_points(df, m, scaler, spec):
    Xz, _ = _build_design_matrix(df, spec, scaler=scaler, fit_scaler=False)
    eta = m.predict(Xz, which = "linear")
    out = df.copy()
    out["rsf_pred"] = np.exp(eta)      

    return out