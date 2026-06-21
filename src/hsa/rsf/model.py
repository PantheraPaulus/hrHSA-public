from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from hsa.features import build_design_matrix
from hsa.types import FeatureSpec


def fit_rsf(
    df: pd.DataFrame,
    spec: FeatureSpec,
    *,
    min_available_proportion: float = 0.0,
    clean: bool = True,
) -> tuple[sm.discrete.discrete_model.BinaryResultsWrapper, StandardScaler, FeatureSpec, dict[str, Any]]:
    """Fit a logistic RSF from used/available samples."""

    if "used" not in df.columns:
        raise KeyError("fit_rsf requires a boolean or 0/1 'used' column.")

    df = df.reset_index(drop=True).copy()
    if clean:
        candidate_cols = ["used", *spec.linear, *spec.categorical]
        candidate_cols = [col for col in candidate_cols if col in df.columns]
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=candidate_cols)

    x, scaler, meta = build_design_matrix(
        df,
        spec,
        scaler=None,
        fit_scaler=True,
        min_available_proportion=min_available_proportion,
        meta=None,
    )
    y = df.loc[x.index, "used"].astype(int)
    model = sm.Logit(y, x).fit(disp=False)
    return model, scaler, spec, meta


def predict_rsf_points(
    df: pd.DataFrame,
    model,
    scaler: StandardScaler,
    spec: FeatureSpec,
    meta: dict[str, Any],
    *,
    pred_col: str = "rsf_pred",
) -> pd.DataFrame:
    """Predict relative RSF values for a point-level dataframe."""

    x, _, _ = build_design_matrix(df, spec, scaler=scaler, fit_scaler=False, meta=meta)
    x = x[model.params.index]
    eta = model.predict(x, which="linear")

    out = df.loc[x.index].copy()
    out[pred_col] = np.exp(eta).to_numpy()
    return out
