from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

from .types import FeatureSpec


def _build_design_matrix(
    df: pd.DataFrame,
    spec: FeatureSpec,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[pd.DataFrame, StandardScaler]:

    # Linear terms
    Xc = df[spec.linear].copy()

    if scaler is None:
        scaler = StandardScaler()

    X = scaler.fit_transform(Xc) if fit_scaler else scaler.transform(Xc)
    X = pd.DataFrame(X, columns=spec.linear, index=df.index)

    # Quadratic terms
    if spec.quadratic:
        for v in spec.quadratic:
            X[f"{v}__sq"] = X[v] ** 2

    # Interactions
    if spec.interactions:
        for a, b in spec.interactions:
            X[f"{a}__x__{b}"] = X[a] * X[b]

    # Intercept
    if spec.add_const:
        X = sm.add_constant(X, has_constant="add")

    return X, scaler