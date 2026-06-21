from __future__ import annotations

import itertools

import pandas as pd

from hsa.rsf.model import fit_rsf
from hsa.types import FeatureSpec


def evaluate_linear_candidates(
    df: pd.DataFrame,
    predictors: list[str],
    *,
    max_predictors: int = 3,
    min_available_proportion: float = 0.0,
) -> pd.DataFrame:
    """Evaluate constrained linear RSF candidate models by AIC/BIC.

    This is intentionally limited by ``max_predictors``. For ecological RSFs,
    prefer biologically defined candidate sets over exhaustive all-subsets
    searches across large predictor stacks.
    """

    rows = []
    max_k = min(max_predictors, len(predictors))
    for k in range(1, max_k + 1):
        for combo in itertools.combinations(predictors, k):
            spec = FeatureSpec(linear=list(combo), add_const=True)
            try:
                model, _, _, _ = fit_rsf(
                    df,
                    spec,
                    min_available_proportion=min_available_proportion,
                )
            except Exception as exc:
                rows.append({"predictors": list(combo), "n_predictors": k, "AIC": None, "BIC": None, "error": str(exc)})
                continue
            rows.append(
                {
                    "predictors": list(combo),
                    "n_predictors": k,
                    "AIC": float(model.aic),
                    "BIC": float(model.bic),
                    "error": None,
                }
            )
    return pd.DataFrame(rows).sort_values(["AIC", "BIC"], na_position="last").reset_index(drop=True)
