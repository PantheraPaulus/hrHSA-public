"""Resource-selection-function tools."""

from hsa.rsf.model import fit_rsf, predict_rsf_points
from hsa.rsf.selection import (
    add_aic_weights,
    compare_rsf_specs,
    compare_single_predictors,
    evaluate_linear_candidates_up_to_k,
    evaluate_regex_candidate_family,
    has_duplicate_base_variables,
    select_best_scale_per_predictor,
    select_predictor_columns,
    split_multiscale_name,
    summarize_univariate_scale_selection,
    variable_frequency_summary,
)
from hsa.rsf.surface import predict_rsf_surface, predict_rsf_surface_multiscale
from hsa.rsf.validation import boyce_quantile_bins, boyce_sliding_window

__all__ = [
    "fit_rsf",
    "predict_rsf_points",
    "predict_rsf_surface",
    "predict_rsf_surface_multiscale",
    "boyce_quantile_bins",
    "boyce_sliding_window",
    "add_aic_weights",
    "compare_rsf_specs",
    "compare_single_predictors",
    "evaluate_linear_candidates_up_to_k",
    "evaluate_regex_candidate_family",
    "has_duplicate_base_variables",
    "select_best_scale_per_predictor",
    "select_predictor_columns",
    "split_multiscale_name",
    "summarize_univariate_scale_selection",
    "variable_frequency_summary",
]
