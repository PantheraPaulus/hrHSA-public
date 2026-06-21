from __future__ import annotations

import pandas as pd

from hsa.movement.distributions import fit_step_distribution, fit_turn_angle_distribution
from hsa.movement.geometry import build_step_data, build_turn_angle_data, prepare_trajectory_data


def fit_movement_kernel_per_id(
    reloc_gdf,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    round_freq: str | None = "h",
    drop_duplicate_fixes: bool = True,
    expected_interval_min: float | None = None,
    tolerance_min: float = 2,
    step_cutoff: float = float("inf"),
) -> dict:
    """Fit step-length and turn-angle distributions per individual.

    ``reloc_gdf`` must already be a projected GeoDataFrame. CRS conversion is
    intentionally left outside this function so callers make the spatial unit
    explicit.
    """

    reloc = prepare_trajectory_data(
        reloc_gdf,
        id_col=id_col,
        timestamp_col=timestamp_col,
        round_freq=round_freq,
        drop_duplicate_fixes=drop_duplicate_fixes,
    )
    reloc = reloc_gdf.loc[reloc.index].copy() if "geometry" not in reloc.columns else reloc

    step_df = build_step_data(
        reloc,
        id_col=id_col,
        timestamp_col=timestamp_col,
        expected_interval_min=expected_interval_min,
        tolerance_min=tolerance_min,
    )
    angle_df = build_turn_angle_data(
        step_df,
        id_col=id_col,
        timestamp_col=timestamp_col,
        expected_interval_min=expected_interval_min,
        tolerance_min=tolerance_min,
    )

    rows = []
    for animal_id in step_df[id_col].dropna().unique():
        steps = step_df.loc[step_df[id_col] == animal_id, "step_m"]
        angles = angle_df.loc[angle_df[id_col] == animal_id, "turn_angle"]
        if steps.dropna().empty or angles.dropna().empty:
            continue

        step_fit = fit_step_distribution(steps, cutoff=step_cutoff)
        angle_fit = fit_turn_angle_distribution(angles)
        rows.append(
            {
                id_col: animal_id,
                "n_steps": step_fit["n"],
                "step_distribution": step_fit["distribution"],
                "step_params": step_fit["params"],
                "step_q25": step_fit["q25"],
                "step_median": step_fit["median"],
                "step_mean": step_fit["mean"],
                "step_q75": step_fit["q75"],
                "step_max": step_fit["max"],
                "n_angles": angle_fit["n"],
                "angle_distribution": angle_fit["distribution"],
                "angle_params": angle_fit["params"],
            }
        )

    return {"reloc_gdf": reloc, "step_df": step_df, "angle_df": angle_df, "summary": pd.DataFrame(rows)}
