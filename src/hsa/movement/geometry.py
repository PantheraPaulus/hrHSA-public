from __future__ import annotations

import numpy as np
import pandas as pd


# TODO: Estimate decorrelation time 
def prepare_trajectory_data(
    df: pd.DataFrame,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    round_freq: str | None = "h",
    drop_duplicate_fixes: bool = True,
) -> pd.DataFrame:
    """Prepare relocation records for step and turn-angle calculations."""

    g = df.copy()
    missing = [col for col in (id_col, timestamp_col) if col not in g.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    g[timestamp_col] = pd.to_datetime(g[timestamp_col], errors="coerce")
    g = g.dropna(subset=[id_col, timestamp_col])

    if round_freq is not None:
        g["time_rounded"] = g[timestamp_col].dt.floor(round_freq)
        if drop_duplicate_fixes:
            g = g.drop_duplicates([id_col, "time_rounded"], keep="first")

    return g.sort_values([id_col, timestamp_col]).reset_index(drop=True)


def build_step_data(
    reloc_gdf,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    expected_interval_min: float | None = None,
    tolerance_min: float = 2,
):
    """Calculate step lengths, time differences, and speeds from a projected GeoDataFrame."""

    g = reloc_gdf.sort_values([id_col, timestamp_col]).reset_index(drop=True).copy()
    g["previous_timestamp"] = g.groupby(id_col)[timestamp_col].shift(1)
    g["previous_location"] = g.groupby(id_col)["geometry"].shift(1)
    g["t_diff_h"] = (g[timestamp_col] - g["previous_timestamp"]).dt.total_seconds() / 3600.0
    g["step_m"] = g.geometry.distance(g["previous_location"])
    g = g.loc[g["t_diff_h"].notna() & (g["t_diff_h"] > 0)].copy()
    g["speed_kmh"] = (g["step_m"] / 1000.0) / g["t_diff_h"]
    g["t_diff_min"] = g["t_diff_h"] * 60.0

    if expected_interval_min is not None:
        lower = expected_interval_min - tolerance_min
        upper = expected_interval_min + tolerance_min
        g = g.loc[g["t_diff_min"].between(lower, upper, inclusive="both")].copy()

    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["step_m", "t_diff_h"])
    g = g.loc[g["step_m"] > 0].copy()
    if g.empty:
        raise ValueError("No valid steps remaining after interval filtering.")
    return g


def build_turn_angle_data(
    step_df,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    expected_interval_min: float | None = None,
    tolerance_min: float = 2,
):
    """Calculate turning angles from consecutive steps."""

    g = step_df.sort_values([id_col, timestamp_col]).reset_index(drop=True).copy()
    valid_ids = g.groupby(id_col).size().loc[lambda s: s >= 3].index
    g = g.loc[g[id_col].isin(valid_ids)].copy()
    if g.empty:
        raise ValueError("No tracks with at least 3 fixes available for turning-angle calculation.")

    g["next_timestamp"] = g.groupby(id_col)[timestamp_col].shift(-1)
    g["next_position"] = g.groupby(id_col)["geometry"].shift(-1)
    g["t_diff_next_min"] = (g["next_timestamp"] - g[timestamp_col]).dt.total_seconds() / 60.0

    if expected_interval_min is not None:
        lower = expected_interval_min - tolerance_min
        upper = expected_interval_min + tolerance_min
        g = g.loc[g["t_diff_next_min"].between(lower, upper, inclusive="both")].copy()

    g["x"] = g.geometry.x
    g["y"] = g.geometry.y
    g["x_prev"] = g["previous_location"].x
    g["y_prev"] = g["previous_location"].y
    g["x_next"] = g["next_position"].x
    g["y_next"] = g["next_position"].y

    heading_in = np.arctan2(g["y"] - g["y_prev"], g["x"] - g["x_prev"])
    heading_out = np.arctan2(g["y_next"] - g["y"], g["x_next"] - g["x"])
    g["turn_angle"] = (heading_out - heading_in + np.pi) % (2 * np.pi) - np.pi

    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["turn_angle"])
    if g.empty:
        raise ValueError("No valid turning angles remaining after filtering.")
    return g
