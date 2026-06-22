from __future__ import annotations

import numpy as np
import pandas as pd


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


def build_displacement_velocity_data(
    reloc_gdf,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    expected_interval_min: float | None = None,
    tolerance_min: float = 2,
):
    """Build a step table with displacement and velocity vector components.

    The input must be a projected GeoDataFrame, so x/y units are metric. Each
    returned row represents the movement from the previous fix to the current
    fix for the same individual.

    Added columns include:
    - ``x``, ``y``: current coordinates
    - ``x_prev``, ``y_prev``: previous coordinates
    - ``dx_m``, ``dy_m``: displacement vector components in metres
    - ``vx_m_per_h``, ``vy_m_per_h``: velocity vector components
    - ``speed_m_per_h``: scalar speed
    - ``heading``: movement direction in radians
    """

    g = build_step_data(
        reloc_gdf,
        id_col=id_col,
        timestamp_col=timestamp_col,
        expected_interval_min=expected_interval_min,
        tolerance_min=tolerance_min,
    ).copy()

    g["x"] = g.geometry.x
    g["y"] = g.geometry.y
    g["x_prev"] = g["previous_location"].x
    g["y_prev"] = g["previous_location"].y

    g["dx_m"] = g["x"] - g["x_prev"]
    g["dy_m"] = g["y"] - g["y_prev"]
    g["vx_m_per_h"] = g["dx_m"] / g["t_diff_h"]
    g["vy_m_per_h"] = g["dy_m"] / g["t_diff_h"]
    g["speed_m_per_h"] = np.hypot(g["vx_m_per_h"], g["vy_m_per_h"])
    g["heading"] = np.arctan2(g["dy_m"], g["dx_m"])

    g = g.replace([np.inf, -np.inf], np.nan)
    g = g.dropna(subset=["dx_m", "dy_m", "vx_m_per_h", "vy_m_per_h", "heading"])
    if g.empty:
        raise ValueError("No valid displacement/velocity vectors available.")
    return g


def build_lagged_vector_pairs(
    vector_df: pd.DataFrame,
    *,
    id_col: str = "Individual_ID",
    timestamp_col: str = "Timestamp",
    vector_cols: tuple[str, str] = ("vx_m_per_h", "vy_m_per_h"),
    max_lag: str | pd.Timedelta = "48h",
    lag_bin: str | pd.Timedelta = "1h",
) -> pd.DataFrame:
    """Create lagged vector pairs for empirical autocorrelation estimation.

    Each returned row pairs one movement vector with a later movement vector
    from the same individual. The dot product and cosine similarity are included
    so downstream code can estimate how vector similarity declines with time lag.

    This is intentionally a dataframe builder, not the final decorrelation-time
    estimator. It belongs in ``geometry`` because it is purely geometric and
    temporal preprocessing.
    """

    required = [id_col, timestamp_col, *vector_cols]
    missing = [col for col in required if col not in vector_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    max_lag_td = pd.Timedelta(max_lag)
    lag_bin_td = pd.Timedelta(lag_bin)
    if max_lag_td <= pd.Timedelta(0):
        raise ValueError("max_lag must be positive.")
    if lag_bin_td <= pd.Timedelta(0):
        raise ValueError("lag_bin must be positive.")

    vx, vy = vector_cols
    rows = []
    for animal_id, group in vector_df.sort_values([id_col, timestamp_col]).groupby(id_col):
        g = group.reset_index(drop=True).copy()
        times = pd.to_datetime(g[timestamp_col]).to_numpy(dtype="datetime64[ns]")
        vec = g[[vx, vy]].to_numpy(dtype=float)
        norms = np.linalg.norm(vec, axis=1)

        for i in range(len(g) - 1):
            dt = pd.to_timedelta(times[i + 1 :] - times[i])
            within = np.where(dt <= max_lag_td)[0]
            if within.size == 0:
                continue

            for offset in within:
                j = i + 1 + int(offset)
                norm_product = norms[i] * norms[j]
                dot = float(np.dot(vec[i], vec[j]))
                cosine = dot / norm_product if norm_product > 0 else np.nan
                lag = pd.Timedelta(times[j] - times[i])
                lag_bin_value = pd.to_timedelta(
                    np.floor(lag / lag_bin_td) * lag_bin_td,
                    unit="ns",
                )

                rows.append(
                    {
                        id_col: animal_id,
                        "t0": pd.Timestamp(times[i]),
                        "t1": pd.Timestamp(times[j]),
                        "lag": lag,
                        "lag_h": lag.total_seconds() / 3600.0,
                        "lag_bin": lag_bin_value,
                        "lag_bin_h": lag_bin_value.total_seconds() / 3600.0,
                        f"{vx}_0": vec[i, 0],
                        f"{vy}_0": vec[i, 1],
                        f"{vx}_1": vec[j, 0],
                        f"{vy}_1": vec[j, 1],
                        "dot_product": dot,
                        "cosine_similarity": cosine,
                        "speed_0": norms[i],
                        "speed_1": norms[j],
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No lagged vector pairs found. Increase max_lag or check timestamps.")
    return out


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
