import numpy as np
import pandas as pd

def prepare_trajectory_data(
    df,
    *,
    id_col="ID",
    timestamp_col="Timestamp",
    round_freq="h",
    drop_duplicate_fixes=True,
):
    df = df.copy()

    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    if id_col not in df.columns:
        raise ValueError(f"Missing ID column: {id_col}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[id_col, timestamp_col])

    if round_freq is not None:
        df["time_rounded"] = df[timestamp_col].dt.floor(round_freq)

        if drop_duplicate_fixes:
            df = df.drop_duplicates([id_col, "time_rounded"], keep="first")

    df = df.sort_values([id_col, timestamp_col]).reset_index(drop=True)
    return df

def build_step_data(
    reloc_gdf,
    *,
    id_col="ID",
    timestamp_col="Timestamp",
    expected_interval_min=120,
    tolerance_min=2,
):
    gdf = reloc_gdf.copy()
    gdf = gdf.sort_values([id_col, timestamp_col]).reset_index(drop=True)

    gdf["previous_timestamp"] = gdf.groupby(id_col)[timestamp_col].shift(1)
    gdf["t_diff_h"] = (
        (gdf[timestamp_col] - gdf["previous_timestamp"]).dt.total_seconds() / 3600.0
    )

    gdf["previous_location"] = gdf.groupby(id_col)["geometry"].shift(1)
    gdf["distance_previous_location_m"] = gdf.geometry.distance(gdf["previous_location"])

    gdf = gdf[gdf["t_diff_h"].notna()].copy()
    gdf = gdf[gdf["t_diff_h"] > 0].copy()

    if gdf.empty:
        raise ValueError("No valid positive time differences available for step calculation.")

    gdf["speed_kmh"] = (gdf["distance_previous_location_m"] / 1000.0) / gdf["t_diff_h"]
    gdf["t_diff_min"] = gdf["t_diff_h"] * 60.0

    lower = expected_interval_min - tolerance_min
    upper = expected_interval_min + tolerance_min

    step_df = gdf[
        gdf["t_diff_min"].between(lower, upper, inclusive="both")
    ].copy()

    step_df["step_m"] = step_df["distance_previous_location_m"]

    step_df = step_df.replace([np.inf, -np.inf], np.nan)
    step_df = step_df.dropna(subset=["step_m", "t_diff_h"])
    step_df = step_df[step_df["step_m"] > 0].copy()

    if step_df.empty:
        raise ValueError(
            "No valid steps remaining after interval filtering. "
            "Check expected_interval_min and tolerance_min."
        )

    return step_df 

def build_turn_angle_data(
    step_df,
    *,
    id_col="ID",
    timestamp_col="Timestamp",
    expected_interval_min=120,
    tolerance_min=2,
):
    df = step_df.copy()
    df = df.sort_values([id_col, timestamp_col]).reset_index(drop=True)

    track_sizes = df.groupby(id_col).size()
    valid_ids = track_sizes[track_sizes >= 3].index
    df = df[df[id_col].isin(valid_ids)].copy()

    if df.empty:
        raise ValueError("No tracks with at least 3 fixes available for turning-angle calculation.")

    df["t_next_fix"] = df.groupby(id_col)[timestamp_col].shift(-1)
    df["t_diff_next_fix_min"] = (
        (df["t_next_fix"] - df[timestamp_col]).dt.total_seconds() / 60.0
    )
    df["next_position"] = df.groupby(id_col)["geometry"].shift(-1)

    lower = expected_interval_min - tolerance_min
    upper = expected_interval_min + tolerance_min

    angle_df = df[
        df["t_diff_next_fix_min"].between(lower, upper, inclusive="both")
    ].copy()

    angle_df["x"] = angle_df.geometry.x
    angle_df["y"] = angle_df.geometry.y

    angle_df["x_prev"] = angle_df["previous_location"].x
    angle_df["y_prev"] = angle_df["previous_location"].y

    angle_df["x_next"] = angle_df["next_position"].x
    angle_df["y_next"] = angle_df["next_position"].y

    angle_df["heading_in"] = np.arctan2(
        angle_df["y"] - angle_df["y_prev"],
        angle_df["x"] - angle_df["x_prev"],
    )

    angle_df["heading_out"] = np.arctan2(
        angle_df["y_next"] - angle_df["y"],
        angle_df["x_next"] - angle_df["x"],
    )

    angle_df["turn_angle"] = angle_df["heading_out"] - angle_df["heading_in"]
    angle_df["turn_angle"] = (angle_df["turn_angle"] + np.pi) % (2 * np.pi) - np.pi

    angle_df = angle_df.replace([np.inf, -np.inf], np.nan)
    angle_df = angle_df.dropna(subset=["turn_angle"])

    if angle_df.empty:
        raise ValueError(
            "No valid turning angles remaining after next-fix filtering. "
            "Check expected_interval_min and tolerance_min."
        )

    return angle_df