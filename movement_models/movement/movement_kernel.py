import numpy as np
import pandas as pd
import scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import vonmises

from movement_models.io import to_reloc_gdf_projected

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

def _aic(loglik, k):
    return 2 * k - 2 * loglik

def _exp_nll(params, x):
    scale = params[0]
    if scale <= 0:
        return np.inf
    return -np.sum(stats.expon.logpdf(x, loc=0, scale=scale)) 

def _gamma_nll(params, x):
    shape, scale = params
    if shape <= 0 or scale <= 0:
        return np.inf
    return -np.sum(stats.gamma.logpdf(x, shape, loc=0, scale=scale))

def _weibull_nll(params, x):
    c, scale = params
    if c <= 0 or scale <= 0:
        return np.inf
    return -np.sum(stats.weibull_min.logpdf(x, c, loc=0, scale=scale))

def _lognorm_nll(params, x):
    s, scale = params
    if s <= 0 or scale <= 0:
        return np.inf
    return -np.sum(stats.lognorm.logpdf(x, s, loc=0, scale=scale))

def _fit_model(nll_func, x, init, bounds, name):
    res = minimize(nll_func, init, args=(x,), bounds=bounds)

    if not res.success:
        return {
            "distribution": name,
            "params": None,
            "loglik": -np.inf,
            "AIC": np.inf,
            "success": False,
        }

    loglik = -res.fun
    k = len(res.x)

    return {
        "distribution": name,
        "params": res.x,
        "loglik": loglik,
        "AIC": _aic(loglik, k),
        "success": True,
    }

def fit_step_distribution(steps, *, cutoff=np.inf):
    steps = pd.Series(steps).dropna()
    steps = steps[(steps > 0) & (steps <= cutoff)].to_numpy()

    if len(steps) == 0:
        raise ValueError("No positive step lengths remaining after filtering.")

    models = []

    models.append(_fit_model(
        _exp_nll,
        steps,
        init=[np.mean(steps)],
        bounds=[(1e-6, None)],
        name="exp"
    ))

    models.append(_fit_model(
        _gamma_nll,
        steps,
        init=[1.0, np.mean(steps)],
        bounds=[(1e-6, None), (1e-6, None)],
        name="gamma"
    ))

    models.append(_fit_model(
        _weibull_nll,
        steps,
        init=[1.0, np.mean(steps)],
        bounds=[(1e-6, None), (1e-6, None)],
        name="weibull"
    ))

    models.append(_fit_model(
        _lognorm_nll,
        steps,
        init=[1.0, np.mean(steps)],
        bounds=[(1e-6, None), (1e-6, None)],
        name="lognorm"
    ))

    results = pd.DataFrame(models).sort_values("AIC").reset_index(drop=True)
    winner = results.iloc[0]

    return {
        "n": len(steps),
        "q25": np.nanquantile(steps, 0.25),
        "median": np.nanmedian(steps),
        "mean": np.nanmean(steps),
        "q75": np.nanquantile(steps, 0.75),
        "max": np.nanmax(steps),
        "distribution": winner["distribution"],
        "params": winner["params"],
        "model_table": results,
    }

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

def _vm_uniform_nll(params, angles):
    kappa, w = params

    if kappa < 0 or not (0 <= w <= 1):
        return np.inf

    vm = vonmises.pdf(angles, kappa, loc=0)
    uniform = 1 / (2 * np.pi)

    mixture = w * vm + (1 - w) * uniform
    mixture = np.clip(mixture, 1e-12, None)

    return -np.sum(np.log(mixture))

def _vonmises_nll(params, angles):
    kappa, mu = params

    if kappa < 0:
        return np.inf

    return -np.sum(vonmises.logpdf(angles, kappa, loc=mu))

def fit_turn_angle_distribution(angles):
    angles = pd.Series(angles).dropna().to_numpy()

    if len(angles) == 0:
        raise ValueError("No turning angles available after filtering.")

    res_vm = minimize(
        _vonmises_nll,
        x0=[1.0, 0.0],
        args=(angles,),
        bounds=[(1e-6, None), (-np.pi, np.pi)],
    )

    if not res_vm.success:
        raise RuntimeError(f"Von Mises fit failed: {res_vm.message}")

    kappa_vm, mu_vm = res_vm.x

    init = [0.5, 0.5]
    res = minimize(
        _vm_uniform_nll,
        init,
        args=(angles,),
        bounds=[(1e-6, None), (0, 1)],
    )

    if not res.success:
        raise RuntimeError(f"Mixture fit failed: {res.message}")

    kappa_mix, w_mix = res.x

    return {
        "n": len(angles),
        "vonmises_kappa": kappa_vm,
        "vonmises_mu": mu_vm,
        "mixture_kappa": kappa_mix,
        "mixture_w": w_mix,
    }

def fit_movement_kernel_per_id(
    df,
    *,
    id_col="ID",
    timestamp_col="Timestamp",
    lon_col="Longitude",
    lat_col="Latitude",
    input_crs="EPSG:4326",
    target_crs="EPSG:29333",
    round_freq="h",
    drop_duplicate_fixes=True,
    expected_interval_min=120,
    tolerance_min=2,
    step_cutoff=np.inf,
):
    reloc = prepare_trajectory_data(
        df,
        id_col=id_col,
        timestamp_col=timestamp_col,
        round_freq=round_freq,
        drop_duplicate_fixes=drop_duplicate_fixes,
    )

    required_cols = [id_col, timestamp_col, lon_col, lat_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    reloc_gdf = to_reloc_gdf_projected(
        reloc,
        lon_col=lon_col,
        lat_col=lat_col,
        timestamp_col=timestamp_col,
        input_crs=input_crs,
        target_crs=target_crs,
    )

    step_df = build_step_data(
        reloc_gdf,
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
        step_subset = step_df.loc[step_df[id_col] == animal_id, "step_m"]
        angle_subset = angle_df.loc[angle_df[id_col] == animal_id, "turn_angle"]

        if len(step_subset.dropna()) == 0 or len(angle_subset.dropna()) == 0:
            continue

        step_fit = fit_step_distribution(step_subset, cutoff=step_cutoff)
        angle_fit = fit_turn_angle_distribution(angle_subset)

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
                "vonmises_kappa": angle_fit["vonmises_kappa"],
                "mixture_kappa": angle_fit["mixture_kappa"],
                "mixture_w": angle_fit["mixture_w"],
            }
        )

    return {
        "reloc_gdf": reloc_gdf,
        "step_df": step_df,
        "angle_df": angle_df,
        "summary": pd.DataFrame(rows),
    }