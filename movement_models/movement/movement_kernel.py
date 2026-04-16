import numpy as np
import pandas as pd
import scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import vonmises

from movement_models.io import to_reloc_gdf_projected

from .movement_geometry import (
    prepare_trajectory_data,
    build_step_data,
    build_turn_angle_data,
)

from .movement_mle import (
    _aic,
    _exp_nll,
    _gamma_nll,
    _weibull_nll,
    _lognorm_nll,
    _fit_model,
    _vm_uniform_nll,
    _vonmises_nll,
)

from .movement_mh import (
    _compute_waic,
    fit_exp_distribution_bayes,
    fit_gamma_distribution_bayes,
    fit_weibull_distribution_bayes,
    fit_lognorm_distribution_bayes,
    fit_vonmises_distribution_bayes,
    fit_vm_uniform_distribution_bayes,
)

from .movement_hmc import (
    fit_exp_distribution_hmc,
    fit_gamma_distribution_hmc,
    fit_weibull_distribution_hmc,
    fit_lognorm_distribution_hmc,
    fit_vonmises_distribution_hmc,
    fit_vm_uniform_distribution_hmc,
)

def fit_step_distribution(
    steps,
    *,
    method="mle",
    bayes_model=None,
    hmc_model=None,
    selection_method="auto",
    cutoff=np.inf,
    **kwargs,
):
    steps = pd.Series(steps).dropna()
    steps = steps[(steps > 0) & (steps <= cutoff)].to_numpy()

    if len(steps) == 0:
        raise ValueError("No positive step lengths remaining after filtering.")

    bayes_kwargs = {
        k.removeprefix("bayes_"): v
        for k, v in kwargs.items()
        if k.startswith("bayes_")
    }

    hmc_kwargs = {
        k.removeprefix("hmc_"): v
        for k, v in kwargs.items()
        if k.startswith("hmc_")
    }

    if method == "bayes":
        if bayes_model is not None:
            if bayes_model == "exp":
                return fit_exp_distribution_bayes(steps, cutoff=cutoff, **bayes_kwargs)

            elif bayes_model == "gamma":
                return fit_gamma_distribution_bayes(steps, cutoff=cutoff, **bayes_kwargs)

            elif bayes_model == "weibull":
                return fit_weibull_distribution_bayes(steps, cutoff=cutoff, **bayes_kwargs)

            elif bayes_model == "lognorm":
                return fit_lognorm_distribution_bayes(steps, cutoff=cutoff, **bayes_kwargs)

            else:
                raise ValueError(
                    "For method='bayes', bayes_model must be one of: "
                    "'exp', 'gamma', 'weibull', 'lognorm', or None."
                )

        model_results = [
            fit_exp_distribution_bayes(steps, cutoff=cutoff, **bayes_kwargs),
            fit_gamma_distribution_bayes(steps, cutoff=cutoff, **bayes_kwargs),
            fit_weibull_distribution_bayes(steps, cutoff=cutoff, **bayes_kwargs),
            fit_lognorm_distribution_bayes(steps, cutoff=cutoff, **bayes_kwargs),
        ]

        model_table = pd.DataFrame([
            {
                "distribution": res["distribution"],
                "waic": res["waic"],
            }
            for res in model_results
        ]).sort_values("waic").reset_index(drop=True)

        winner_name = model_table.iloc[0]["distribution"]
        winner = next(res for res in model_results if res["distribution"] == winner_name)

        winner["model_table"] = model_table
        return winner

    elif method == "hmc":
        if hmc_model is not None:
            if hmc_model == "exp":
                return fit_exp_distribution_hmc(steps, cutoff=cutoff, **hmc_kwargs)

            elif hmc_model == "gamma":
                return fit_gamma_distribution_hmc(steps, cutoff=cutoff, **hmc_kwargs)

            elif hmc_model == "weibull":
                return fit_weibull_distribution_hmc(steps, cutoff=cutoff, **hmc_kwargs)

            elif hmc_model == "lognorm":
                return fit_lognorm_distribution_hmc(steps, cutoff=cutoff, **hmc_kwargs)

            else:
                raise ValueError(
                    "For method='hmc', hmc_model must be one of: "
                    "'exp', 'gamma', 'weibull', 'lognorm', or None."
                )

        model_results = [
            fit_exp_distribution_hmc(steps, cutoff=cutoff, **hmc_kwargs),
            fit_gamma_distribution_hmc(steps, cutoff=cutoff, **hmc_kwargs),
            fit_weibull_distribution_hmc(steps, cutoff=cutoff, **hmc_kwargs),
            fit_lognorm_distribution_hmc(steps, cutoff=cutoff, **hmc_kwargs),
        ]

        model_table = pd.DataFrame([
            {
                "distribution": res["distribution"],
                "waic": res["waic"],
            }
            for res in model_results
        ]).sort_values("waic").reset_index(drop=True)

        winner_name = model_table.iloc[0]["distribution"]
        winner = next(res for res in model_results if res["distribution"] == winner_name)

        winner["model_table"] = model_table
        return winner

    elif method == "mle":
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

    else:
        raise ValueError("method must be 'mle', 'bayes', or 'hmc'")


def fit_turn_angle_distribution(
    angles,
    *,
    method="mle",
    bayes_model=None,
    hmc_model=None,
    selection_method="auto",
    **kwargs,
):
    angles = pd.Series(angles).dropna().to_numpy()

    if len(angles) == 0:
        raise ValueError("No turning angles available after filtering.")

    bayes_kwargs = {
        k.removeprefix("bayes_"): v
        for k, v in kwargs.items()
        if k.startswith("bayes_")
    }

    hmc_kwargs = {
        k.removeprefix("hmc_"): v
        for k, v in kwargs.items()
        if k.startswith("hmc_")
    }

    if method == "bayes":
        if bayes_model == "vonmises":
            return fit_vonmises_distribution_bayes(angles, **bayes_kwargs)

        elif bayes_model == "vm_uniform":
            return fit_vm_uniform_distribution_bayes(angles, **bayes_kwargs)

        elif bayes_model is None:
            model_results = [
                fit_vonmises_distribution_bayes(angles, **bayes_kwargs),
                fit_vm_uniform_distribution_bayes(angles, **bayes_kwargs),
            ]

            model_table = pd.DataFrame([
                {
                    "distribution": res["distribution"],
                    "waic": res["waic"],
                }
                for res in model_results
            ]).sort_values("waic").reset_index(drop=True)

            winner_name = model_table.iloc[0]["distribution"]
            winner = next(res for res in model_results if res["distribution"] == winner_name)

            winner["model_table"] = model_table
            return winner

        else:
            raise ValueError(
                "For method='bayes', bayes_model must be one of: "
                "'vonmises', 'vm_uniform', or None."
            )

    elif method == "hmc":
        if hmc_model == "vonmises":
            return fit_vonmises_distribution_hmc(angles, **hmc_kwargs)

        elif hmc_model == "vm_uniform":
            return fit_vm_uniform_distribution_hmc(angles, **hmc_kwargs)

        elif hmc_model is None:
            model_results = [
                fit_vonmises_distribution_hmc(angles, **hmc_kwargs),
                fit_vm_uniform_distribution_hmc(angles, **hmc_kwargs),
            ]

            model_table = pd.DataFrame([
                {
                    "distribution": res["distribution"],
                    "waic": res["waic"],
                }
                for res in model_results
            ]).sort_values("waic").reset_index(drop=True)

            winner_name = model_table.iloc[0]["distribution"]
            winner = next(res for res in model_results if res["distribution"] == winner_name)

            winner["model_table"] = model_table
            return winner

        else:
            raise ValueError(
                "For method='hmc', hmc_model must be one of: "
                "'vonmises', 'vm_uniform', or None."
            )

    elif method == "mle":
        res_vm = minimize(
            _vonmises_nll,
            x0=[1.0, 0.0],
            args=(angles,),
            bounds=[(1e-6, None), (-np.pi, np.pi)],
        )

        if not res_vm.success:
            raise RuntimeError(f"Von Mises fit failed: {res_vm.message}")

        kappa_vm, mu_vm = res_vm.x

        res_mix = minimize(
            _vm_uniform_nll,
            x0=[0.5, 0.5],
            args=(angles,),
            bounds=[(1e-6, None), (0, 1)],
        )

        if not res_mix.success:
            raise RuntimeError(f"Mixture fit failed: {res_mix.message}")

        kappa_mix, w_mix = res_mix.x

        return {
            "n": len(angles),
            "distribution": "angle_mle",
            "vonmises_kappa": kappa_vm,
            "vonmises_mu": mu_vm,
            "mixture_kappa": kappa_mix,
            "mixture_w": w_mix,
        }

    else:
        raise ValueError("method must be 'mle', 'bayes', or 'hmc'")


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
    step_method="mle",
    step_bayes_model=None,
    step_hmc_model=None,
    angle_method="mle",
    angle_bayes_model=None,
    angle_hmc_model=None,
    **kwargs,
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

    step_backend_kwargs = {
        k.removeprefix("step_"): v
        for k, v in kwargs.items()
        if k.startswith("step_")
    }

    angle_backend_kwargs = {
        k.removeprefix("angle_"): v
        for k, v in kwargs.items()
        if k.startswith("angle_")
    }

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

        step_fit = fit_step_distribution(
            step_subset,
            method=step_method,
            bayes_model=step_bayes_model,
            hmc_model=step_hmc_model,
            cutoff=step_cutoff,
            **step_backend_kwargs,
        )

        angle_fit = fit_turn_angle_distribution(
            angle_subset,
            method=angle_method,
            bayes_model=angle_bayes_model,
            hmc_model=angle_hmc_model,
            **angle_backend_kwargs,
        )

        row = {
            id_col: animal_id,
            "n_steps": step_fit["n"],
            "step_distribution": step_fit["distribution"],
            "step_q25": step_fit["q25"],
            "step_median": step_fit["median"],
            "step_mean": step_fit["mean"],
            "step_q75": step_fit["q75"],
            "step_max": step_fit["max"],
            "n_angles": angle_fit["n"],
            "angle_distribution": angle_fit["distribution"],
        }

        if "params" in step_fit:
            row["step_params"] = step_fit["params"]

        if "waic" in step_fit:
            row["step_waic"] = step_fit["waic"]

        if "acceptance_rate" in step_fit:
            row["step_acceptance_rate"] = step_fit["acceptance_rate"]

        if "waic" in angle_fit:
            row["angle_waic"] = angle_fit["waic"]

        if "posterior_mean" in step_fit:
            row["step_posterior_mean"] = step_fit["posterior_mean"]
            row["step_posterior_median"] = step_fit["posterior_median"]
            row["step_posterior_q025"] = step_fit["posterior_q025"]
            row["step_posterior_q975"] = step_fit["posterior_q975"]

        if "posterior_mean_shape" in step_fit:
            row["step_posterior_mean_shape"] = step_fit["posterior_mean_shape"]
            row["step_posterior_median_shape"] = step_fit["posterior_median_shape"]
            row["step_posterior_q025_shape"] = step_fit["posterior_q025_shape"]
            row["step_posterior_q975_shape"] = step_fit["posterior_q975_shape"]

        if "posterior_mean_scale" in step_fit:
            row["step_posterior_mean_scale"] = step_fit["posterior_mean_scale"]
            row["step_posterior_median_scale"] = step_fit["posterior_median_scale"]
            row["step_posterior_q025_scale"] = step_fit["posterior_q025_scale"]
            row["step_posterior_q975_scale"] = step_fit["posterior_q975_scale"]

        if "posterior_mean_mu" in step_fit:
            row["step_posterior_mean_mu"] = step_fit["posterior_mean_mu"]
            row["step_posterior_median_mu"] = step_fit["posterior_median_mu"]
            row["step_posterior_q025_mu"] = step_fit["posterior_q025_mu"]
            row["step_posterior_q975_mu"] = step_fit["posterior_q975_mu"]

        if "posterior_mean_sigma" in step_fit:
            row["step_posterior_mean_sigma"] = step_fit["posterior_mean_sigma"]
            row["step_posterior_median_sigma"] = step_fit["posterior_median_sigma"]
            row["step_posterior_q025_sigma"] = step_fit["posterior_q025_sigma"]
            row["step_posterior_q975_sigma"] = step_fit["posterior_q975_sigma"]

        if "posterior_mean_params" in step_fit:
            row["step_posterior_mean_params"] = step_fit["posterior_mean_params"]
        if "posterior_median_params" in step_fit:
            row["step_posterior_median_params"] = step_fit["posterior_median_params"]
        if "posterior_q025_params" in step_fit:
            row["step_posterior_q025_params"] = step_fit["posterior_q025_params"]
        if "posterior_q975_params" in step_fit:
            row["step_posterior_q975_params"] = step_fit["posterior_q975_params"]

        if "vonmises_kappa" in angle_fit:
            row["vonmises_kappa"] = angle_fit["vonmises_kappa"]

        if "vonmises_mu" in angle_fit:
            row["vonmises_mu"] = angle_fit["vonmises_mu"]

        if "mixture_kappa" in angle_fit:
            row["mixture_kappa"] = angle_fit["mixture_kappa"]

        if "mixture_w" in angle_fit:
            row["mixture_w"] = angle_fit["mixture_w"]

        if "acceptance_rate" in angle_fit:
            row["angle_acceptance_rate"] = angle_fit["acceptance_rate"]

        if "posterior_mean_kappa" in angle_fit:
            row["angle_posterior_mean_kappa"] = angle_fit["posterior_mean_kappa"]
            row["angle_posterior_median_kappa"] = angle_fit["posterior_median_kappa"]
            row["angle_posterior_q025_kappa"] = angle_fit["posterior_q025_kappa"]
            row["angle_posterior_q975_kappa"] = angle_fit["posterior_q975_kappa"]

        if "posterior_mean_mu" in angle_fit:
            row["angle_posterior_mean_mu"] = angle_fit["posterior_mean_mu"]
            row["angle_posterior_median_mu"] = angle_fit["posterior_median_mu"]
            row["angle_posterior_q025_mu"] = angle_fit["posterior_q025_mu"]
            row["angle_posterior_q975_mu"] = angle_fit["posterior_q975_mu"]

        if "posterior_mean_w" in angle_fit:
            row["angle_posterior_mean_w"] = angle_fit["posterior_mean_w"]
            row["angle_posterior_median_w"] = angle_fit["posterior_median_w"]
            row["angle_posterior_q025_w"] = angle_fit["posterior_q025_w"]
            row["angle_posterior_q975_w"] = angle_fit["posterior_q975_w"]

        if "posterior_mean_params" in angle_fit:
            row["angle_posterior_mean_params"] = angle_fit["posterior_mean_params"]
        if "posterior_median_params" in angle_fit:
            row["angle_posterior_median_params"] = angle_fit["posterior_median_params"]
        if "posterior_q025_params" in angle_fit:
            row["angle_posterior_q025_params"] = angle_fit["posterior_q025_params"]
        if "posterior_q975_params" in angle_fit:
            row["angle_posterior_q975_params"] = angle_fit["posterior_q975_params"]

        rows.append(row)

    return {
        "reloc_gdf": reloc_gdf,
        "step_df": step_df,
        "angle_df": angle_df,
        "summary": pd.DataFrame(rows),
    }