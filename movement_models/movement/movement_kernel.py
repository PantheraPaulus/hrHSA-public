import numpy as np
import pandas as pd
import scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import vonmises
from scipy.special import expit, logit

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

def _compute_waic(log_lik_samples):
    """
    log_lik_samples: array of shape (n_posterior_samples, n_observations)
    """
    log_lik_samples = np.asarray(log_lik_samples)

    lppd = np.sum(
        np.log(
            np.mean(np.exp(log_lik_samples), axis=0)
        )
    )
    p_waic = np.sum(np.var(log_lik_samples, axis=0))
    waic = -2 * (lppd - p_waic)

    return waic

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


def _exp_loglik(scale, x):
    if scale <= 0:
        return -np.inf
    return np.sum(stats.expon.logpdf(x, loc=0, scale=scale))


def _exp_logprior(scale, sigma=5000.0):
    if scale <= 0:
        return -np.inf
    return stats.halfnorm.logpdf(scale, loc=0, scale=sigma)


def _exp_logposterior(scale, x, sigma=5000.0):
    return _exp_loglik(scale, x) + _exp_logprior(scale, sigma=sigma)


def fit_exp_distribution_bayes(
    steps,
    *,
    cutoff=np.inf,
    n_iter=20000,
    init_scale=None,
    proposal_sd=0.08,
    prior_scale=5000.0,
    burn_in=2000,
):
    x = pd.Series(steps).dropna()
    x = x[x > 0].to_numpy()

    if len(x) == 0:
        raise ValueError("No positive step lengths available.")

    if init_scale is None:
        init_scale = np.mean(x)

    if init_scale <= 0:
        raise ValueError("Initial scale must be positive.")

    samples = np.empty(n_iter)
    accepted = 0

    current = init_scale
    current_lp = _exp_logposterior(current, x, sigma=prior_scale)

    for i in range(n_iter):
        proposal = np.exp(np.random.normal(np.log(current), proposal_sd))
        proposal_lp = _exp_logposterior(proposal, x, sigma=prior_scale)

        log_alpha = proposal_lp - current_lp

        if np.log(np.random.rand()) < log_alpha:
            current = proposal
            current_lp = proposal_lp
            accepted += 1

        samples[i] = current

    kept = samples[burn_in:]

    log_lik_samples = np.array([
        stats.expon.logpdf(x, loc=0, scale=scale_sample)
        for scale_sample in kept
    ])
    
    waic = _compute_waic(log_lik_samples)

    return {
        "n": len(x),
        "q25": np.nanquantile(x, 0.25),
        "median": np.nanmedian(x),
        "mean": np.nanmean(x),
        "q75": np.nanquantile(x, 0.75),
        "max": np.nanmax(x),
        "distribution": "exp_bayes",
        "posterior_mean_params": (np.mean(kept),),
        "posterior_median_params": (np.median(kept),),
        "posterior_q025_params": (np.quantile(kept, 0.025),),
        "posterior_q975_params": (np.quantile(kept, 0.975),),
        "acceptance_rate": accepted / n_iter,
        "init_scale": init_scale,
        "proposal_sd": proposal_sd,
        "prior_scale": prior_scale,
        "samples": samples,
        "posterior_samples": kept,
        "log_lik_samples": log_lik_samples,
        "waic": waic
    }

def _gamma_loglik(shape, scale, x):
    if shape <= 0 or scale <= 0:
        return -np.inf
    return np.sum(stats.gamma.logpdf(x, a=shape, loc=0, scale=scale))


def _gamma_logprior(shape, scale, shape_prior=2.0, scale_prior=5000.0):
    if shape <= 0 or scale <= 0:
        return -np.inf
    lp_shape = stats.gamma.logpdf(shape, a=2.0, loc=0, scale=shape_prior)
    lp_scale = stats.halfnorm.logpdf(scale, loc=0, scale=scale_prior)
    return lp_shape + lp_scale


def _gamma_logposterior(
    shape,
    scale,
    x,
    *,
    shape_prior=2.0,
    scale_prior=5000.0,
):
    return _gamma_loglik(shape, scale, x) + _gamma_logprior(
        shape,
        scale,
        shape_prior=shape_prior,
        scale_prior=scale_prior,
    )


def fit_gamma_distribution_bayes(
    steps,
    *,
    cutoff=np.inf,
    n_iter=20000,
    init_params=None,
    proposal_sd=0.05,
    prior_scale=5000.0,
    burn_in=2000,
):
    x = pd.Series(steps).dropna()
    x = x[(x > 0) & (x <= cutoff)].to_numpy()

    if len(x) == 0:
        raise ValueError("No positive step lengths remaining after filtering.")

    if init_params is None:
        init_shape = 1.0
        init_scale = np.mean(x)
    else:
        init_shape, init_scale = init_params
    
    if init_shape <= 0 or init_scale <= 0:
        raise ValueError("Initial gamma parameters must be positive.")

    samples = np.empty((n_iter, 2))
    accepted = 0

    current_shape = init_shape
    current_scale = init_scale
    current_lp = _gamma_logposterior(
        current_shape,
        current_scale,
        x,
        shape_prior=2.0,
        scale_prior=prior_scale,
    )

    for i in range(n_iter):
        proposal_shape = np.exp(
            np.random.normal(np.log(current_shape), proposal_sd)
        )
        proposal_scale = np.exp(
            np.random.normal(np.log(current_scale), proposal_sd)
        )

        proposal_lp = _gamma_logposterior(
            proposal_shape,
            proposal_scale,
            x,
            shape_prior=2.0,
            scale_prior=prior_scale,
        )

        log_alpha = proposal_lp - current_lp

        if np.log(np.random.rand()) < log_alpha:
            current_shape = proposal_shape
            current_scale = proposal_scale
            current_lp = proposal_lp
            accepted += 1

        samples[i, 0] = current_shape
        samples[i, 1] = current_scale

    kept = samples[burn_in:]

    shape_samples = kept[:, 0]
    scale_samples = kept[:, 1]

    log_lik_samples = np.array([
        stats.gamma.logpdf(x, a=shape_sample, loc=0, scale=scale_sample)
        for shape_sample, scale_sample in kept
    ])

    waic = _compute_waic(log_lik_samples)

    return {
        "n": len(x),
        "q25": np.nanquantile(x, 0.25),
        "median": np.nanmedian(x),
        "mean": np.nanmean(x),
        "q75": np.nanquantile(x, 0.75),
        "max": np.nanmax(x),
        "distribution": "gamma_bayes",
        "posterior_mean_params": (np.mean(shape_samples), np.mean(scale_samples)),
        "posterior_median_params": (np.median(shape_samples), np.median(scale_samples)),
        "posterior_q025_params": (
            np.quantile(shape_samples, 0.025),
            np.quantile(scale_samples, 0.025),
        ),
        "posterior_q975_params": (
            np.quantile(shape_samples, 0.975),
            np.quantile(scale_samples, 0.975),
        ),
        "acceptance_rate": accepted / n_iter,
        "init_params": (init_shape, init_scale),
        "proposal_sd": proposal_sd,
        "prior_scale": prior_scale,
        "samples": samples,
        "posterior_samples": kept,
        "log_lik_samples": log_lik_samples,
        "waic": waic
    }

def _weibull_loglik(shape, scale, x):
    if shape <= 0 or scale <= 0:
        return -np.inf
    return np.sum(stats.weibull_min.logpdf(x, c=shape, loc=0, scale=scale))


def _weibull_logprior(shape, scale, shape_prior=2.0, scale_prior=5000.0):
    if shape <= 0 or scale <= 0:
        return -np.inf
    lp_shape = stats.gamma.logpdf(shape, a=2.0, loc=0, scale=shape_prior)
    lp_scale = stats.halfnorm.logpdf(scale, loc=0, scale=scale_prior)
    return lp_shape + lp_scale


def _weibull_logposterior(
    shape,
    scale,
    x,
    *,
    shape_prior=2.0,
    scale_prior=5000.0,
):
    return _weibull_loglik(shape, scale, x) + _weibull_logprior(
        shape,
        scale,
        shape_prior=shape_prior,
        scale_prior=scale_prior,
    )


def fit_weibull_distribution_bayes(
    steps,
    *,
    cutoff=np.inf,
    n_iter=20000,
    init_params=None,
    proposal_sd=0.05,
    prior_scale=5000.0,
    burn_in=2000,
):
    x = pd.Series(steps).dropna()
    x = x[(x > 0) & (x <= cutoff)].to_numpy()
    
    if len(x) == 0:
        raise ValueError("No positive step lengths remaining after filtering.")
    
    if init_params is None:
        init_shape = 1.0
        init_scale = np.mean(x)
    else:
        init_shape, init_scale = init_params
    
    if init_shape <= 0 or init_scale <= 0:
        raise ValueError("Initial Weibull parameters must be positive.")

    samples = np.empty((n_iter, 2))
    accepted = 0

    current_shape = init_shape
    current_scale = init_scale
    current_lp = _weibull_logposterior(
        current_shape,
        current_scale,
        x,
        shape_prior=2.0,
        scale_prior=prior_scale,
    )

    for i in range(n_iter):
        proposal_shape = np.exp(
            np.random.normal(np.log(current_shape), proposal_sd)
        )
        proposal_scale = np.exp(
            np.random.normal(np.log(current_scale), proposal_sd)
        )

        proposal_lp = _weibull_logposterior(
            proposal_shape,
            proposal_scale,
            x,
            shape_prior=2.0,
            scale_prior=prior_scale,
        )

        log_alpha = proposal_lp - current_lp

        if np.log(np.random.rand()) < log_alpha:
            current_shape = proposal_shape
            current_scale = proposal_scale
            current_lp = proposal_lp
            accepted += 1

        samples[i, 0] = current_shape
        samples[i, 1] = current_scale

    kept = samples[burn_in:]

    shape_samples = kept[:, 0]
    scale_samples = kept[:, 1]

    log_lik_samples = np.array([
        stats.weibull_min.logpdf(x, c=shape_sample, loc=0, scale=scale_sample)
        for shape_sample, scale_sample in kept
    ])
    
    waic = _compute_waic(log_lik_samples)

    return {
        "n": len(x),
        "q25": np.nanquantile(x, 0.25),
        "median": np.nanmedian(x),
        "mean": np.nanmean(x),
        "q75": np.nanquantile(x, 0.75),
        "max": np.nanmax(x),
        "distribution": "weibull_bayes",
        "posterior_mean_params": (np.mean(shape_samples), np.mean(scale_samples)),
        "posterior_median_params": (np.median(shape_samples), np.median(scale_samples)),
        "posterior_q025_params": (
            np.quantile(shape_samples, 0.025),
            np.quantile(scale_samples, 0.025),
        ),
        "posterior_q975_params": (
            np.quantile(shape_samples, 0.975),
            np.quantile(scale_samples, 0.975),
        ),
        "acceptance_rate": accepted / n_iter,
        "init_params": (init_shape, init_scale),
        "proposal_sd": proposal_sd,
        "prior_scale": prior_scale,
        "samples": samples,
        "posterior_samples": kept,
        "log_lik_samples": log_lik_samples,
        "waic": waic
    }

def _lognorm_loglik(mu, sigma, x):
    if sigma <= 0:
        return -np.inf
    return np.sum(stats.lognorm.logpdf(x, s=sigma, loc=0, scale=np.exp(mu)))


def _lognorm_logprior(mu, sigma, mu_prior=0.0, sigma_prior=10.0):
    if sigma <= 0:
        return -np.inf
    lp_mu = stats.norm.logpdf(mu, loc=mu_prior, scale=10.0)
    lp_sigma = stats.halfnorm.logpdf(sigma, loc=0, scale=sigma_prior)
    return lp_mu + lp_sigma


def _lognorm_logposterior(
    mu,
    sigma,
    x,
    *,
    mu_prior=0.0,
    sigma_prior=10.0,
):
    return _lognorm_loglik(mu, sigma, x) + _lognorm_logprior(
        mu,
        sigma,
        mu_prior=mu_prior,
        sigma_prior=sigma_prior,
    )


def fit_lognorm_distribution_bayes(
    steps,
    *,
    cutoff=np.inf,
    n_iter=20000,
    init_params=None,
    proposal_sd=0.05,
    prior_scale=10.0,
    burn_in=2000,
):
    x = pd.Series(steps).dropna()
    x = x[(x > 0) & (x <= cutoff)].to_numpy()
    
    if len(x) == 0:
        raise ValueError("No positive step lengths remaining after filtering.")
    
    logx = np.log(x)
    
    if init_params is None:
        init_mu = np.mean(logx)
        init_sigma = 1.0
    else:
        init_mu, init_sigma = init_params
    
    if init_sigma <= 0:
        raise ValueError("Initial sigma must be positive.")

    samples = np.empty((n_iter, 2))
    accepted = 0

    current_mu = init_mu
    current_sigma = init_sigma

    current_lp = _lognorm_logposterior(
        current_mu,
        current_sigma,
        x,
        mu_prior=0.0,
        sigma_prior=prior_scale,
    )

    for i in range(n_iter):
        proposal_mu = np.random.normal(current_mu, proposal_sd)
        proposal_sigma = np.exp(
            np.random.normal(np.log(current_sigma), proposal_sd)
        )

        proposal_lp = _lognorm_logposterior(
            proposal_mu,
            proposal_sigma,
            x,
            mu_prior=0.0,
            sigma_prior=prior_scale,
        )

        log_alpha = proposal_lp - current_lp

        if np.log(np.random.rand()) < log_alpha:
            current_mu = proposal_mu
            current_sigma = proposal_sigma
            current_lp = proposal_lp
            accepted += 1

        samples[i, 0] = current_mu
        samples[i, 1] = current_sigma

    kept = samples[burn_in:]

    mu_samples = kept[:, 0]
    sigma_samples = kept[:, 1]

    log_lik_samples = np.array([
        stats.lognorm.logpdf(x, s=sigma_sample, loc=0, scale=np.exp(mu_sample))
        for mu_sample, sigma_sample in kept
    ])
    
    waic = _compute_waic(log_lik_samples)

    return {
        "n": len(x),
        "q25": np.nanquantile(x, 0.25),
        "median": np.nanmedian(x),
        "mean": np.nanmean(x),
        "q75": np.nanquantile(x, 0.75),
        "max": np.nanmax(x),
        "distribution": "lognorm_bayes",
        "posterior_mean_params": (np.mean(mu_samples), np.mean(sigma_samples)),
        "posterior_median_params": (np.median(mu_samples), np.median(sigma_samples)),
        "posterior_q025_params": (
            np.quantile(mu_samples, 0.025),
            np.quantile(sigma_samples, 0.025),
        ),
        "posterior_q975_params": (
            np.quantile(mu_samples, 0.975),
            np.quantile(sigma_samples, 0.975),
        ),
        "acceptance_rate": accepted / n_iter,
        "init_params": (init_mu, init_sigma),
        "proposal_sd": proposal_sd,
        "prior_scale": prior_scale,
        "samples": samples,
        "posterior_samples": kept,
        "log_lik_samples": log_lik_samples,
        "waic": waic
    }



def fit_step_distribution(
    steps,
    *,
    method="mle",
    bayes_model=None,
    cutoff=np.inf,
    **bayes_kwargs,
):
    steps = pd.Series(steps).dropna()
    steps = steps[(steps > 0) & (steps <= cutoff)].to_numpy()

    if len(steps) == 0:
        raise ValueError("No positive step lengths remaining after filtering.")

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
        raise ValueError("method must be 'mle' or 'bayes'")


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


def _vonmises_loglik(kappa, mu, angles):
    if kappa <= 0:
        return -np.inf
    return np.sum(vonmises.logpdf(angles, kappa, loc=mu))


def _vonmises_logprior(kappa, mu, prior_scale=10.0):
    if kappa <= 0:
        return -np.inf

    lp_kappa = stats.halfnorm.logpdf(kappa, loc=0, scale=prior_scale)
    lp_mu = stats.uniform.logpdf(mu, loc=-np.pi, scale=2 * np.pi)

    return lp_kappa + lp_mu


def _vonmises_logposterior(kappa, mu, angles, prior_scale=10.0):
    return _vonmises_loglik(kappa, mu, angles) + _vonmises_logprior(
        kappa, mu, prior_scale=prior_scale
    )


def fit_vonmises_distribution_bayes(
    angles,
    *,
    n_iter=20000,
    init_params=None,
    proposal_sd=0.5,
    prior_scale=10.0,
    burn_in=2000,
):
    angles = pd.Series(angles).dropna().to_numpy()

    if len(angles) == 0:
        raise ValueError("No turning angles available after filtering.")

    if init_params is None:
        init_kappa = 1.0
        init_mu = 0.0
    else:
        init_kappa, init_mu = init_params

    if init_kappa <= 0:
        raise ValueError("Initial kappa must be positive.")

    samples = np.empty((n_iter, 2))
    accepted = 0

    current_kappa = init_kappa
    current_mu = init_mu

    current_lp = _vonmises_logposterior(
        current_kappa,
        current_mu,
        angles,
        prior_scale=prior_scale,
    )

    for i in range(n_iter):
        proposal_kappa = np.exp(
            np.random.normal(np.log(current_kappa), proposal_sd)
        )
        proposal_mu = np.random.normal(current_mu, proposal_sd)
        proposal_mu = ((proposal_mu + np.pi) % (2 * np.pi)) - np.pi

        proposal_lp = _vonmises_logposterior(
            proposal_kappa,
            proposal_mu,
            angles,
            prior_scale=prior_scale,
        )

        log_alpha = proposal_lp - current_lp

        if np.log(np.random.rand()) < log_alpha:
            current_kappa = proposal_kappa
            current_mu = proposal_mu
            current_lp = proposal_lp
            accepted += 1

        samples[i, 0] = current_kappa
        samples[i, 1] = current_mu

    kept = samples[burn_in:]

    kappa_samples = kept[:, 0]
    mu_samples = kept[:, 1]

    return {
        "n": len(angles),
        "distribution": "vonmises_bayes",
        "posterior_mean_kappa": np.mean(kappa_samples),
        "posterior_median_kappa": np.median(kappa_samples),
        "posterior_q025_kappa": np.quantile(kappa_samples, 0.025),
        "posterior_q975_kappa": np.quantile(kappa_samples, 0.975),
        "posterior_mean_mu": np.mean(mu_samples),
        "posterior_median_mu": np.median(mu_samples),
        "posterior_q025_mu": np.quantile(mu_samples, 0.025),
        "posterior_q975_mu": np.quantile(mu_samples, 0.975),
        "posterior_mean_params": (np.mean(kappa_samples), np.mean(mu_samples)),
        "posterior_median_params": (np.median(kappa_samples), np.median(mu_samples)),
        "posterior_q025_params": (
            np.quantile(kappa_samples, 0.025),
            np.quantile(mu_samples, 0.025),
        ),
        "posterior_q975_params": (
            np.quantile(kappa_samples, 0.975),
            np.quantile(mu_samples, 0.975),
        ),
        "acceptance_rate": accepted / n_iter,
        "init_params": (init_kappa, init_mu),
        "proposal_sd": proposal_sd,
        "prior_scale": prior_scale,
        "samples": samples,
        "posterior_samples": kept,
    }

def _vm_uniform_loglik(kappa, w, angles):
    if kappa <= 0 or not (0 <= w <= 1):
        return -np.inf

    vm = vonmises.pdf(angles, kappa, loc=0)
    uniform = 1 / (2 * np.pi)

    mixture = w * vm + (1 - w) * uniform
    mixture = np.clip(mixture, 1e-12, None)

    return np.sum(np.log(mixture))


def _vm_uniform_logprior(kappa, w, prior_scale=10.0):
    if kappa <= 0 or not (0 <= w <= 1):
        return -np.inf

    lp_kappa = stats.halfnorm.logpdf(kappa, loc=0, scale=prior_scale)
    lp_w = stats.beta.logpdf(w, a=1.0, b=1.0)

    return lp_kappa + lp_w


def _vm_uniform_logposterior(kappa, w, angles, prior_scale=10.0):
    return _vm_uniform_loglik(kappa, w, angles) + _vm_uniform_logprior(
        kappa, w, prior_scale=prior_scale
    )


def fit_vm_uniform_distribution_bayes(
    angles,
    *,
    n_iter=20000,
    init_params=None,
    proposal_sd=0.3,
    prior_scale=10.0,
    burn_in=2000,
):
    angles = pd.Series(angles).dropna().to_numpy()

    if len(angles) == 0:
        raise ValueError("No turning angles available after filtering.")

    if init_params is None:
        init_kappa = 1.0
        init_w = 0.5
    else:
        init_kappa, init_w = init_params

    if init_kappa <= 0:
        raise ValueError("Initial kappa must be positive.")
    if not (0 <= init_w <= 1):
        raise ValueError("Initial mixture weight w must be in [0, 1].")

    samples = np.empty((n_iter, 2))
    accepted = 0

    current_kappa = init_kappa
    current_w = init_w

    current_lp = _vm_uniform_logposterior(
        current_kappa,
        current_w,
        angles,
        prior_scale=prior_scale,
    )

    for i in range(n_iter):
        proposal_kappa = np.exp(
            np.random.normal(np.log(current_kappa), proposal_sd)
        )

        proposal_w = expit(
            np.random.normal(logit(current_w), proposal_sd)
        )
        proposal_w = np.clip(proposal_w, 1e-6, 1 - 1e-6)

        proposal_lp = _vm_uniform_logposterior(
            proposal_kappa,
            proposal_w,
            angles,
            prior_scale=prior_scale,
        )

        log_alpha = proposal_lp - current_lp

        if np.log(np.random.rand()) < log_alpha:
            current_kappa = proposal_kappa
            current_w = proposal_w
            current_lp = proposal_lp
            accepted += 1

        samples[i, 0] = current_kappa
        samples[i, 1] = current_w

    kept = samples[burn_in:]

    kappa_samples = kept[:, 0]
    w_samples = kept[:, 1]

    return {
        "n": len(angles),
        "distribution": "vm_uniform_bayes",
        "posterior_mean_kappa": np.mean(kappa_samples),
        "posterior_median_kappa": np.median(kappa_samples),
        "posterior_q025_kappa": np.quantile(kappa_samples, 0.025),
        "posterior_q975_kappa": np.quantile(kappa_samples, 0.975),
        "posterior_mean_w": np.mean(w_samples),
        "posterior_median_w": np.median(w_samples),
        "posterior_q025_w": np.quantile(w_samples, 0.025),
        "posterior_q975_w": np.quantile(w_samples, 0.975),
        "posterior_mean_params": (np.mean(kappa_samples), np.mean(w_samples)),
        "posterior_median_params": (np.median(kappa_samples), np.median(w_samples)),
        "posterior_q025_params": (
            np.quantile(kappa_samples, 0.025),
            np.quantile(w_samples, 0.025),
        ),
        "posterior_q975_params": (
            np.quantile(kappa_samples, 0.975),
            np.quantile(w_samples, 0.975),
        ),
        "acceptance_rate": accepted / n_iter,
        "init_params": (init_kappa, init_w),
        "proposal_sd": proposal_sd,
        "prior_scale": prior_scale,
        "samples": samples,
        "posterior_samples": kept,
    }    

    

def fit_turn_angle_distribution(
    angles,
    *,
    method="mle",
    bayes_model=None,
    **bayes_kwargs,
):
    angles = pd.Series(angles).dropna().to_numpy()

    if len(angles) == 0:
        raise ValueError("No turning angles available after filtering.")

    if method == "bayes":
        if bayes_model == "vonmises":
            return fit_vonmises_distribution_bayes(angles, **bayes_kwargs)

        elif bayes_model == "vm_uniform":
            return fit_vm_uniform_distribution_bayes(angles, **bayes_kwargs)

        else:
            raise ValueError(
                "For method='bayes', bayes_model must be one of: "
                "'vonmises', 'vm_uniform'."
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
        raise ValueError("method must be 'mle' or 'bayes'")


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
    angle_method="mle",
    angle_bayes_model=None,
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

    step_bayes_kwargs = {
        k.removeprefix("step_"): v
        for k, v in kwargs.items()
        if k.startswith("step_")
    }
    
    angle_bayes_kwargs = {
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
            cutoff=step_cutoff,
            **step_bayes_kwargs,
        )
        angle_fit = fit_turn_angle_distribution(
            angle_subset,
            method=angle_method,
            bayes_model=angle_bayes_model,
            **angle_bayes_kwargs,
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

        rows.append(row)

    return {
        "reloc_gdf": reloc_gdf,
        "step_df": step_df,
        "angle_df": angle_df,
        "summary": pd.DataFrame(rows),
    }