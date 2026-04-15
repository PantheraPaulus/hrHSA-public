import numpy as np
import pandas as pd
import scipy.stats as stats

from scipy.stats import vonmises
from scipy.special import expit, logit

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

    log_lik_samples = np.array([
        vonmises.logpdf(angles, kappa_sample, loc=mu_sample)
        for kappa_sample, mu_sample in kept
    ])
    
    waic = _compute_waic(log_lik_samples)

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
        "log_lik_samples": log_lik_samples,
        "waic": waic
    }
    
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

    log_lik_samples = np.array([
        vonmises.logpdf(angles, kappa_sample, loc=mu_sample)
        for kappa_sample, mu_sample in kept
    ])
    
    waic = _compute_waic(log_lik_samples)

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
        "log_lik_samples": log_lik_samples,
        "waic": waic
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

    log_lik_samples = np.array([
        np.log(
            np.clip(
                w_sample * vonmises.pdf(angles, kappa_sample, loc=0) +
                (1 - w_sample) * (1 / (2 * np.pi)),
                1e-12,
                None,
            )
        )
        for kappa_sample, w_sample in kept
    ])

    waic = _compute_waic(log_lik_samples)

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
        "log_lik_samples": log_lik_samples,
        "waic": waic
    }