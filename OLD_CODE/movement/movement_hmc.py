import numpy as np
import pandas as pd

import jax.numpy as jnp
from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def _compute_waic(log_lik_samples):
    log_lik_samples = np.asarray(log_lik_samples)

    lppd = np.sum(
        np.log(
            np.mean(np.exp(log_lik_samples), axis=0)
        )
    )
    p_waic = np.sum(np.var(log_lik_samples, axis=0))
    waic = -2 * (lppd - p_waic)

    return waic

def _exp_step_model(steps):
    rate = numpyro.sample("rate", dist.HalfNormal(1.0))
    numpyro.sample("obs", dist.Exponential(rate), obs=steps)

def fit_exp_distribution_hmc(
    steps,
    *,
    cutoff=np.inf,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    rng_seed=0,
):
    x = pd.Series(steps).dropna()
    x = x[(x > 0) & (x <= cutoff)].to_numpy()

    if len(x) == 0:
        raise ValueError("No positive step lengths remaining after filtering.")

    x_jax = jnp.asarray(x)

    kernel = NUTS(_exp_step_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(random.PRNGKey(rng_seed), steps=x_jax)

    samples = mcmc.get_samples()
    rate_samples = np.asarray(samples["rate"])
    scale_samples = 1.0 / rate_samples

    log_lik_samples = np.array([
        dist.Exponential(rate_sample).log_prob(x_jax)
        for rate_sample in rate_samples
    ])
    
    waic = _compute_waic(log_lik_samples)

    return {
        "n": len(x),
        "q25": np.nanquantile(x, 0.25),
        "median": np.nanmedian(x),
        "mean": np.nanmean(x),
        "q75": np.nanquantile(x, 0.75),
        "max": np.nanmax(x),
        "distribution": "exp_hmc",
        "posterior_mean_params": (np.mean(scale_samples),),
        "posterior_median_params": (np.median(scale_samples),),
        "posterior_q025_params": (np.quantile(scale_samples, 0.025),),
        "posterior_q975_params": (np.quantile(scale_samples, 0.975),),
        "samples": samples,
        "rate_samples": rate_samples,
        "scale_samples": scale_samples,
        "mcmc": mcmc,
        "log_lik_samples": log_lik_samples,
        "waic": waic
    }

def _gamma_step_model(steps):
    concentration = numpyro.sample("concentration", dist.HalfNormal(5.0))
    rate = numpyro.sample("rate", dist.HalfNormal(1.0))
    numpyro.sample("obs", dist.Gamma(concentration=concentration, rate=rate), obs=steps)

def fit_gamma_distribution_hmc(
    steps,
    *,
    cutoff=np.inf,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    rng_seed=0,
):
    x = pd.Series(steps).dropna()
    x = x[(x > 0) & (x <= cutoff)].to_numpy()

    if len(x) == 0:
        raise ValueError("No positive step lengths remaining after filtering.")

    x_jax = jnp.asarray(x)

    kernel = NUTS(_gamma_step_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(random.PRNGKey(rng_seed), steps=x_jax)

    samples = mcmc.get_samples()

    shape_samples = np.asarray(samples["concentration"])
    rate_samples = np.asarray(samples["rate"])
    scale_samples = 1.0 / rate_samples

    log_lik_samples = np.array([
        dist.Gamma(concentration=shape_sample, rate=rate_sample).log_prob(x_jax)
        for shape_sample, rate_sample in zip(shape_samples, rate_samples)
    ])
    
    waic = _compute_waic(log_lik_samples)

    return {
        "n": len(x),
        "q25": np.nanquantile(x, 0.25),
        "median": np.nanmedian(x),
        "mean": np.nanmean(x),
        "q75": np.nanquantile(x, 0.75),
        "max": np.nanmax(x),
        "distribution": "gamma_hmc",
        "posterior_mean_params": (
            np.mean(shape_samples),
            np.mean(scale_samples),
        ),
        "posterior_median_params": (
            np.median(shape_samples),
            np.median(scale_samples),
        ),
        "posterior_q025_params": (
            np.quantile(shape_samples, 0.025),
            np.quantile(scale_samples, 0.025),
        ),
        "posterior_q975_params": (
            np.quantile(shape_samples, 0.975),
            np.quantile(scale_samples, 0.975),
        ),
        "samples": samples,
        "shape_samples": shape_samples,
        "rate_samples": rate_samples,
        "scale_samples": scale_samples,
        "mcmc": mcmc,
        "log_lik_samples": log_lik_samples,
        "waic": waic
    }

def _weibull_step_model(steps):
    shape = numpyro.sample("shape", dist.HalfNormal(5.0))
    scale = numpyro.sample("scale", dist.HalfNormal(5000.0))
    numpyro.sample("obs", dist.Weibull(concentration=shape, scale=scale), obs=steps)

def fit_weibull_distribution_hmc(
    steps,
    *,
    cutoff=np.inf,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    rng_seed=0,
):
    x = pd.Series(steps).dropna()
    x = x[(x > 0) & (x <= cutoff)].to_numpy()

    if len(x) == 0:
        raise ValueError("No positive step lengths remaining after filtering.")

    x_jax = jnp.asarray(x)

    kernel = NUTS(_weibull_step_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(random.PRNGKey(rng_seed), steps=x_jax)

    samples = mcmc.get_samples()

    shape_samples = np.asarray(samples["shape"])
    scale_samples = np.asarray(samples["scale"])

    log_lik_samples = np.array([
        dist.Weibull(concentration=shape_sample, scale=scale_sample).log_prob(x_jax)
        for shape_sample, scale_sample in zip(shape_samples, scale_samples)
    ])

    waic = _compute_waic(log_lik_samples)

    return {
        "n": len(x),
        "q25": np.nanquantile(x, 0.25),
        "median": np.nanmedian(x),
        "mean": np.nanmean(x),
        "q75": np.nanquantile(x, 0.75),
        "max": np.nanmax(x),
        "distribution": "weibull_hmc",
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
        "samples": samples,
        "shape_samples": shape_samples,
        "scale_samples": scale_samples,
        "mcmc": mcmc,
        "log_lik_samples": log_lik_samples,
        "waic": waic
    }

def _lognorm_step_model(steps):
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(10.0))
    numpyro.sample("obs", dist.LogNormal(mu, sigma), obs=steps)

def fit_lognorm_distribution_hmc(
    steps,
    *,
    cutoff=np.inf,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    rng_seed=0,
):
    x = pd.Series(steps).dropna()
    x = x[(x > 0) & (x <= cutoff)].to_numpy()

    if len(x) == 0:
        raise ValueError("No positive step lengths remaining after filtering.")

    x_jax = jnp.asarray(x)

    kernel = NUTS(_lognorm_step_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(random.PRNGKey(rng_seed), steps=x_jax)

    samples = mcmc.get_samples()

    mu_samples = np.asarray(samples["mu"])
    sigma_samples = np.asarray(samples["sigma"])

    log_lik_samples = np.array([
        dist.LogNormal(mu_sample, sigma_sample).log_prob(x_jax)
        for mu_sample, sigma_sample in zip(mu_samples, sigma_samples)
    ])
    
    waic = _compute_waic(log_lik_samples)

    return {
        "n": len(x),
        "q25": np.nanquantile(x, 0.25),
        "median": np.nanmedian(x),
        "mean": np.nanmean(x),
        "q75": np.nanquantile(x, 0.75),
        "max": np.nanmax(x),
        "distribution": "lognorm_hmc",
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
        "samples": samples,
        "mu_samples": mu_samples,
        "sigma_samples": sigma_samples,
        "mcmc": mcmc,
        "log_lik_samples": log_lik_samples,
        "waic": waic
    }




def _vonmises_angle_model(angles):
    kappa = numpyro.sample("kappa", dist.HalfNormal(10.0))
    mu = numpyro.sample("mu", dist.Uniform(-jnp.pi, jnp.pi))
    numpyro.sample("obs", dist.VonMises(loc=mu, concentration=kappa), obs=angles)

def fit_vonmises_distribution_hmc(
    angles,
    *,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    rng_seed=0,
):
    x = pd.Series(angles).dropna().to_numpy()

    if len(x) == 0:
        raise ValueError("No turning angles available after filtering.")

    x_jax = jnp.asarray(x)

    kernel = NUTS(_vonmises_angle_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(random.PRNGKey(rng_seed), angles=x_jax)

    samples = mcmc.get_samples()

    kappa_samples = np.asarray(samples["kappa"])
    mu_samples = np.asarray(samples["mu"])

    log_lik_samples = np.array([
        dist.VonMises(loc=mu_sample, concentration=kappa_sample).log_prob(x_jax)
        for kappa_sample, mu_sample in zip(kappa_samples, mu_samples)
    ])
    
    waic = _compute_waic(log_lik_samples)

    return {
        "n": len(x),
        "distribution": "vonmises_hmc",
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
        "kappa_samples": kappa_samples,
        "mu_samples": mu_samples,
        "samples": samples,
        "mcmc": mcmc,
        "log_lik_samples": log_lik_samples,
        "waic": waic
    }

def _vm_uniform_angle_model(angles):
    kappa = numpyro.sample("kappa", dist.HalfNormal(10.0))
    w = numpyro.sample("w", dist.Beta(1.0, 1.0))

    vm = dist.VonMises(loc=0.0, concentration=kappa)
    vm_logprob = vm.log_prob(angles)

    mixture_logprob = jnp.logaddexp(
        jnp.log(w) + vm_logprob,
        jnp.log1p(-w) - jnp.log(2.0 * jnp.pi),
    )

    numpyro.factor("obs_loglik", jnp.sum(mixture_logprob))

def fit_vm_uniform_distribution_hmc(
    angles,
    *,
    num_warmup=1000,
    num_samples=2000,
    num_chains=1,
    rng_seed=0,
):
    x = pd.Series(angles).dropna().to_numpy()

    if len(x) == 0:
        raise ValueError("No turning angles available after filtering.")

    x_jax = jnp.asarray(x)

    kernel = NUTS(_vm_uniform_angle_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(random.PRNGKey(rng_seed), angles=x_jax)

    samples = mcmc.get_samples()

    kappa_samples = np.asarray(samples["kappa"])
    w_samples = np.asarray(samples["w"])

    log_lik_samples = np.array([
        np.logaddexp(
            np.log(w_sample) + np.asarray(dist.VonMises(loc=0.0, concentration=kappa_sample).log_prob(x_jax)),
            np.log1p(-w_sample) - np.log(2.0 * np.pi),
        )
        for kappa_sample, w_sample in zip(kappa_samples, w_samples)
    ])
    
    waic = _compute_waic(log_lik_samples)

    return {
        "n": len(x),
        "distribution": "vm_uniform_hmc",
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
        "kappa_samples": kappa_samples,
        "w_samples": w_samples,
        "samples": samples,
        "mcmc": mcmc,
        "log_lik_samples": log_lik_samples,
        "waic": waic
        
    }

