import numpy as np
import pandas as pd
import scipy.stats as stats

from scipy.optimize import minimize
from scipy.stats import vonmises

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