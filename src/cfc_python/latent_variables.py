# cfc_latent_confidence.py
# Trial-level latent-variable extraction from a fitted CFC model. Produces the model-implied internal quantities, suitable as regressors for every raw trial.

# Latent variables per trial :
    # z1, z2 - signed criterion-referenced sensory evidence, ((mu - sens_crit)/sens_noise), deterministic from stimulus values and Type-1 fit
    # p_r1, p_r2 - P(observed T1 response | stimulus), the model's probability of the participant's actual response.
    # conf_ev1/2 - E[w | stim, response]. Mean of the confidence evidence conditional on the relevant response quadrant. Units = same normalized conf-evidence scale cfc_core uses internally.
    # conf_mag1/2 - E[|w + conf_noise|]. Expected confidence magnitude per interval. Approximation : the truncated-normal conditional distribution of evidence is treated as normal with matched mean/var before adding conf_noise and taking the folded-normal mean. Both mean and var are exact, the shape is approximated.
    # delta_conf_mag - (conf_mag1 - conf_mag2). Signed confidence differential, the model's proxy for the interval-comparison decision variable.
    # p_choose1 - P(C=1 | s1, s2, r1, r2). The model's exact confidence-choice probability, computed by cfc_core
    # conf_chosen - P assigned by the model to the interval the participant chose. Mean > 0.5 IFF the model predicts the realised confidence choices above chance


import argparse
import json

import numpy as np
import scipy.stats as stats

from .cfc_core import cfc_core

sqrt_2_over_pi = np.sqrt(2.0 / np.pi)

def truncnorm_moments(mu, sn, sc, resp): # Mean/variance of x ~ N(mu, sn) conditional on the response quadrant

    a = (sc - mu) / sn
    pdf_a = stats.norm.pdf(a)
    cdf_a = stats.norm.cdf(a)
    upper = resp == 1

    # resp=1 keeps x > sc, resp=0 keeps x < sc
    lam = np.where(upper,
                   pdf_a / np.clip(1.0 - cdf_a, 1e-12, None),
                   -pdf_a / np.clip(cdf_a, 1e-12, None))
    mean = mu + sn * lam
    # standardized truncated variance: 1 + a*lam - lam^2 (holds for both tails with signed lam)
    var = sn ** 2 * np.clip(1.0 + a * lam - lam ** 2, 1e-12, None)

    return mean, var


def folded_normal_mean(m, s): # E[|X|] for X ~ N(m, s), exact closed-form

    s = np.clip(s, 1e-12, None)

    return s * sqrt_2_over_pi * np.exp(-m ** 2 / (2 * s ** 2)) + m * (2 * stats.norm.cdf(m / s) - 1)

# raw (N, 8) per-trial array with columns : s1, s2, r1, r2, chose1, chose2, task1, task2
# params is a dict with fitted model's parameters : sens_noise, sens_crit, conf_noise, conf_boost, conf_bias. Each length-2 array/list indexed by task (1=index 0, 2=index 1). conf_crit and interval_bias default to 0 if omitted
def latent_variables(raw, params):

    full_params = {
        'tasks_list': np.array([1, 2]),
        'sens_noise': np.asarray(params['sens_noise'], dtype=float),
        'sens_crit': np.asarray(params['sens_crit'], dtype=float),
        'conf_noise': np.asarray(params['conf_noise'], dtype=float),
        'conf_boost': np.asarray(params['conf_boost'], dtype=float),
        'conf_bias': np.asarray(params.get('conf_bias', [1.0, 1.0]), dtype=float),
        'conf_crit': np.asarray(params.get('conf_crit', [0.0, 0.0]), dtype=float),
        'intrvl_bias': params.get('intrvl_bias', 0.0),
    }

    s = raw[:, 0:2]
    r = raw[:, 2:4]
    tasks = raw[:, 6:8].astype(int)
    ti = tasks - 1  # param index per interval (0=V, 1=T)
 
    sn = full_params['sens_noise'][ti]
    sc = full_params['sens_crit'][ti]
    cn = full_params['conf_noise'][ti]
    cb = full_params['conf_boost'][ti]
    cbias = full_params['conf_bias'][ti]
 
    # deterministic evidence transform + exact Type-1 response probability
    z = (s - sc) / sn
    cdf = stats.norm.cdf((sc - s) / sn)
    p_r = np.where(r == 1, 1.0 - cdf, cdf)
 
    # conditional confidence-evidence moments (exact truncated-normal closed forms)
    ev_mean = np.empty_like(s)
    ev_var = np.empty_like(s)

    for col in (0, 1):
        m, v = truncnorm_moments(s[:, col], sn[:, col], sc[:, col], r[:, col])
        ev_mean[:, col] = m
        ev_var[:, col] = v
 
    # confidence evidence c = ((1 - conf_boost)x + b*mu - sens_crit) * bias / sens_noise)
    # conf_crit = 0, conditional mean/variance follow by linearity
    scale = cbias / sn
    c_mean = ((1.0 - cb) * ev_mean + cb * s - sc) * scale
    c_sd_total = np.sqrt(((1.0 - cb) * scale) ** 2 * ev_var + cn ** 2)
    conf_mag = folded_normal_mean(c_mean, c_sd_total)  # normal-shape approximation
 
    # exact model confidence-choice probability, via cfc_core on unique (s, r, task) combos
    key = np.column_stack([s, r, tasks])
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    grouped = np.column_stack([
        uniq[:, 0], uniq[:, 1], uniq[:, 2], uniq[:, 3],
        np.ones(len(uniq)), np.zeros(len(uniq)),
        uniq[:, 4], uniq[:, 5],
    ]).astype(float)
    
    prob_u, _, _ = cfc_core(grouped, full_params)
    p_choose1 = prob_u[inv]
 
    chose1 = raw[:, 4]
    conf_chosen = np.where(chose1 == 1, p_choose1, 1.0 - p_choose1)
 
    return {
        'z1': z[:, 0], 'z2': z[:, 1],
        'p_r1': p_r[:, 0], 'p_r2': p_r[:, 1],
        'conf_ev1': c_mean[:, 0], 'conf_ev2': c_mean[:, 1],
        'conf_mag1': conf_mag[:, 0], 'conf_mag2': conf_mag[:, 1],
        'delta_conf_mag': conf_mag[:, 0] - conf_mag[:, 1],
        'p_choose1': p_choose1,
        'conf_chosen': conf_chosen,
    }