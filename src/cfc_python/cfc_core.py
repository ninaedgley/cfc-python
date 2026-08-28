# cfc_core.py
# Responsible for producing confidence forced-choice predictions through the core deterministic and probabilistic decision variables for a given trial of a confidence forced-choice experiment.
# P(C=1 | s1, s2, D1, D2)
# Equations 24, 25, 26, 27, 28 in Mamassian & de Gardelle (2022)

# Input : `grouped_data` [contains stimulus intensities and perceptual decisions, grouped across blocks in the dataset], `model_params_vals`
# Output: `conf_choice_prob` [type 2 choice probability to select interval 1], `conf_choice_freq` [nb of confidence choices for intervals1 and 2], `choose1_numerator_resp` [P(C=1 | D1, D2)]

# `grouped_data` column index:
    # 0 = stimulus intensity for interval 1 (s1)
    # 1 = stimulus intensity for interval 2 (s2)
    # 2 = perceptual decision for interval 1 (r1)
    # 3 = perceptual decision for interval 2 (r2)
    # 4 = number of confidence choices for interval 1
    # 5 = number of confidence choices for interval 2
    # 6 = stimulus task for interval 1 (optional)
    # 7 = stimulus task for interval 2 (optional) 

# `model_params_vals` index:
    # 'tasks_list' : vector of tasks ('1', or '[1,2]')
    # 'sens_noise' : type 1 sensory stddev of noise (0 = perfectly sensitive)
    # 'sens_crit' : type 1 sensory criterion
    # 'conf_noise' : type 2 confidence stddev of noise (0 = ideal observer)
    # 'conf_boost' : fraction of super-ideal confidence evidence access (0 = ideal observer, 1 = super-ideal observer)
    # 'conf_crit' : type 2 confidence criterion
    # 'intrvl_bias' bias in favour of interval 1 > interval 2
    # 'conf_bias' : overconfidence relative to one of the tasks

# 2 changes to original MATLAB code : a closed-form solution for the confidence-choice probability, and Gauss-Legendre quadrature to replace the previous `dblquad` adaptive nested integration in python.
# The latter was implemented to reduce runtime and avoid numerical instability. See Methods & Materials for details on the closed-form derivation.

# The two are used in different regimes of conf_noise. GL quadrature integrates over the raw sensory evidence, and its integrand tends to a step function as conf_noise -> 0, which polynomial quadrature cannot represent (24 nodes are accurate to ~1e-5 at conf_noise = 0.1, but wrong in the 3rd decimal by conf_noise = 0.01, and more nodes have a very marginal effect). 
# The closed form is exact at every conf_noise, so it takes over below `closed_form_threshold`, where the two agree to ~3e-6.


import numpy as np
import scipy.stats as stats
from numpy.polynomial.legendre import leggauss
from .closed_form import closed_form_choice_prob

closed_form_threshold = 0.1  # below this conf_noise, uses the closed form instead of GL quadrature. The switch introduces no discontinuity, as the two agree to ~3e-6

def cfc_core(grouped_data, model_params_vals, n_nodes=24):

    nb_kinds = grouped_data.shape[0] # number of unique s1 x s2 pairs, expressed as number of rows
    conf_choice_prob = np.full(nb_kinds, np.nan)
    choose1_numerator_resp = np.full(nb_kinds, np.nan) 

    if(grouped_data.shape[1] < 7):
        tasks_nn = np.tile([1, 1], (nb_kinds, 1)) # if there is only 1 task / interval, fills in tiles of 1s in the remaining column
        grouped_data = np.concatenate((grouped_data, tasks_nn), axis=1)
        
    verbose = 0 # set to 1 to display intermediate outputs

    # Type 1 Parameters
    tasks_list = np.atleast_1d(model_params_vals['tasks_list'])
    sens_noise_task = np.atleast_1d(model_params_vals['sens_noise'])
    sens_crit_task = np.atleast_1d(model_params_vals['sens_crit'])
    nb_tasks = len(tasks_list)

    # Type 2 Parameters -> obligatory
    conf_noise_task = np.atleast_1d(model_params_vals['conf_noise']).copy()
    conf_boost_task = np.atleast_1d(model_params_vals['conf_boost'])

    # Type 2 Parameters
    if 'conf_crit' in model_params_vals:
        conf_crit_task = np.atleast_1d(model_params_vals['conf_crit']).astype(float)
    else:
        conf_crit_task = np.zeros(nb_tasks) # default to 0 if not provided
    if 'intrvl_bias' in model_params_vals: # fixed as favouring interval 1 : make sure that this aligns with experimental design. If the bias is modal, not fixed by interval, estimates will be biased
        intrvl_bias = model_params_vals['intrvl_bias']
    else:
        intrvl_bias = 0 # default to 0 if not provided
    if 'conf_bias' in model_params_vals: # ratio, only measured relative to one of the tasks. >1 overconfident, =1 baseline, <1 underconfident
        conf_bias_task = np.atleast_1d(model_params_vals['conf_bias']).astype(float)
    else:
        conf_bias_task = np.ones(nb_tasks) # default to 1 if not provided
    
    noise2_inds = np.isnan(conf_noise_task) # tasks with no type 2 noise (ideal observer)
    conf_noise_task[noise2_inds] = conf_noise_task[0]
    below_floor = conf_noise_task < closed_form_threshold # defined at top of script. Avoids numerical issues in integration
    conf_noise_gl = np.maximum(conf_noise_task, closed_form_threshold) # no operation given branch below, kept as guard against a delta-function integrand
    conf_scale_task = conf_bias_task / sens_noise_task

    zLeg, wLeg = leggauss(n_nodes) # GL replaces integral with a weighted sum of function evaluations at quadrature points on [-1, 1]. n_nodes = 24, set in function definition. z_i = quadrature nodes, and w_i = corresponding weights. Sum(wLeg) == 2, as dictated by the interval

    # P(C=1 | s1, s2, D1, D2) - equations 25 / 26 / 27
    for row_idx in range(nb_kinds):

        if verbose:
            if(row_idx + 1) % 50 == 0:
                print('.\n')
            elif(row_idx + 1) % 10 == 0:
                print('.')
        
        row = grouped_data[row_idx]
        tsk1 = row[6]
        tsk2 = row[7]

        tsk1_ind = np.where(tasks_list == tsk1)[0][0]
        tsk2_ind = np.where(tasks_list == tsk2)[0][0]

        mu1 = row[0]
        mu2 = row[1]
        resp1 = row[2] # Type 1 responses for intervals 1, 2
        resp2 = row[3]

        sn1, sc1 = sens_noise_task[tsk1_ind], sens_crit_task[tsk1_ind]
        sn2, sc2 = sens_noise_task[tsk2_ind], sens_crit_task[tsk2_ind]

       # Closed form rewrite : exact at every conf_noise, the only accurate option as conf_noise -> 0. See ideal_closed_form.py for details and function definition
        if below_floor[tsk1_ind] or below_floor[tsk2_ind]:
           choose1_num, joint_prob = closed_form_choice_prob(
               row,
               sn1, sc1, conf_noise_task[tsk1_ind], conf_boost_task[tsk1_ind], conf_crit_task[tsk1_ind], conf_bias_task[tsk1_ind],
               sn2, sc2, conf_noise_task[tsk2_ind], conf_boost_task[tsk2_ind], conf_crit_task[tsk2_ind], conf_bias_task[tsk2_ind],
               intrvl_bias,
           )
           conf_choice_prob[row_idx] = choose1_num / joint_prob if joint_prob > 0 else 0.5
           choose1_numerator_resp[row_idx] = choose1_num
           continue
       
        else:
           cn1, cn2 = conf_noise_gl[tsk1_ind], conf_noise_gl[tsk2_ind]
           cb1, cb2 = conf_boost_task[tsk1_ind], conf_boost_task[tsk2_ind]
           cc1, cc2 = conf_crit_task[tsk1_ind], conf_crit_task[tsk2_ind]
           cs1, cs2 = conf_scale_task[tsk1_ind], conf_scale_task[tsk2_ind]
        
       # Defines the relevant perceptual quadrants in probability space
        e1 = (sc1 - mu1) / sn1 # normalised evidence across intervals, converting  into z-scores
        e2 = (sc2 - mu2) / sn2
        lo1, hi1 = (stats.norm.cdf(e1), 1.0) if resp1 == 1 else (0.0, stats.norm.cdf(e1)) # maps the evidence space into probability space, defining the low / high boundaries according to the perceptual stimuli and responses
        lo2, hi2 = (stats.norm.cdf(e2), 1.0) if resp2 == 1 else (0.0, stats.norm.cdf(e2))
        joint_prob = (hi1 - lo1) * (hi2 - lo2) # assumes independence of evidence across both intervals

       # Gauss-Legendre nodes are mapped into lo, hi on the probability scale, then back to the evidence space using an inverse-normal transformation (ppf). 
        p1 = lo1 + (hi1 - lo1) / 2.0 * (zLeg + 1.0) # takes the GL nodes, maps them into the probability space defined above, which gives a percentile of the evidence distribution
        p2 = lo2 + (hi2 - lo2) / 2.0 * (zLeg + 1.0)
        x1 = mu1 + sn1 * stats.norm.ppf(p1) # maps the percentile of the evidence distribution back into the evidence space, giving the actual evidence (x1, x2) values for each interval. Done for all GL nodes, giving a 24x24 grid across both intervals
        x2 = mu2 + sn2 * stats.norm.ppf(p2)

        # E[w | s] 
        c_Ox = (x1 + (mu1 - x1) * cb1 - sc1 - cc1) * cs1
        c_Oy = (x2 + (mu2 - x2) * cb2 - sc2 - cc2) * cs2
        C_OX, C_OY = np.meshgrid(c_Ox, c_Oy, indexing='ij')
        c_Oxr = C_OX / cn1 # confidence evidence normalised by conf_noise, giving a standard normal distribution of confidence evidence across the GL nodes
        c_Oyr = C_OY / cn2

        bias_threshold_z = intrvl_bias / np.sqrt(cn1**2 + cn2**2) # interval bias, standardised by the SD of confidence noise (shifts confidence-choice boundary)
        slope1 = -cn1 / cn2 # orientation of confidence-choice boundaries before the rotation, determined by relative confidence noise across intervals
        slope2 = cn1 / cn2
        proj_sum = (-slope1 * c_Oxr + c_Oyr) / np.sqrt(1 + slope1**2) # projection of confidence-evidence points onto perpendcular axes (coordinate transformation <> rotation of confidence evidence plane, equation 26)
        proj_diff = (-slope2 * c_Oxr + c_Oyr) / np.sqrt(1 + slope2**2)

        # P(C=1 | s1, s2, D1, D2), taking interval biases into account and the relevant perceptual quadrant - equation 27
        if resp1 == 1 and resp2 == 1:
            choose1_kernel = stats.norm.cdf(bias_threshold_z, loc=proj_diff, scale=1)
        elif resp1 == 1 and resp2 == 0:
            choose1_kernel = 1.0 - stats.norm.cdf(-bias_threshold_z, loc=proj_sum, scale=1)
        elif resp1 == 0 and resp2 == 1:
             choose1_kernel = stats.norm.cdf(bias_threshold_z, loc=proj_sum, scale=1)
        elif resp1 == 0 and resp2 == 0:
             choose1_kernel = 1.0 - stats.norm.cdf(-bias_threshold_z, loc=proj_diff, scale=1)
        else:
             raise ValueError(f"row {row_idx}: responses must be 0/1, got ({resp1}, {resp2})")
 
        W1, W2 = np.meshgrid(wLeg * (hi1 - lo1) / 2.0, wLeg * (hi2 - lo2) / 2.0, indexing='ij')
        choose1_numerator = np.sum(W1 * W2 * choose1_kernel) # numerator for equation 28
 
        conf_choice_prob[row_idx] = choose1_numerator / joint_prob if joint_prob > 0 else 0.5 # P(C=1 | D1, D2) - full equation 28
        choose1_numerator_resp[row_idx] = choose1_numerator
 
    conf_choice_prob[np.isnan(conf_choice_prob)] = 0.5
    conf_choice_prob[conf_choice_prob < 0.0] = 0.0
    conf_choice_prob[conf_choice_prob > 1.0] = 1.0
 
    if verbose:
        print(f'\n')
 
    obs_conf_choice_counts = grouped_data[:, 4:6]
    total_conf_choice_counts = obs_conf_choice_counts.sum(axis=1)
    predicted_int1_choices = np.round(total_conf_choice_counts * conf_choice_prob)
    conf_choice_freq = np.column_stack([predicted_int1_choices, total_conf_choice_counts - predicted_int1_choices])
 
    return conf_choice_prob, conf_choice_freq, choose1_numerator_resp