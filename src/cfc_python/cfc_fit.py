# CFC Fit 
# Fits the generative model - takes `grouped_data` as input, outputs a `cfc_struct` with estimated parameter values.
# Requires 4 arguments:
#   `model_parameters`: structure defining which parameters are treated as free under the specific experimental application of the model, 
#   `model_fixed_values`: applies the same principle for fixed parameters,
#   `boost_init_list`: True / False, determines the starting boost values for the Type 2 fit
#   `skip_efficiency`: True / False, determines whether the metacognitive efficiency computations (steps 3 and 4) are skipped
#   `verbose`: 0 = silent, 1 = fit summaries, 2 = per-step optimiser detail + timing

# There are 3 optional implementation choices - boost_init, skip_efficiency, and verbose. Each subcomponent of the structures (dicts, here) should be expressed as a vector is task # > 1.
# cfc_fit requires several separate functions, defined at the top of the script here.

## cfc_type1 - Type 1 choice probabilities per task (Gaussian CDF)
## cfc_core_wrap - Wrapper around the core function that allows for flexibly adding parameters
## pack_params_in_struct - writes a flat free-parameter vector into a full parameter struct
## extract_params_from_struct - transforms a parameter struct (here, dict) into a vector of free parameters for optimisation
## loglikefcn - binomial negative summed log-likelihood (for minimisation)
## fitnll - single-start bounded Nelder-Mead MLE
## fit_nll_multistart - fitnll from several scaled starts, keeps the rest


import numpy as np
import scipy.stats as stats
import time as time
from scipy.optimize import minimize
import copy as copy
from .cfc_core import cfc_core


# Type 1 choice probabilities / task. sens_noise and sens_crit are taken from the free-param vector where params_set[field][task] > 0, or from fixed_vals where it is 0
def cfc_type1(xx_vals, variable_params, params_set, fixed_vals):

# xx_vals : (n_stim, 2) array where col 0 = task, col 1 = stim
# variable_params : current free-parameter vector
# params_set : free param map (non-0 = 1-based index, 0 = fixed)
    
    noise_set_list = params_set['sens_noise']
    crit_set_list = params_set['sens_crit']
    tsk_list = fixed_vals['tasks_list']
    noise_fix_list = fixed_vals['sens_noise']
    crit_fix_list = fixed_vals['sens_crit']
    nb_tasks = len(noise_set_list)
    noise_val_list = np.full(nb_tasks, np.nan)
    crit_val_list = np.full(nb_tasks, np.nan)

    for t1_tt in range(nb_tasks):
        if noise_set_list[t1_tt]:
            noise_val_list[t1_tt] = variable_params[noise_set_list[t1_tt] - 1]
        else:
            noise_val_list[t1_tt] = noise_fix_list[t1_tt]
        
        if crit_set_list[t1_tt]:
            crit_val_list[t1_tt] = variable_params[crit_set_list[t1_tt] - 1]
        else:
            crit_val_list[t1_tt] = crit_fix_list[t1_tt]

    nb_stims = xx_vals.shape[0]
    t1_tsk_vals, tsk_ic = np.unique(xx_vals[:,0], return_inverse=True)
    nb_tasks = len(t1_tsk_vals)

    yy_vals = np.full(nb_stims, np.nan)

    for t1_tt in range(nb_tasks):
        t1_inds = (tsk_ic==t1_tt)
        xx = xx_vals[t1_inds, 1]
        my_tsk = t1_tsk_vals[t1_tt]
        tsk_ind = np.where(tsk_list==my_tsk)[0][0]
        noise = noise_val_list[tsk_ind]
        crit = crit_val_list[tsk_ind]

        yy_vals[t1_inds] = stats.norm.cdf(xx, crit, noise) # P(C = 1 | stimulus) = phi((stim - crit)/noise) for every row of xx_vals
    
    return yy_vals # (n_stim,) array of T1 choice probabilities


# Adapter - packs a free-parameter vector into a full struct, calls cfc_core, then returns resulting predictions (pred_resp, pred_nn1)
def cfc_core_wrap(wrap_data, variable_params, params_set, fixed_vals):

    params_vals = pack_params_in_struct(variable_params, params_set, fixed_vals)
    pred_resp, pred_nn1,_ = cfc_core(wrap_data, params_vals)

    return pred_resp, pred_nn1


# Inverse of extract_params_from_struct - writes a flat free-parameter vector into a full parameter struct, starting from deep copy of fixed_vals --> overwrites positions flagged in params_set
def pack_params_in_struct(variable_params, params_set, fixed_vals):
    
    params_struct = copy.deepcopy(fixed_vals)
    variable_nb = len(variable_params)
    
    for kk in range(1, variable_nb + 1):
        for field_name, field_value in params_set.items():
            field_value = np.atleast_1d(field_value)
            positions = np.where(field_value == kk)[0]
            for pos in positions:
                params_struct[field_name][pos] = variable_params[kk - 1]

    return params_struct


# Inverse of pack_params_in_struct - reads a full parameter struct into a flat free-parameter vector, ordered by indices in params_set
def extract_params_from_struct(params_set, params_struct):

    variable_nb = int( # max. index in params_set
        max(np.max(np.atleast_1d(v))
            for v in params_set.values())
    )

    params_vals = np.empty(variable_nb)

    for kk in range(1, variable_nb + 1): # assumes indices contain no gaps - a missing index will leave a np.empty entry
        for field, mapping in params_set.items():
            mapping = np.atleast_1d(mapping)
            matches = np.where(mapping == kk)[0]
            if matches.size:
                params_vals[kk - 1] = np.atleast_1d(
                    params_struct[field]
                )[matches[0]]
                break

    return params_vals


# Binomial negative summed log-likelihood function
def loglikefcn(pp, ff, nn):
# pp : parameter vector passed to ff
# ff : callable returning per-row P(C=1)
# nn : (n_rows, 2) observed counts

    ypred = ff(pp)
    if isinstance(ypred, tuple):
        ypred = ypred[0]
    ypred = np.clip(ypred, 1e-6, 1 - 1e-6)

    ll1 = np.log(ypred)
    ll0 = np.log(1 - ypred)

    ll_vect = nn[:,0] * ll1 + nn[:,1] * ll0
    nglglk = -(np.sum(ll_vect))

    return nglglk # p clipped to [1e-6, 1-1e-6], positive value - fitnll negates it back to log-likelihood on return


# Type 1 fit (unimodal, with a single start max. likelihood fit via bounded Nelder-Mead sufficient)
# MATLAB code had separate fminsearchbnd to enforce bounds, which here are handled by scipy >= 1.7
def fitnll(fit_fcn, nn1_list, params_0, params_LB, params_UB, fit_options):
    fun = lambda xx: loglikefcn(xx, fit_fcn,nn1_list)
    result = minimize(fun, x0 = params_0, bounds = list(zip(params_LB, params_UB)), method="Nelder-Mead", options=fit_options)
    params_best = result.x
    nll_best = result.fun

    return params_best, -nll_best


# Type 2 multistart fit (mitigates local minima due to noise-boost ridge in the full model's likelihood surface)
def fitnll_multistart(fit_fcn, nn1_list, params_0, params_LB, params_UB, fit_options,
                       start_multipliers=(0.3, 3.0, 20.0)):
    
    if not start_multipliers:
        raise ValueError("fitnll_multistart: start_multipliers must be non-empty")

    best_params = None
    best_loglike = -np.inf

    for mult in start_multipliers:
        p0 = np.clip(params_0 * mult, params_LB, params_UB)
        params_best, loglike = fitnll(fit_fcn, nn1_list, p0, params_LB, params_UB, fit_options)

        if best_params is None or loglike > best_loglike:
            best_params, best_loglike = params_best, loglike

    assert best_params is not None  # guaranteed by non-empty check above

    return best_params, best_loglike


def cfc_fit(
    
    # Parameter definitions, default and optional values    
        grouped_data,
        model_parameters=None,
        model_fixed_values=None,
        boost_init_list=None,
        skip_efficiency=False,
        verbose=1,
):  
    # Defaults - dicts and list
    if model_parameters is None:
        model_parameters = {}
    
    if model_fixed_values is None:
        model_fixed_values = {}
    
    if boost_init_list is None:
        boost_init_list = []
    
    if not isinstance(grouped_data, np.ndarray):
        raise TypeError("grouped_data must be a NumPy array")
    
    if not isinstance(model_parameters, dict):
        raise TypeError("model_parameters must be a dict")

    if not isinstance(model_fixed_values, dict):
        raise TypeError("model_fixed_values must be a dict")

    if not isinstance(skip_efficiency, bool):
        raise TypeError("skip_efficiency must be bool")
    
    if not isinstance(verbose, (int, float)):
        raise TypeError("verbose must be numeric")
    
    options1 = {} 
    if verbose >= 2:
        options2 = {'xatol': 1e-3,'fatol': 1e-3, 'disp': True,}
        tstart = time.time()
    else:
        options2 = {'xatol': 1e-3, 'fatol': 1e-3}
        tstart = 0

    compute_efficiency = not skip_efficiency

    # Default parameter values
    default_sens_noise = 1.0
    default_sens_crit = 0.0
    default_conf_noise = 0.0
    default_conf_boost = 0.0
    default_conf_crit = 0.0
    default_intrvl_bias = 0.0
    default_conf_bias = 1.0

    # Initial parameter values 
    initial_sens_noise = 3.2 # raw stimulus values
    initial_sens_crit = 4.0 # raw stimulus values
    initial_conf_noise = 0.5 # standardised scale
    initial_conf_boost = 0.2
    initial_conf_crit = 0.4
    initial_intrvl_bias = 0.6
    initial_conf_bias = 1.5

    # Lower and upper bound parameter values
    lo_bnd_sens_noise = 0.0
    lo_bnd_sens_crit = float('-inf')
    lo_bnd_conf_noise = 0.0
    lo_bnd_conf_boost = 0.0
    lo_bnd_conf_crit = float('-inf')
    lo_bnd_intrvl_bias = float('-inf')
    lo_bnd_conf_bias = 0.0
    
    hi_bnd_sens_noise = float('inf')
    hi_bnd_sens_crit = float('inf')
    hi_bnd_conf_noise = float('inf')
    hi_bnd_conf_boost = 1.0
    hi_bnd_conf_crit = float('inf')
    hi_bnd_intrvl_bias = float('inf')
    hi_bnd_conf_bias = float('inf')


    ## STEP 0 - Extract Data + Initialise Internal Variables
    stim_pair = grouped_data[:,0:2]
    response_pair = grouped_data[:,2:4] # perceptual response
    choice_pair = grouped_data[:,4:6] # confidence choice
    task_pair = grouped_data[:,6:8] # task index

    # Total number of trials for stimulus 1, 2 and responses 1, 2
    total1 = np.sum(choice_pair, axis=1)

    stim_all = np.concatenate([stim_pair[:,0], stim_pair[:,1]])
    task_all = np.concatenate([task_pair[:,0], task_pair[:,1]])
    resp_all = np.concatenate([response_pair[:,0], response_pair[:,1]])
    tot_all = np.concatenate([total1, total1])

    stim_task_all = np.column_stack([stim_all, task_all])
    stim_values, stm_inv_index = np.unique(stim_task_all, axis=0, return_inverse=True) # stim_inv_index returns the unique (stim, task) combination each stim_task_all row belongs to
    nb_unique_stims2 = stim_values.shape[0]
    task_values = np.unique(task_pair)
    nb_unique_tasks = len(task_values)

    # Type 1 parameter objects
    sens_noise_task = np.full(nb_unique_tasks, np.nan)
    sens_crit_task = np.full(nb_unique_tasks, np.nan)

    # Type 2 parameter objects
    conf_noise_task = np.full(nb_unique_tasks, np.nan)
    conf_crit_task = np.full(nb_unique_tasks, np.nan)
    conf_boost_task = np.full(nb_unique_tasks, np.nan)
    conf2_bias = np.nan # assumes that confidence bias for interval 2 is defined relative to task 1 - if equal = 1, if task 2 underweighted rel. to task 1 = < 1, if overweighted rel. to task 1 = >1
    intrvl1_bias = np.nan # positional bias, fixed to interval 1 if activated
    chosen_full = np.nan
    nn1_full = np.nan
    boost_search_diagnostics = None

    # `params_set` defines the free parameters and their order. If a variable is not a free parameter in specific experimental paradigm, '0'
    n = nb_unique_tasks
    
    default_params_set = {
        'tasks_list': np.zeros(n),
        'sens_noise': np.zeros(n),
        'sens_crit': np.zeros(n),
        'conf_noise': np.zeros(n),
        'conf_boost': np.zeros(n),
        'conf_crit': np.zeros(n),
        'intrvl_bias': 0, # task indepenedent single scalar
        'conf_bias': np.zeros(n)
    }
    default_params_values = {
        'tasks_list': np.arange(1, n+1),
        'sens_noise': np.ones(n) * default_sens_noise,
        'sens_crit': np.ones(n) * default_sens_crit,
        'conf_noise': np.ones(n) * default_conf_noise,
        'conf_boost': np.ones(n) * default_conf_boost,
        'conf_crit': np.ones(n) * default_conf_crit,
        'intrvl_bias': default_intrvl_bias,
        'conf_bias': np.ones(n)
    }
    initial_params = {
        'tasks_list': np.arange(1, n + 1),
        'sens_noise': np.full(n, initial_sens_noise),
        'sens_crit': np.full(n, initial_sens_crit),
        'conf_noise': np.full(n, initial_conf_noise),
        'conf_boost': np.full(n, initial_conf_boost),
        'conf_crit': np.full(n, initial_conf_crit),
        'intrvl_bias': initial_intrvl_bias, # task independent single scalar
        'conf_bias': np.full(n, initial_conf_bias),
    }

    lo_bnd_params = {
        'tasks_list': np.arange(1, n + 1),
        'sens_noise': np.full(n, lo_bnd_sens_noise),
        'sens_crit': np.full(n, lo_bnd_sens_crit),
        'conf_noise': np.full(n, lo_bnd_conf_noise),
        'conf_boost': np.full(n, lo_bnd_conf_boost),
        'conf_crit': np.full(n, lo_bnd_conf_crit),
        'intrvl_bias': lo_bnd_intrvl_bias,
        'conf_bias': np.full(n, lo_bnd_conf_bias),
    }

    hi_bnd_params = {
        'tasks_list':   np.arange(1, n + 1),
        'sens_noise':   np.full(n, hi_bnd_sens_noise),
        'sens_crit':    np.full(n, hi_bnd_sens_crit),
        'conf_noise':   np.full(n, hi_bnd_conf_noise),
        'conf_boost':   np.full(n, hi_bnd_conf_boost),
        'conf_crit':    np.full(n, hi_bnd_conf_crit),
        'intrvl_bias':  hi_bnd_intrvl_bias,
        'conf_bias':    np.full(n, hi_bnd_conf_bias),
    }

    # Determines how many free parameters the model has
    if not model_parameters: 
        param_free_nb = 0
    else:
        all_param_indices = np.concatenate([np.atleast_1d(v) for v in model_parameters.values()])
        param_free_nb = int(np.max(all_param_indices))

    # Loops over specific cell values in params_set (init., 1, 2) to find which ones contain my_kk corresponding to a free parameter value. 
    # When it does, loops over the specific position in which this happens, and the field_name that corresponds to it (parameter name). This information is then stored below
    type1_names = {'sens_noise', 'sens_crit'} # defines T1 parameters, leaving all others classified as T2

    param_free_nb1 = 0 # counter initialisation
    param_free_nb2 = 0
    params_set = {k: (np.copy(v) if isinstance(v, np.ndarray) else v)
                  for k, v in default_params_set.items()}
    params_set1 = {k: (np.copy(v) if isinstance(v, np.ndarray) else v)
                   for k, v in default_params_set.items()}
    params_set2 = {k: (np.copy(v) if isinstance(v, np.ndarray) else v)
                   for k, v in default_params_set.items()}
    
    for my_kk in range(1, param_free_nb + 1):
        
        for field_name, field_value in model_parameters.items():
            field_value = np.atleast_1d(field_value)
            positions = np.where(field_value == my_kk)[0]

            for pos in positions: # counts T1 vs T2 free parameters 
                is_type1 = field_name in type1_names
                if is_type1:
                    param_free_nb1 += 1
                else:
                    param_free_nb2 += 1

                params_set[field_name][pos] = my_kk

                if is_type1:
                    params_set1[field_name][pos] = my_kk
                else:
                    params_set2[field_name][pos] = my_kk - param_free_nb1 # renumbers T2 parameters so they start at 1, independently of sequence in T1

    fixed_set = copy.deepcopy(default_params_set)
    fixed_values = copy.deepcopy(default_params_values)

    for field_name, field_value in model_fixed_values.items():
        field_value = np.atleast_1d(field_value)
        fixed_positions = np.flatnonzero(~np.isnan(field_value)) # finds all positions that aren't NaN

        for pos in fixed_positions:
            fixed_set[field_name][pos] = 1
            fixed_values[field_name][pos] = field_value[pos]

    n_free_conf_bias = int(np.count_nonzero(np.atleast_1d(params_set['conf_bias'])))
    assert n_free_conf_bias <= nb_unique_tasks -1, ("conf_bias must keep a fixed reference task to remain identifiable")

    ## STEP 1 - Fit Type 1 Performance
    nn1_list = np.full((nb_unique_stims2,2), np.nan) # one row for each stimuli-task combination
    for ww in range(nb_unique_stims2):
        inds = (stm_inv_index == ww)
        nn1_list[ww, 0] = np.sum(resp_all[inds]*tot_all[inds])
        nn1_list[ww, 1] = np.sum((1 - resp_all[inds])*tot_all[inds])
    
    params_set_type1 = copy.deepcopy(default_params_set) # parameter mapping index, used to store values rather than positions

    if 'sens_noise' in model_parameters:
        params_set_type1['sens_noise'] = model_parameters['sens_noise']
    else:
        params_set_type1['sens_noise'] = np.arange(1, nb_unique_tasks + 1)
    
    if 'sens_crit' in model_parameters:
        params_set_type1['sens_crit'] = model_parameters['sens_crit']
    else:
        params_set_type1['sens_crit'] = np.arange(nb_unique_tasks + 1, 2 * nb_unique_tasks + 1)
    
    params_set_cumul = copy.deepcopy(params_set_type1)
    params_set_cumul['sens_noise'] = 1
    params_set_cumul['sens_crit'] = 2

    # `extract_params_from_struct` is a nested function defined prior to cfc_fit
    params0_cumul = extract_params_from_struct(params_set_cumul, initial_params)
    params_LB_cumul = extract_params_from_struct(params_set_cumul, lo_bnd_params)
    params_UB_cumul = extract_params_from_struct(params_set_cumul, hi_bnd_params)

    for tt in range(nb_unique_tasks):
        tsk = task_values[tt]
        inds = np.where(stim_values[:,1] == tsk)[0]
        stm_tsk_vals = stim_values[inds,0]
        nn1_tsk_list = nn1_list[inds, :]
        
        # pp[1] and pp[0] are integrated in sensory space - stimulus values for task 1 and task 2. P(C=1 | stimulus)
        my_fun_0 = lambda pp: stats.norm.cdf(
            stm_tsk_vals,
            loc=pp[1], # sensory crit
            scale=pp[0] # sensory noise
            )

        # `fitnll` is a nested function defined prior to cfc_fit
        param_type1, loglike = fitnll(my_fun_0, nn1_tsk_list, params0_cumul, params_LB_cumul, params_UB_cumul, options1)
        sens_noise_task[tt] = param_type1[0]
        sens_crit_task[tt] = param_type1[1]

        if verbose >= 2:
            print(f"Unsorted trials for task {int(tsk)}: (crit1, noise1) = {sens_crit_task[tt]:7.3f},{sens_noise_task[tt]:7.3f}")
            print(f"Log-likelihood = {loglike:7.3f}\n")
        
    model_params_type1 = copy.deepcopy(default_params_values) # new full parameter dictionary
    model_params_type1['tasks_list'] = task_values # takes newly fit Type 1 parameter values
    model_params_type1['sens_noise'] = sens_noise_task
    model_params_type1['sens_crit'] = sens_crit_task
    
    params0_0 = extract_params_from_struct(params_set_type1, model_params_type1)
    params_LB_0 = extract_params_from_struct(params_set_type1, lo_bnd_params)
    params_UB_0 = extract_params_from_struct(params_set_type1, hi_bnd_params)

    xx_vals = np.full([nb_unique_stims2, 2], np.nan)
    xx_ind = 0
    nn1_tsk_list = np.full([nb_unique_stims2, 2], np.nan)

    for tt in range(nb_unique_tasks): # reconstructs task x stim array, combining both tasks
        
        tsk = task_values[tt]
        inds = np.where(stim_values[:,1] == tsk)[0]
        stm_tsk_vals = stim_values[inds, 0]

        xx_ind_new = xx_ind + len(stm_tsk_vals)
        xx_vals[(xx_ind):xx_ind_new, 0] = tsk
        xx_vals[(xx_ind):xx_ind_new, 1] = stm_tsk_vals

        nn1_tsk_list[xx_ind:xx_ind_new, :] = nn1_list[inds, :]

        xx_ind = xx_ind_new
    
    my_fun_0 = lambda pp: cfc_type1(xx_vals, pp, params_set_type1, fixed_values) # fits the parameter vector [noise_task1, noise_task2, criterion_task1, criterion_task2] and separates values again by task, before running the CDF for all observations
    param_type1, loglike = fitnll(my_fun_0, nn1_tsk_list, params0_0, params_LB_0, params_UB_0, options1) # joint fit of all 4 parameters, which accomodates shared parameters if they arise

    model_params_type_2 = pack_params_in_struct(param_type1, params_set_type1, fixed_values)
    sens_noise_task = model_params_type_2['sens_noise']
    sens_crit_task = model_params_type_2['sens_crit']


    ## STEP 2 - Simulate performance for ideal observer (noise2 = 0, boost2 = 0)

    model_params_ideal = copy.deepcopy(model_params_type_2) # contains fitted T1 parameters, while T2 (confidence) parameters remain at their fixed/default values
    chosen_ideal, nn1_ideal, _ = cfc_core(grouped_data, model_params_ideal) # generate confidence choices predicted by ideal observer, given fitted T1 params values

    model_params_super_ideal = copy.deepcopy(model_params_type_2) # repeats process with super-ideal observer (conf_boost = 1)
    model_params_super_ideal['conf_boost'] = np.ones(nb_unique_tasks)
    chosen_super_ideal, nn1_super_ideal, _ = cfc_core(grouped_data, model_params_super_ideal)


    ## STEP 3 - Get the equivalent confidence noise for this ideal performance
    ## Efficiency is defined relative to a super-ideal observer (boost = 1)
    if compute_efficiency:
        
        wrap_ideal = copy.deepcopy(grouped_data)
        wrap_ideal[:,4:6] = nn1_ideal
        params_set_ideal = copy.deepcopy(default_params_set)
        fixed_values_ideal = copy.deepcopy(model_params_type_2)

        if param_free_nb2 > 0:
            params_set_ideal['conf_noise'] = params_set2['conf_noise']
        else:
            params_set_ideal['conf_noise'] = np.arange(1, nb_unique_tasks + 1)
        
        # Fix confidence boost to 1
        fixed_values_ideal['conf_boost'] = np.ones(nb_unique_tasks)

        params_0_ideal = extract_params_from_struct(params_set_ideal, initial_params)
        params_LB_ideal = extract_params_from_struct(params_set_ideal, lo_bnd_params)
        params_UB_ideal = extract_params_from_struct(params_set_ideal, hi_bnd_params)

        my_fun_ideal = lambda pp: cfc_core_wrap(wrap_ideal, pp, params_set_ideal, fixed_values_ideal)
        noise2_ideal, loglike = fitnll_multistart(my_fun_ideal, nn1_ideal, params_0_ideal, params_LB_ideal, params_UB_ideal, options2)

        if verbose >= 1:
            for tt in range (len(noise2_ideal)):
                tsk = task_values[tt]
                print(f"Ideal for task {tsk}: Equivalent confidence noise = {noise2_ideal[tt]:7.3f}")
            print(f"Log-likelihood = {loglike:7.3f}")
        

        ## STEP 4 - Efficiency Computation
        my_fun_eff = lambda pp: cfc_core_wrap(grouped_data, pp, params_set_ideal, fixed_values_ideal) # equivalent confidence noise for actual data, still assuming boost = 1
        noise2_data, loglike = fitnll_multistart(my_fun_eff, choice_pair, params_0_ideal, params_LB_ideal, params_UB_ideal, options2)
        efficiency = (noise2_ideal / noise2_data)**2 

        if verbose >= 1:
            for tt in range(len(noise2_data)):
                tsk = task_values[tt]
                print(f"Data for task {tsk}")
                print(f"Equivalent confidence noise = {noise2_data[tt]:7.3f}")

            print(f"Log-likelihood = {loglike:7.3f}")

            for tt in range(len(efficiency)):
                tsk = task_values[tt]
                print(f"Confidence efficiency for task {tsk}: {efficiency[tt]:7.3f}")
            print(f'\n')
        
        chosen_eff, nn1_eff = my_fun_eff(noise2_data)

    else:
        noise2_ideal = np.nan
        noise2_data = np.nan
        efficiency = np.nan
        chosen_eff = np.nan
        nn1_eff = np.nan
    

    ## STEP 5 - Full Model Fit
    my_fun_full = lambda pp: cfc_core_wrap(grouped_data, pp, params_set2, fixed_values)

    if param_free_nb2 > 0:

        fixed_values['sens_noise'] = model_params_type_2['sens_noise']
        fixed_values['sens_crit'] = model_params_type_2['sens_crit']

        # Fit below uses multiple starting point : the Type 2 likelihood surface can contain local optima + poorly identified noise-boost combinations
        if len(boost_init_list) == 0:
            boost_init_list = [0.0, 0.2, 0.5, 0.8, 1.0]
        
        boost2_init_nb = len(boost_init_list)

        # one row per starting points, one column per free T2 parameter
        paramBest_mat = np.full([boost2_init_nb,param_free_nb2], np.nan)
        loglike_list = np.full([boost2_init_nb, 1], np.nan)    

        for bb in range (boost2_init_nb): # multiple starting values fit : a scalar start propagates to both tasks, while a vector start allows for asymmetric starting points

            boost2_init_val = boost_init_list[bb] 
            params_0_bb = copy.deepcopy(initial_params) # starts with global / initial default values, replaces them with fits below.

            if isinstance(boost2_init_val, dict):

                for fld in ('conf_noise', 'conf_boost', 'conf_bias'):
                    if fld in boost2_init_val:
                        arr = np.atleast_1d(np.asarray(boost2_init_val[fld], dtype=float))

                        if arr.size == 1: # scalar applied to every task
                            arr = np.full(nb_unique_tasks, arr.item())
                        elif arr.size != nb_unique_tasks: # otherwise, one value per task required
                            raise ValueError(
                                f"boost_init_list[{bb}]['{fld}'] has {arr.size} values, but "
                                f"there are {nb_unique_tasks} tasks."
                            )
                        params_0_bb[fld] = arr
                boost2_init_desc = str({k: list(np.atleast_1d(v)) for k, v in boost2_init_val.items()})

            else:
                boost2_init_arr = np.atleast_1d(np.asarray(boost2_init_val, dtype=float))

                if boost2_init_arr.size == 1:
                    boost2_init_arr = np.full(nb_unique_tasks, boost2_init_arr.item())
                elif boost2_init_arr.size != nb_unique_tasks:
                    raise ValueError(
                        f"boost_init_list[{bb}] has {boost2_init_arr.size} values but there are "
                        f"{nb_unique_tasks} tasks - give either one scalar (broadcast to all tasks) "
                        f"or exactly {nb_unique_tasks} values (one per task)."
                    )
                params_0_bb['conf_boost'] = boost2_init_arr
                boost2_init_desc = "(" + ", ".join(f"{v:.3f}" for v in boost2_init_arr) + ")"

            # Converts full parameter dictionaries into optimisation vector
            params_0_full = extract_params_from_struct(params_set2, params_0_bb)
            params_LB_full = extract_params_from_struct(params_set2, lo_bnd_params)
            params_UB_full = extract_params_from_struct(params_set2, hi_bnd_params)

            # T2 parameter optimisation, stored final values
            paramBest, loglike = fitnll(my_fun_full, choice_pair, params_0_full, params_LB_full, params_UB_full, options2)
            paramBest_mat[bb, :] = paramBest
            loglike_list[bb] = loglike

            if (boost2_init_nb > 1) and (verbose >= 1):
                print(f"For type2_init = {boost2_init_desc}, loglike = {loglike:7.3f}")
            
            # Starting point diagnostics
            if (boost2_init_nb > 1) and (verbose >= 1):
                paramBest_struct = pack_params_in_struct(paramBest, params_set2, fixed_values)
                conf_noise_vals = paramBest_struct['conf_noise']
                conf_boost_vals = paramBest_struct['conf_boost'] # reconstructs the parameter dict to display noise and boost task-by-task

                for tt in range(nb_unique_tasks):
                    print(f"  Task{tt + 1}: (noise2, boost2) = " 
                          f"({conf_noise_vals[tt]:7.3f}, {conf_boost_vals[tt]:7.3f})")
        
        # Pick initial boost value that led to highest likelihood
        ii = np.argmax(loglike_list)
        paramBest2 = paramBest_mat[ii, :]
        paramBest_struct = pack_params_in_struct(paramBest2, params_set2, fixed_values) # reconstructs selected T2 parameter dictionary
        # Rebuild full free-parameter vector from the global params_set map
        paramBest_display = extract_params_from_struct(params_set, paramBest_struct) # combine with already-fitted T1 parameters for full optimisation vector
        loglike = loglike_list[ii, 0]
        
        boost_search_diagnostics = []

        for bb in range(boost2_init_nb):
            bb_struct = pack_params_in_struct(paramBest_mat[bb, :], params_set2, fixed_values)
            bb_init = boost_init_list[bb]

            if isinstance(bb_init, dict): # converts into JSON-friendly format for output file
                bb_init_serialized = {k: list(np.atleast_1d(np.asarray(v, dtype=float)))
                                       for k, v in bb_init.items()}
            else:
                bb_init_arr = np.atleast_1d(np.asarray(bb_init, dtype=float))
                bb_init_serialized = float(bb_init_arr.item()) if bb_init_arr.size == 1 else bb_init_arr.tolist()

            boost_search_diagnostics.append({
                'boost_init': bb_init_serialized,
                'loglike': float(loglike_list[bb, 0]),
                'conf_noise': np.atleast_1d(bb_struct['conf_noise']).tolist(),
                'conf_boost': np.atleast_1d(bb_struct['conf_boost']).tolist(),
                'is_best': bool(bb == ii),
            })
            
        if verbose >= 1:
            print("Best fit of full model:")

            my_params_cell = list(params_set.values())
            my_fld_nms = list(params_set.keys())

            for my_kk in range(1, param_free_nb + 1): # params_set contains the mapping between optimisation vector positions and named model parameters
                my_fld_ind = []

                for xx in my_params_cell:
                    inds = np.where(np.atleast_1d(xx) == my_kk)[0]
                    my_fld_ind.append(inds)

                my_uu = [i for i, inds in enumerate(my_fld_ind) if len(inds) > 0]

                for my_pp in my_uu:
                    my_vv = my_fld_ind[my_pp]

                    for vv in my_vv:
                        print(
                            f"Parameter #{my_kk}: "
                            f"{my_fld_nms[my_pp]:>11s}"
                            f"({vv + 1}) = "
                            f"{paramBest_display[my_kk - 1]:7.3f}"
                        )

            print(f"Full Model Log-Likelihood = {loglike:7.3f}")

        # Extracts final fitted Type 2 parameters
        conf_noise_task = paramBest_struct['conf_noise']
        conf_boost_task = paramBest_struct['conf_boost']
        conf_crit_task  = paramBest_struct['conf_crit']
        intrvl1_bias = paramBest_struct['intrvl_bias']
        conf2_bias = paramBest_struct['conf_bias']

        chosen_full, nn1_full = my_fun_full(paramBest2)

    if verbose >= 2:
        elapsed_time = time.time() - tstart
        print(f"\nElapsed time to perform fit : {elapsed_time:7.3f} sec")
    if verbose >= 1:
        print("\n********************\n")
    
    cfc_struct = {
        'sens_noise': sens_noise_task,
        'sens_crit': sens_crit_task,

        'conf_noise': conf_noise_task,
        'conf_boost': conf_boost_task,
        'conf_crit': conf_crit_task,
        'intrvl_bias': intrvl1_bias,
        'conf_bias': conf2_bias,

        'efficiency': efficiency,
        'equiv_conf_noise_ideal': noise2_ideal,
        'equiv_conf_noise_human': noise2_data,

        'choice_prob_ideal': chosen_ideal,
        'choice_prob_super_ideal': chosen_super_ideal,
        'choice_prob_eff': chosen_eff,
        'choice_prob_model': chosen_full,

        'chosen_pair_ideal': nn1_ideal,
        'choice_pair_super_ideal': nn1_super_ideal,
        'choice_pair_eff': nn1_eff,
        'choice_pair_model': nn1_full,

        'loglike': loglike,
        'boost_search_diagnostics': boost_search_diagnostics
    }

    return cfc_struct