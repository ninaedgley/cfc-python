# Simulates a confidence-forced choice dataset for a Type 1 discrimination task. Takes `simul_params`, and `model_params`, returning a `raw_data` matrix.

# Inputs
# 1 - `simul_params` : simulation parameters, struct in original MATLAB, dict in Python
    # simul_params(task_no).sens_intens: difficulty levels, constant stimuli
    # simul_params(task_no).sens_intens_min: minimum of stimuli range, uniform sampling
    # simul_params(task_no).sens_intens_max: maximum of stimuli range, uniform sampling
    # simul_params(task_no).method: use of method of constant stimuli ('1' or '0')
    # simul_params(1).nb_trials: number of confidence pairs to simulate

# 2 - `model_params` : model parameter values, structure in original MATLAB, dict in Python
    # 'tasks_list' : vector of tasks, e.g. '1' or '[1, 2]'
    # 'sens_noise' : sensory (Type 1) sdtev of noise (0 = perfectly sensitive)
    # 'sens_crit' : sensory (Type 1) criterion
    # 'conf_noise' : confidence (Type 2) sdtev of noise (0 = ideal)
    # 'conf_boost' : fraction super-ideal (0 = ideal, 1 = super-ideal)
    # 'conf_crit' : confidence (Type 2) criterion
    # 'intrvl_bias' : bias in favour of interval 1 over interval 2
    # 'conf_bias' : overconfidence relative to one of the tasks

# Outputs
# `raw_data`, matrix with:
    # Col 0: stimulus intensity interval 1
    # Col 1: stimulus intensity interval 2
    # Col 2: perceptual decision interval 1 (1 = 'A', 0 = 'B')
    # Col 3: perceptual decision interval 2 (1 = 'A', 0 = 'B')
    # Col 4: confidence choice for interval 1 (1 = chosen, 0 = declined)
    # Col 5: confidence choice for interval 2 (1 = chosen, 0 = declined)
    # Col 6: stimulus task for interval 1 (optional)
    # Col 7: stimulus task for interval 2 (optional)


import numpy as np
import scipy.stats as stats


def cfc_simul_discrim(simul_params, model_params):
    
    # Type 1 parameters - takes input from `model_parameters`
    tasks_list = model_params['tasks_list']
    sens_noise_task = model_params['sens_noise']
    sens_crit_task = model_params['sens_crit']
    nb_tasks = len(tasks_list)

    # Type 2 parameters
    conf_noise_task = model_params['conf_noise']
    conf_boost_task = model_params['conf_boost']
    
    # All below are OPTIONAL parameters
    # Confidence criterion
    if "conf_crit" in model_params:
        conf_crit_task = model_params['conf_crit']
    else:
        conf_crit_task = np.zeros(nb_tasks)

    # Interval bias in favour of interval 1
    if "intrvl_bias" in model_params:
        intrvl_bias = model_params['intrvl_bias']
    else:
        intrvl_bias = 0.0

    # Confidence bias: overconfidence >= 1, relative to a task = 1
    if "conf_bias" in model_params:
        conf_scale = model_params['conf_bias']
    else:
        conf_scale = np.ones(nb_tasks)
    
    # Actual confidence pairs of trials
    nb_trials = simul_params[0]['nb_trials']

    tasks_nn = np.full((nb_trials,2), np.nan)
    sens_intensity_nn = np.full((nb_trials,2), np.nan)
    percs_nn = np.full((nb_trials,2), np.nan)
    choic_nn = np.full((nb_trials,2), np.nan)
    sens_smpl = np.full(2, np.nan)
    conf_evd = np.full(2, np.nan)
    conf_mag = np.full(2, np.nan)
    type1_resp = np.full(2, np.nan)
    type2_resp2 = np.full(2, np.nan)

    for tt in range(nb_trials):

        for intrv in range(2): # interval kk of confidence pair
            ind_task = np.random.randint(nb_tasks)
            tasks_nn[tt, intrv] = tasks_list[ind_task]

            if simul_params[ind_task]['method']:
                sens_intens = simul_params[ind_task]['sens_intens']
                nb_sens_intens = len(sens_intens)
                ind_intens = np.random.randint(nb_sens_intens)
                sens_intensity_nn[tt, intrv] = sens_intens[ind_intens]

            else:
                sens_intens_min = simul_params[ind_task]['sens_intens_min']
                sens_intens_max = simul_params[ind_task]['sens_intens_max']
                stim_val = np.random.rand() * (sens_intens_max - sens_intens_min) + sens_intens_min
                sens_intensity_nn[tt, intrv] = stim_val
            
            # Independent noisy samples of the stimuli
            sens_smpl[intrv] = sens_intensity_nn[tt, intrv] + np.random.randn() * sens_noise_task[ind_task]

            # Sensory decision based on the side of the sensory criterion
            type1_resp[intrv] = sens_smpl[intrv] > sens_crit_task[ind_task]
            percs_nn[tt, intrv] = type1_resp[intrv]

            # Super-ideal observer boosted sensory evidence, with criterion applied
            boosted_sens_smpl = (1 - conf_boost_task[ind_task]) * sens_smpl[intrv] + conf_boost_task[ind_task]*sens_intensity_nn[tt, intrv]
            conf_smpl = boosted_sens_smpl  - sens_crit_task[ind_task] - conf_crit_task[ind_task]

            # Scaling to sensory sensitivity, corrupted with Type 2 noise
            scaled_smpl = conf_smpl / sens_noise_task[ind_task] * conf_scale[ind_task]
            conf_evd[intrv] = scaled_smpl + np.random.randn()*conf_noise_task[ind_task]

            # Confidence magnitude -> pseudo-perceptual decision
            conf_mag[intrv] = np.abs(conf_evd[intrv])
            type2_resp2[intrv] = (conf_evd[intrv] > 0)

            if type2_resp2[intrv] != type1_resp[intrv]:
                conf_mag[intrv] = -conf_mag[intrv] # sets up w' for computing a confidence probability : P(confident | w, D) = cdf(w')
        
        pref_intrvl1 = conf_mag[0] - conf_mag[1] + intrvl_bias
        choice = int(pref_intrvl1 > 0) 
        choice_intrvl = 2 - choice
        choic_nn[tt, choice_intrvl - 1] = 1
        choic_nn[tt, 2 - choice_intrvl] = 0

    raw_data = np.hstack([sens_intensity_nn, percs_nn, choic_nn, tasks_nn])

    return raw_data