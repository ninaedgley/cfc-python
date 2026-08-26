# cfc_group takes raw-data trials from experimental datasets, and replaces them with group-level summary stats / organised pairs / hyperparameters

# Input
# `raw_data` (output of cfc_simul_discrim.py), which specifies stimulus intensities, perceptual decisions, confidence judgements, and task
    # Col 0: stimulus intensity interval 1
    # Col 1: stimulus intensity interval 2
    # Col 2: perceptual decision interval 1 (1 = 'A', 0 = 'B')
    # Col 3: perceptual decision interval 2 (1 = 'A', 0 = 'B')
    # Col 4: confidence choice for interval 1 (1 = chosen, 0 = declined)
    # Col 5: confidence choice for interval 2 (1 = chosen, 0 = declined)
    # Col 6: stimulus task for interval 1 (optional)
    # Col 7: stimulus task for interval 2 (optional)

# Optional arguments
# 1. `bins` - pools stim intensities into `bins` quantile bins per task. Replaces each intensity with bin mean before grouping, best used when stimuli take continuous values within a range
# 2. merge_2intervals - merges trials from both intervals into a common pool. Dangerous if an interval bias is present in the dataset, risks systematically biasing dataset


# Output
# `grouped_data` - one row per unique pair of stimulus intensities, with cols 4/5 now counts (columns outlined below) 
    # Col 0: stimulus intensity interval 1
    # Col 1: stimulus intensity interval 2
    # Col 2: perceptual decision interval 1
    # Col 3: perceptual decision interval 2
    # Col 4: nb of confidence choices for interval 1
    # Col 5: nb of confidence choices for interval 2
    # Col 6: stimulus task for interval 1
    # Col 7: stimulus task for interval 2
# `conf_choice_prob` - type 2 probabilities of selecting interval 1, per kind
# `interval_bias_pairs` - per symmetric pair, interval 1 proportions / counts (merge-mode only)
# `interval_bias_flag` - interval-bias diagnostic (merge-mode only, NaN otherwise)


import numpy as np
import scipy.stats as stats


def cfc_group (raw_data, bins: int=0, merge_2intervals = False):

    if bins < 0:
        raise ValueError("bins must be a positive scalar")
    if not isinstance(merge_2intervals, bool):
        raise TypeError("merge_2intervals must be boolean")

    raw_data = np.asarray(raw_data, dtype=float) # prevents writing the bin means back into raw_data
    interval_bias_flag = np.nan
    nb_trials = raw_data.shape[0]

    if (raw_data.shape[1] < 7):
        tasks_nn = np.tile([1,1],(nb_trials,1)) # default both intervals to task 1 if unspecified
        raw_data = np.column_stack((raw_data, tasks_nn))

    tsk_vals = np.unique(raw_data[:,6:8]) 
    nb_tasks = len(tsk_vals) # count of unique task labels across both intervals

    # OPTIONAL - Quantile binning of stimulus intensities
    if bins > 0: 
        assert np.array_equal(tsk_vals, np.arange(1, nb_tasks + 1)), "bins branch assumes integer task labels are 1 ... nb_tasks" # binning requires bins <= number of stim levels, otherwise bins can be empty and np.mean() will return NaN. Relevant for discrete stimulus levels
        stm_data = np.vstack([raw_data[:,[0,6]], raw_data[:,[1,7]]])
        stm_mean = np.full([nb_tasks, bins], np.nan)
        inds_mat = np.full((nb_trials, 2), np.nan, dtype=int) # per trial bin index for [int1, int2]
        
        for tt in range (1, nb_tasks+1):
            inds3 = stm_data[:,1] == tt # rows belonging to task tt
            inds4 = np.where(inds3)[0] # positions in the stacked array
            stm_tsk_vals = stm_data[inds3, 0] # stimulus intensities for task tt

            q_levels = np.arange(1, bins) / bins # bins-1 interior quantile levels
            qq = np.quantile(stm_tsk_vals, q_levels) # cut points
            stm_tsk_bin = np.digitize(stm_tsk_vals, qq) + 1 # label in 1 ... bins

            for nn in range(1, bins + 1):
                inds5 = stm_tsk_bin == nn
                stm_mean[tt - 1, nn - 1] = np.mean(stm_tsk_vals[inds5]) # NaN if this bin is empty
                inds6 = inds4[inds5]
                # positions < nb_trials came from interval 1, positions >= nb_trials from interval 2
                inds1 = inds6 < nb_trials
                inds2 = inds6 >= nb_trials
                inds_mat[inds6[inds1], 0] = nn
                inds_mat[inds6[inds2] - nb_trials, 1] = nn

        # Replace each trial's raw intenities with the mean of the bin it fell into
        for rr in range (nb_trials):
            tsk1 = raw_data[rr, 6]
            tsk2 = raw_data[rr, 7]
            raw_data[rr, 0] = stm_mean[int(tsk1) - 1, inds_mat[rr, 0] - 1]
            raw_data[rr, 1] = stm_mean[int(tsk2) - 1, inds_mat[rr, 1] - 1]

    type2_choice_prob = raw_data[:,4] / np.sum(raw_data[:,4:6], axis=1) # group into unique kinds, per (s1, s2, r1, r2) --> P(choose interval 1) per trial. Denominator = 1 for valid CFC trials, division kept general in case of malformed rows

    cols = raw_data[:,[0,1,2,3,6,7]]
    knd_vals, knd_ic = np.unique(cols, axis=0, return_inverse=True)
    knd_ic = np.ravel(knd_ic) # force to 1D array, return_inverse arguments change depending on the numpy version used
    nb_knds = knd_vals.shape[0]

    grouped_data = np.full((nb_knds, 8), np.nan)
    grouped_data[:, [0,1,2,3,6,7]] = knd_vals

    for ww in range (nb_knds): # for each kind, counts interval 1 vs. 2 confidence choices across its trials
        inds = (knd_ic==ww)
        type2_choice_vals = type2_choice_prob[inds]
        nn1 = np.sum(type2_choice_vals) # interval 1 choices
        nn0 = np.sum(1 - type2_choice_vals) # interval 2 choices
        grouped_data[ww,4:6] = [nn1, nn0]

    interval_bias_pairs = np.full([grouped_data.shape[0], 4], np.nan) # populated only if merge_2intervals = True, otherwise remains NaN

    # Optional interval merging, treats the two intervals as interchangeable and collapses each kind with its mirror, so s1<->s2, r1<->r2, and t1<->t2. Dangerous if an interval bias if present, it will average it away. Default value is False
    if merge_2intervals:
        nb_knds2 = 0
        grouped_data2 = np.full_like(grouped_data, np.nan)
        knd_vals2 = np.full_like(knd_vals, np.nan)

        for ww in range(nb_knds):
            uu = knd_vals[ww]
            vv = [uu[1], uu[0], uu[3], uu[2], uu[5], uu[4]] # mirrored kind (cols 4, 5 here are tasks)
            pp = grouped_data[ww, :]
            qq = [pp[1], pp[0], pp[3], pp[2], pp[5], pp[4], pp[7], pp[6]]
            aa = np.all(knd_vals2 == vv, axis=1)
            matches = np.flatnonzero(aa) # replaces MATLAB ismember

            if matches.size:
                bb = matches[0]
                new_vals = pp[[5, 4]] # swapped counts, so they add into the mirror's orientation
                grouped_data2[bb, 4:6] = grouped_data2[bb, 4:6] + new_vals
                interval_bias_pairs[bb,3] = sum(new_vals)
                interval_bias_pairs[bb, 1] = new_vals[0] / interval_bias_pairs[bb, 3]
            else:
                nb_knds2 = nb_knds2 + 1
                knd_vals2[nb_knds2 - 1, :] = uu
                grouped_data2[nb_knds2 - 1,:] = pp
                interval_bias_pairs[nb_knds2 - 1,2] = np.sum(grouped_data2[nb_knds2 - 1, 4:6])
                interval_bias_pairs[nb_knds2 - 1, 0] = \
                    grouped_data2[nb_knds2 - 1, 4] / interval_bias_pairs[nb_knds2 - 1, 2]
        
        grouped_data = grouped_data2[:nb_knds2,:]
        interval_bias_pairs = interval_bias_pairs[:nb_knds2, :]

        for ww in range(nb_knds2): # put each merged pair into a canonical interval order (lower task first, then lower stim)
            pp = grouped_data2[ww, :]
            qq = [pp[1], pp[0], pp[3], pp[2], pp[5], pp[4], pp[7], pp[6]]
            rr = interval_bias_pairs[ww, :]
            ss = [rr[1], rr[0], rr[3], rr[2]]

            if pp[7] < pp[6]:
                grouped_data[ww,:] = qq
                interval_bias_pairs[ww,:] = ss

            elif pp[6] == pp[7]:
                if pp[1] < pp[0]:
                    grouped_data[ww,:] = qq
                    interval_bias_pairs[ww,:] = ss
        
        prob1 = interval_bias_pairs[:,0] # P(choose interval 1), o.g. ordering
        prob2 = interval_bias_pairs[:, 1] # P(choose interval 1), mirrored ordering
        diffe = stats.norm.ppf(prob1) - stats.norm.ppf(prob2)

        interval_bias_flag = np.nanmean(diffe) / np.nanstd(diffe)

        if np.isfinite(interval_bias_flag):
            print(f"\nWARNING: data contains a likely interval bias (z-score = {interval_bias_flag:7.3f}). Keep intervals separated, using grouped_data = cfc_group(raw_data, merge_2intervals=False)\n")
        else:
            print("No interval bias found")
    
    conf_choice_prob = grouped_data[:, 4] / np.sum(grouped_data[:, 4:6], axis=1) # final confidence choice probabilities after any merging

    return grouped_data, conf_choice_prob, interval_bias_pairs, interval_bias_flag