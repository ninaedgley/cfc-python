# CFC Plot Scripts
# Python version of the data-aggregation layer of cfc_plot.m. It builds the per-task Type-1 psychometric summaries (plot_data). 
# Does not render anything, as figures will be specific to the code's application. This translates the reusable data aggregation layer.


import numpy as np

def cfc_plot_data(grouped_data):
   
    stims2 = grouped_data[:, 0:2]
    resps2 = grouped_data[:, 2:4]
    choic2 = grouped_data[:, 4:6]
    tasks2 = grouped_data[:, 6:8]

    tsk_vals = np.unique(tasks2)
    nb_unique_tsk = tsk_vals.size

    nb_knds = grouped_data.shape[0]

    total1 = np.sum(choic2, axis=1)
    total2 = np.tile(total1[:, None], (1, 2))
    fraction_chosen = choic2[:,0] / total1

    # MATLAB pools stimulus/task pairs from both interval columns into a flat list using a complex number as a joint key for the real part (= stimulus) and imaginary part (= task). 
    # This works because MATLAB's unique() on a full matrix flattens it first, using column-major order. Numpy's default flatten/ravel is row-major order, meaning using its default would pair the wrong stim/task values with the wrong flattened positions later.
    # order='F' is required to match MATLAB's linear-indexing convention.
    stimtask = stims2 + 1j * tasks2
    stimtask_flat = stimtask.flatten(order='F')
    stm_vals, stm_ic2 = np.unique(stimtask_flat, return_inverse=True)
    nb_unique_stims2 = stm_vals.size

    # Unique stimulus values for each task
    stims_dim = np.full((nb_unique_tsk, nb_unique_stims2), np.nan)
    stims_ind = np.full((nb_unique_tsk, nb_unique_stims2), np.nan)
    stims_nb = np.zeros(nb_unique_tsk, dtype=int)

    for task_no in range(1, nb_unique_tsk + 1):
        tsk_inds = np.flatnonzero(np.imag(stm_vals) == task_no)
        nbs = tsk_inds.size
        order = np.argsort(np.real(stm_vals[tsk_inds]))
        stims_dim[task_no - 1, :nbs] = np.real(stm_vals[tsk_inds])[order]
        stims_ind[task_no - 1, :nbs] = tsk_inds[order]
        stims_nb[task_no - 1] = nbs

    stims_nb_max = stims_nb.max()

    # Internal accumulators for plot_data
    unsorted_prob_lst = np.full((nb_unique_tsk, nb_unique_stims2), np.nan)
    chosen_prob_lst = np.full((nb_unique_tsk, nb_unique_stims2), np.nan)
    declined_prob_lst = np.full((nb_unique_tsk, nb_unique_stims2), np.nan)
    unsorted_count_lst = np.full((nb_unique_tsk, nb_unique_stims2), np.nan)
    chosen_count_lst = np.full((nb_unique_tsk, nb_unique_stims2), np.nan)
    declined_count_lst = np.full((nb_unique_tsk, nb_unique_stims2), np.nan)

    plot_data = []

    for task_no in range(1, nb_unique_tsk + 1):
        nbs = stims_nb[task_no - 1]
        tsk_inds = stims_ind[task_no - 1, :nbs].astype(int)

        for ww in range(nbs):
            inds = (stm_ic2 == tsk_inds[ww])  # matches MATLAB's flat (col-major) mask

            # resps2/total2/choic2 must be flattened the same way (order='F') for `inds` (built from the column-major stimtask_flat) to line up with the right entries.
            resps2_flat = resps2.flatten(order='F')
            total2_flat = total2.flatten(order='F')
            choic2_flat = choic2.flatten(order='F')

            nb_resps = np.sum(total2_flat[inds])
            rsp_prob = np.sum(resps2_flat[inds] * total2_flat[inds]) / nb_resps
            unsorted_prob_lst[task_no - 1, ww] = rsp_prob
            unsorted_count_lst[task_no - 1, ww] = nb_resps

            nb_chosen = np.sum(choic2_flat[inds])
            chosen_prob_lst[task_no - 1, ww] = np.sum(resps2_flat[inds] * choic2_flat[inds]) / nb_chosen
            chosen_count_lst[task_no - 1, ww] = nb_chosen

            decli2_flat = total2_flat - choic2_flat
            nb_declined = np.sum(decli2_flat[inds])
            declined_prob_lst[task_no - 1, ww] = np.sum(resps2_flat[inds] * decli2_flat[inds]) / nb_declined
            declined_count_lst[task_no - 1, ww] = nb_declined

        plot_data.append({
            "task": tsk_vals[task_no - 1],
            "sensory_strength": stims_dim[task_no - 1, :nbs].copy(),
            "unsorted_prob": unsorted_prob_lst[task_no - 1, :nbs].copy(),
            "chosen_prob": chosen_prob_lst[task_no - 1, :nbs].copy(),
            "declined_prob": declined_prob_lst[task_no - 1, :nbs].copy(),
            "unsorted_count": unsorted_count_lst[task_no - 1, :nbs].copy(),
            "chosen_count": chosen_count_lst[task_no - 1, :nbs].copy(),
            "declined_count": declined_count_lst[task_no - 1, :nbs].copy(),
        })

    # Type-2 (confidence choice) data, per pair of tasks -> plot_data_intrvl
    # response-pairs are indexed : (r1, r2) > 1, 2, 3, or 4, matching MATLAB's bin2dec(char(resp_pair + '0')) + 1. These indices are equivalent to:
    # 1 = (0,0)
    # 2 = (0,1)
    # 3 = (1,0)
    # 4 = (1,1)
    
    def resp_pair_to_index(r1, r2):
        return int(r1) * 2 + int(r2) + 1
    
    tsk_pairs_vals = np.unique(tasks2, axis=0)
    nb_unique_tsk_pairs = tsk_pairs_vals.shape[0]

    human_choices   = np.full((stims_nb_max, stims_nb_max, 4, nb_unique_tsk_pairs), np.nan)
    intrvl1_counts  = np.zeros((stims_nb_max, stims_nb_max, 4, nb_unique_tsk_pairs))
    intrvl1_freq1   = np.zeros((stims_nb_max, stims_nb_max, 4, nb_unique_tsk_pairs))

    for kk in range(nb_knds):
        stim_pair = stims2[kk, :]
        resp_pair = resps2[kk, :]
        choi_pair = choic2[kk, :]
        task_pair = tasks2[kk, :]

        resp_ind = resp_pair_to_index(resp_pair[0], resp_pair[1]) - 1  # 0-indexed

        # index of this task pair within tsk_pairs_vals
        task_ind = np.flatnonzero(np.all(tsk_pairs_vals == task_pair, axis=1))[0]

        stim1_ind = np.flatnonzero(stims_dim[int(task_pair[0]) - 1, :] == stim_pair[0])[0]
        stim2_ind = np.flatnonzero(stims_dim[int(task_pair[1]) - 1, :] == stim_pair[1])[0]

        human_choices[stim1_ind, stim2_ind, resp_ind, task_ind] = fraction_chosen[kk]
        intrvl1_freq1[stim1_ind, stim2_ind, resp_ind, task_ind] = choi_pair[0]
        intrvl1_counts[stim1_ind, stim2_ind, resp_ind, task_ind] = total1[kk]

    plot_data_intrvl = []

    for task_ind in range(nb_unique_tsk_pairs):
        task_pair = tsk_pairs_vals[task_ind, :]
        task1, task2 = int(task_pair[0]), int(task_pair[1])
        nbs1 = stims_nb[task1 - 1]
        nbs2 = stims_nb[task2 - 1]

        unsorted_prob_intrvl1  = np.full(nbs1, np.nan)
        unsorted_prob_intrvl2  = np.full(nbs2, np.nan)
        chosen_prob_intrvl1    = np.full(nbs1, np.nan)
        chosen_prob_intrvl2    = np.full(nbs2, np.nan)
        declined_prob_intrvl1  = np.full(nbs1, np.nan)
        declined_prob_intrvl2  = np.full(nbs2, np.nan)
        unsorted_count_intrvl1 = np.full(nbs1, np.nan)
        unsorted_count_intrvl2 = np.full(nbs2, np.nan)
        chosen_count_intrvl1   = np.full(nbs1, np.nan)
        chosen_count_intrvl2   = np.full(nbs2, np.nan)
        declined_count_intrvl1 = np.full(nbs1, np.nan)
        declined_count_intrvl2 = np.full(nbs2, np.nan)

        # Interval-1 marginal: fix s1, sum over s2 and split by r1.
        # resp_ind (0-indexed):  (0,1) > r1=0 , (2,3) > r1=1
        # intrvl1_freq1 already counts "interval 1 chosen", which becomes the chosen quantity from interval 1's pov
        
        for i1 in range(nbs1):
            resp0 = np.sum(intrvl1_counts[i1, :, [0, 1], task_ind])
            resp1 = np.sum(intrvl1_counts[i1, :, [2, 3], task_ind])
            unsorted_count_intrvl1[i1] = resp0 + resp1
            unsorted_prob_intrvl1[i1] = resp1 / (resp0 + resp1)

            chos0 = np.sum(intrvl1_freq1[i1, :, [0, 1], task_ind])
            chos1 = np.sum(intrvl1_freq1[i1, :, [2, 3], task_ind])
            chosen_count_intrvl1[i1] = chos0 + chos1
            chosen_prob_intrvl1[i1] = chos1 / (chos0 + chos1)

            decl0, decl1 = resp0 - chos0, resp1 - chos1
            declined_count_intrvl1[i1] = decl0 + decl1
            declined_prob_intrvl1[i1] = decl1 / (decl0 + decl1)

        # Interval 2 marginal: fix s2, sum over s1 and split by r2.
        # resp_ind (0-indexed): (0,2) > r2=0, (1,3) > r2=1
        # From interval 2's pov, intrvl1_freq1 counts as "declined" trials. "Chosen" for interval 2 then = total - intrvl1_freq1
        
        for i2 in range(nbs2):
            resp0 = np.sum(intrvl1_counts[:, i2, [0, 2], task_ind])
            resp1 = np.sum(intrvl1_counts[:, i2, [1, 3], task_ind])
            unsorted_count_intrvl2[i2] = resp0 + resp1
            unsorted_prob_intrvl2[i2] = resp1 / (resp0 + resp1)

            decl0 = np.sum(intrvl1_freq1[:, i2, [0, 2], task_ind])
            decl1 = np.sum(intrvl1_freq1[:, i2, [1, 3], task_ind])
            declined_count_intrvl2[i2] = decl0 + decl1
            declined_prob_intrvl2[i2] = decl1 / (decl0 + decl1)

            chos0, chos1 = resp0 - decl0, resp1 - decl1
            chosen_count_intrvl2[i2] = chos0 + chos1
            chosen_prob_intrvl2[i2] = chos1 / (chos0 + chos1)

        # MATLAB only keeps this task pair if interval-1's unsorted prob has no gaps (every stim1 bin has data)
        if not np.any(np.isnan(unsorted_prob_intrvl1)):
            plot_data_intrvl.append({
                "tasks": task_pair.copy(),
                "unsorted_prob_intrvl1": unsorted_prob_intrvl1,
                "unsorted_prob_intrvl2": unsorted_prob_intrvl2,
                "chosen_prob_intrvl1": chosen_prob_intrvl1,
                "chosen_prob_intrvl2": chosen_prob_intrvl2,
                "declined_prob_intrvl1": declined_prob_intrvl1,
                "declined_prob_intrvl2": declined_prob_intrvl2,
                "unsorted_count_intrvl1": unsorted_count_intrvl1,
                "unsorted_count_intrvl2": unsorted_count_intrvl2,
                "chosen_count_intrvl1": chosen_count_intrvl1,
                "chosen_count_intrvl2": chosen_count_intrvl2,
                "declined_count_intrvl1": declined_count_intrvl1,
                "declined_count_intrvl2": declined_count_intrvl2,
            })

    return plot_data, plot_data_intrvl