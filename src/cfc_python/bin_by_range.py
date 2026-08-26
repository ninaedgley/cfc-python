# cfc_bin_by_range.py
#
# cfc_group's `bins` argument bins stimulus intensity per task using np.quantile — equal
# population per bin, wherever the data happens to be denser.
#
# This module implements equal-width binning per task instead, as a preprocessing step
# ahead of cfc_group. Same contract as cfc_group's own bins path (pools both intervals per
# task to define bin edges, replaces each stimulus value with its bin's mean).


import numpy as np

def bin_by_width(raw_data, width_by_task, center='mean'):
    raw_data = np.asarray(raw_data, dtype=float).copy()
    nb_trials = raw_data.shape[0]

    stim_all = np.concatenate([raw_data[:, 0], raw_data[:, 1]])
    task_all = np.concatenate([raw_data[:, 6], raw_data[:, 7]])
    binned_all = np.full_like(stim_all, np.nan)

    for task, width in width_by_task.items():
        mask = task_all == task
        vals = stim_all[mask]

        if vals.size == 0:
            continue
        lo = vals.min()

        # bin edges anchored at the task's own minimum, equal width across its own range
        edges = lo + width * np.arange(0, int(np.ceil((vals.max() - lo) / width)) + 2)
        bin_idx = np.digitize(vals, edges[1:-1])  # interior edges only, so idx in [0, n_bins-1]
        centers = np.full(bin_idx.max() + 1, np.nan)
        
        for b in range(bin_idx.max() + 1):
            in_bin = bin_idx == b
            if not in_bin.any():
                continue
            if center == 'mean':
                centers[b] = vals[in_bin].mean()
            else:
                centers[b] = edges[b] + width / 2.0

        binned_all[mask] = centers[bin_idx]

    raw_data[:, 0] = binned_all[:nb_trials]
    raw_data[:, 1] = binned_all[nb_trials:]
    return raw_data


def _cell_stats(cols, n_trials):
    _, counts = np.unique(cols, axis=0, return_counts=True)
    return {
        'n_cells': len(counts),
        'trials_per_cell_mean': n_trials / len(counts),
        'trials_per_cell_min': int(counts.min()),
        'trials_per_cell_p10': float(np.percentile(counts, 10)),
        'trials_per_cell_median': float(np.median(counts)),
    }


def n_cells_all_task_pairs(raw_data, width_by_task):
    binned = bin_by_width(raw_data, width_by_task)
    return _cell_stats(binned[:, [0, 1, 2, 3, 6, 7]], binned.shape[0])


def n_cells_unbinned(raw_data): 
    return _cell_stats(raw_data[:, [0, 1, 2, 3, 6, 7]], raw_data.shape[0])