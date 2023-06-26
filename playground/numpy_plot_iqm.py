import numpy as np

from matplotlib import pyplot as plt
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

import tikzplotlib


if __name__ == "__main__":
    file_path = "/home/hongyi/Codes/alr_ma/wandb2numpy/wandb_data/metaworld_bbrl_prodmp"
    is_success_path = file_path + "evaluation_is_success.npy"
    is_success = np.load(is_success_path)
    simulation_steps_path = file_path + "simulation_steps.npy"
    simulation_steps = np.load(simulation_steps_path)

    num_frame = 20
    frames = np.floor(np.linspace(1, is_success.shape[-1], num_frame)).astype(int) - 1

    is_success = is_success[:, None, :]
    is_success = is_success[:, :, frames]
    mask = np.isnan(is_success)
    is_success[mask] = 0.0
    frames_scores_dict = {"bp_sac": is_success}
    iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(frames_scores_dict, iqm, reps=5000)
    plot_utils.plot_sample_efficiency_curve(simulation_steps[0, frames], iqm_scores, iqm_cis,
                                            algorithms=["meta_world"], xlabel="Iteration", ylabel="IQM")
    # tikzplotlib.save(f'meta_sac_sparse.tex')
    plt.show()
