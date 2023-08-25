import matplotlib
import numpy as np
import os

import tikzplotlib
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def read_box_dense_world_data(data_path):

    subdict_list = get_immediate_subdirectories(data_path)

    is_success_list = []
    global_steps_list = []

    for subdict in subdict_list:
        # subdict = subdict + "/experiment1"
        if "evaluation_is_success_mean.npy" in os.listdir(subdict):
            is_success_path = subdict + "/evaluation_is_success_mean.npy"
            is_success = np.load(is_success_path)
            is_success_list.append(is_success)
        if "num_global_steps.npy" in os.listdir(subdict):
            simulation_steps_path = subdict + "/num_global_steps.npy"
            simulation_steps = np.load(simulation_steps_path)
            global_steps_list.append(simulation_steps)

    # b = np.zeros((4, 304))
    # a = is_success_list[21]
    # b[:3] = a
    # b[-1] = a[0]
    # is_success_list[21] = b

    is_success_array = np.array(is_success_list)


    is_success_array = np.swapaxes(is_success_array, 0, 1)

    # b = np.zeros((4, 304))
    # a = global_steps_list[21]
    # b[:3] = a
    # b[-1] = a[0]
    # global_steps_list[21] = b

    simulation_steps_array = np.array(global_steps_list)
    simulation_steps_array = np.swapaxes(simulation_steps_array, 0, 1)

    return is_success_array, simulation_steps_array


def draw_box_pushing_iqm(is_success, simulation_steps, algorithm):
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    num_frame = 15
    frames = np.floor(np.linspace(1, is_success.shape[-1], num_frame)).astype(int) - 1

    is_success = is_success[:, None, :]
    is_success = is_success[:, :, frames]
    mask = np.isnan(is_success)
    is_success[mask] = 0.0
    frames_scores_dict = {algorithm: is_success}
    iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(frames_scores_dict, iqm, reps=5000)
    plot_utils.plot_sample_efficiency_curve(simulation_steps[0, frames], iqm_scores, iqm_cis,
                                            algorithms=[algorithm], xlabel="Iteration", ylabel="IQM")
    # plot_utils.plot_interval_estimates(simulation_steps[0, frames],
    #                                         iqm_scores, iqm_cis,
    #                                         algorithms=[algorithm],
    #                                         xlabel="Iteration", ylabel="IQM")
    # plt.show()
    tikzplotlib.get_tikz_code(figure=fig)
    # tikzplotlib.save("box_dense_bbrl_iqm.tex")
    # tikzplotlib.save("box_dense_tcp_iqm.tex")
    # tikzplotlib.save("box_t_sparse_bbrl_iqm.tex")
    tikzplotlib.save("box_t_sparse_tcp_iqm.tex")


if __name__ == "__main__":
    # BBRL Dense
    # is_success, simulation_steps = read_box_dense_world_data("/home/lige/Codes/wandb2numpy/wandb_data/box_dense_bbrl_prodmp")
    # TCP Dense
    # is_success, simulation_steps = read_box_dense_world_data("/home/lige/Codes/wandb2numpy/wandb_data/box_dense_tcp_prodmp")

    # BBRL T Sparse
    # is_success, simulation_steps = read_box_dense_world_data("/home/lige/Codes/wandb2numpy/wandb_data/box_temporal_sparse_bbrl_prodmp")
    # TCP T Sparse
    is_success, simulation_steps = read_box_dense_world_data("/home/lige/Codes/wandb2numpy/wandb_data/box_temporal_sparse_tcp_prodmp")

    # draw the iqm curve
    reshaped_is_success = np.reshape(is_success, (-1, is_success.shape[-1]))
    reshaped_simulation_steps = np.reshape(simulation_steps, (-1, simulation_steps.shape[-1]))
    draw_box_pushing_iqm(reshaped_is_success, reshaped_simulation_steps, None)