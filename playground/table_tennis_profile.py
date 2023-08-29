import matplotlib
import numpy as np
import os

import tikzplotlib
from matplotlib import pyplot as plt
import matplotlib

from wandb2numpy import util

matplotlib.use('TkAgg')
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def read_tt_data(data_path):
    subdict_list = get_immediate_subdirectories(data_path)

    is_success_list = []
    hitting_list = []
    ep_reward_list = []
    global_steps_list = []

    for subdict in subdict_list:
        # subdict = subdict + "/experiment1"
        if "evaluation_is_success_mean.npy" in os.listdir(subdict):
            is_success_path = subdict + "/evaluation_is_success_mean.npy"
            is_success = np.load(is_success_path)
            is_success_list.append(is_success)

        if "evaluation_hit_ball_mean.npy" in os.listdir(subdict):
            hitting = np.load(subdict + "/evaluation_hit_ball_mean.npy")
            hitting_list.append(hitting)

        if "evaluation_hit_ball_mean.npy" in os.listdir(subdict):
            ep_reward = np.load(subdict + "/evaluation_episode_reward_mean.npy")
            ep_reward_list.append(ep_reward)

        if "num_global_steps.npy" in os.listdir(subdict):
            simulation_steps_path = subdict + "/num_global_steps.npy"
            simulation_steps = np.load(simulation_steps_path)
            global_steps_list.append(simulation_steps)

    is_success_array = np.array(is_success_list)
    is_success_array = np.swapaxes(is_success_array, 0, 1)

    hitting_array = np.array(hitting_list)
    hitting_array = np.swapaxes(hitting_array, 0, 1)

    ep_reward_array = np.array(ep_reward_list)
    ep_reward_array = np.swapaxes(ep_reward_array, 0, 1)

    simulation_steps_array = np.array(global_steps_list)
    simulation_steps_array = np.swapaxes(simulation_steps_array, 0, 1)

    return is_success_array, hitting_array, ep_reward_array, simulation_steps_array


def draw_tt_iqm(is_success, simulation_steps, algorithm, method, label):
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    num_frame = 35
    frames = np.floor(np.linspace(1, is_success.shape[-1], num_frame)).astype(
        int) - 1

    is_success = is_success[:, None, :]
    is_success = is_success[:, :, frames]
    mask = np.isnan(is_success)
    is_success[mask] = 0.0
    frames_scores_dict = {algorithm: is_success}
    iqm = lambda scores: np.array(
        [metrics.aggregate_iqm(scores[..., frame]) for frame in
         range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(frames_scores_dict, iqm,
                                                     reps=5000)
    plot_utils.plot_sample_efficiency_curve(simulation_steps[0, frames],
                                            iqm_scores, iqm_cis,
                                            algorithms=[algorithm],
                                            xlabel="Iteration", ylabel="IQM")
    # plt.show()
    tikzplotlib.get_tikz_code(figure=fig)
    tikzplotlib.save(f"table_tennis_{method}_{label}_iqm.tex")


if __name__ == "__main__":
    # method = "bbrl"
    method = "tcp"

    data_path = f"/home/lige/Codes/wandb2numpy/wandb_data/table_tennis_{method}_prodmp"

    is_success, hitting, ep_reward, simulation_steps = read_tt_data(data_path)

    # draw the iqm curve
    reshaped_is_success = np.reshape(is_success, (-1, is_success.shape[-1]))
    smooth_reshaped_is_success = util.smooth(reshaped_is_success, 0.6)

    reshaped_hitting = np.reshape(hitting, (-1, hitting.shape[-1]))
    smooth_reshaped_hitting = util.smooth(reshaped_hitting, 0.1)

    reshaped_ep_reward = np.reshape(ep_reward, (-1, ep_reward.shape[-1]))
    smooth_reshaped_ep_reward = util.smooth(reshaped_ep_reward, 0.6)

    reshaped_simulation_steps = np.reshape(simulation_steps,
                                           (-1, simulation_steps.shape[-1]))

    draw_tt_iqm(smooth_reshaped_is_success, reshaped_simulation_steps, None,
                method, "success")
    draw_tt_iqm(smooth_reshaped_hitting, reshaped_simulation_steps, None,
                method, "hitting")
    draw_tt_iqm(smooth_reshaped_ep_reward, reshaped_simulation_steps, None,
                method, "reward")
