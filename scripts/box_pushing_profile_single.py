import os
import numpy as np

import matplotlib
import tikzplotlib

from matplotlib import pyplot as plt
from wandb2numpy import util

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
matplotlib.use('TkAgg')


def get_immediate_subdirectories(_dir):
    return [os.path.join(_dir, name) for name in os.listdir(_dir)
            if os.path.isdir(os.path.join(_dir, name))]


def read_tt_data(_path: str):
    subdict_list = get_immediate_subdirectories(_path)

    _ep_reward_list = []
    _is_success_list = []
    _is_hit_ball_list = []
    _global_steps_list = []

    for subdict in subdict_list:
        if "reward.npy" in os.listdir(subdict + '/eval'):
            _ep_reward = np.load(subdict + "/eval/reward.npy")
            _ep_reward_list.append(_ep_reward)

        if "is_success.npy" in os.listdir(subdict + '/eval'):
            _is_success = np.load(subdict + "/eval/is_success.npy")
            _is_success_list.append(_is_success)

        if "hit_ball.npy" in os.listdir(subdict + '/eval'):
            _is_hit_ball = np.load(subdict + "/eval/is_success.npy")
            _is_hit_ball_list.append(_is_hit_ball)

        if "num_global_steps.npy" in os.listdir(subdict):
            _global_steps = np.load(subdict + "/num_global_steps.npy")
            _global_steps_list.append(_global_steps)

    _ep_reward_array = np.array(_ep_reward_list)
    _ep_reward_array = np.swapaxes(_ep_reward_array, 0, 1)

    _is_success_array = np.array(_is_success_list)
    _is_success_array = np.swapaxes(_is_success_array, 0, 1)

    _is_hit_ball_array = np.array(_is_hit_ball_list)
    _is_hit_ball_array = np.swapaxes(_is_hit_ball_array, 0, 1)

    _global_steps_array = np.array(_global_steps_list)
    _global_steps_array = np.swapaxes(_global_steps_array, 0, 1)

    return _ep_reward_array, _is_success_array, _is_hit_ball_array, _global_steps_array


def draw_tt_iqm(_metric, _steps, algorithm, method=None, label=None):
    fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    num_frame = 35
    frames = np.floor(np.linspace(1, _metric.shape[-1], num_frame)).astype(int) - 1

    _metric = _metric[:, None, :]
    _metric = _metric[:, :, frames]
    mask = np.isnan(_metric)
    _metric[mask] = 0.0
    frames_scores_dict = {'gpt': _metric[:10], 'rnn': _metric[10:]}
    iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])])
    iqm_scores, iqm_cis = rly.get_interval_estimates(frames_scores_dict, iqm, reps=5000)
    plot_utils.plot_sample_efficiency_curve(_steps[0, frames],
                                            iqm_scores, iqm_cis,
                                            algorithms=['gpt', 'rnn'],
                                            xlabel="Iteration", ylabel="IQM")
    plt.show()
    # tikzplotlib.get_tikz_code(figure=fig)
    # tikzplotlib.save(f"table_tennis_{method}_{label}_iqm.tex")


if __name__ == "__main__":
    data_path = "/home/zeqi_jin/Desktop/thesis/code/wandb2numpy/wandb_data/table_tennis/tt_mdp"
    ep_reward, is_success, is_hit_ball, global_steps = read_tt_data(data_path)

    # draw the iqm curve
    reshaped_reward = np.reshape(ep_reward, (-1, ep_reward.shape[-1]))
    smooth_reshaped_reward = util.smooth(reshaped_reward, 0.6)

    reshaped_hitting = np.reshape(is_hit_ball, (-1, is_hit_ball.shape[-1]))
    smooth_reshaped_hitting = util.smooth(reshaped_hitting, 0.1)

    reshaped_success = np.reshape(is_success, (-1, is_success.shape[-1]))
    smooth_reshaped_success = util.smooth(reshaped_success, 0.6)

    reshaped_global_steps = np.reshape(global_steps, (-1, global_steps.shape[-1]))

    # draw_tt_iqm(smooth_reshaped_reward, reshaped_global_steps, None)
    # draw_tt_iqm(smooth_reshaped_hitting, reshaped_global_steps, None)
    draw_tt_iqm(smooth_reshaped_success, reshaped_global_steps, None)
