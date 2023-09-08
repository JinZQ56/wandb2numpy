import os
import numpy as np

import matplotlib
import tikzplotlib
import seaborn as sns

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

    _ep_reward_dict = {}
    _is_success_dict = {}
    _is_hit_ball_dict = {}
    _global_steps_dict = {}
    _names = []

    for subdict in subdict_list:
        _name = subdict.split("/")[-1]
        _names.append(_name)
        if "reward.npy" in os.listdir(subdict + '/eval'):
            _ep_reward = np.load(subdict + "/eval/reward.npy")
            _ep_reward_dict[_name] = _ep_reward

        if "is_success.npy" in os.listdir(subdict + '/eval'):
            _is_success = np.load(subdict + "/eval/is_success.npy")
            _is_success_dict[_name] = _is_success

        if "hit_ball.npy" in os.listdir(subdict + '/eval'):
            _is_hit_ball = np.load(subdict + "/eval/is_success.npy")
            _is_hit_ball_dict[_name] = _is_hit_ball

        if "num_global_steps.npy" in os.listdir(subdict):
            _global_steps = np.load(subdict + "/num_global_steps.npy")
            _global_steps_dict[_name] = _global_steps

    for _name in _names:
        _ep_reward_array = np.array(_ep_reward_dict[_name])
        # _ep_reward_array = np.swapaxes(_ep_reward_array, 0, 1)
        _ep_reward_dict[_name] = _ep_reward_array

        _is_success_array = np.array(_is_success_dict[_name])
        # _is_success_array = np.swapaxes(_is_success_array, 0, 1)
        _is_success_dict[_name] = _is_success_array

        _is_hit_ball_array = np.array(_is_hit_ball_dict[_name])
        # _is_hit_ball_array = np.swapaxes(_is_hit_ball_array, 0, 1)
        _is_hit_ball_dict[_name] = _is_hit_ball_array

        _global_steps_array = np.array(_global_steps_dict[_name])
        # _global_steps_array = np.swapaxes(_global_steps_array, 0, 1)
        _global_steps_dict[_name] = _global_steps_array

    return _ep_reward_dict, _is_success_dict, _is_hit_ball_dict, _global_steps_dict, _names


def draw_tt_iqm(_metric: dict, _steps: dict, _names: list, _num_frame: int = 35, env='mdp', label=None):
    # fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
    fig, ax = plt.subplots()
    color_palette = sns.color_palette('colorblind', n_colors=len(_names))
    colors = dict(zip(names, color_palette))
    colors = {'mp3_rp_gpt': (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
              'mp3_rp_rnn': (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
              'mp3_rp': (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
              'mp3_bb': (0.8352941176470589, 0.3686274509803922, 0.0)}
    # colors = {'mp3_rp_gpt': (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    #           'mp3_rp_rnn': (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
    #           'mp3_rp': (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
    #           'mp3_bb': (0.8352941176470589, 0.3686274509803922, 0.0)}
    iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame]) for frame in range(scores.shape[-1])])
    s = []
    for _name in _names:
        _frames = np.floor(np.linspace(1, _metric[_name].shape[-1], _num_frame)).astype(int) - 1
        _metric[_name] = _metric[_name][:, None, :]
        _metric[_name] = _metric[_name][:, :, _frames]
        mask = np.isnan(_metric[name])
        _metric[name][mask] = 0.0
        frames_scores_dict = {_name: _metric[_name]}
        iqm_scores, iqm_cis = rly.get_interval_estimates(frames_scores_dict, iqm, reps=5000)
        _s = _steps[_name][0, _frames]
        if len(s) > 0:
            s = _s if s[-1] < _s[-1] else s
        else:
            s = _s
        ax = plot_utils.plot_sample_efficiency_curve(_s,
                                                     iqm_scores,
                                                     iqm_cis,
                                                     ax=ax,
                                                     colors={_name: colors[_name]},
                                                     algorithms=[_name],
                                                     xlabel="",
                                                     ylabel=""
                                                     )
    ax.plot(s, [0.90] * len(s), '--', color='red', linewidth=2)
    # ax.set_xlabel('steps', fontsize=10)
    # ax.set_ylabel('success rate', fontsize=10)
    # plt.show()
    plt.savefig(f"./svg/table_tennis_{env}_iqm_{label}.svg")
    # tikzplotlib.get_tikz_code(figure=fig)
    # tikzplotlib.save(f"./tex/ttable_tennis_{env}_iqm_{label}.tex")


if __name__ == "__main__":
    # data_path = "/home/zeqi_jin/Desktop/thesis/code/wandb2numpy/wandb_data/table_tennis/tt_mdp"
    # data_path = "/home/zeqi_jin/Desktop/thesis/code/wandb2numpy/wandb_data/table_tennis/tt_wind"
    # data_path = "/home/zeqi_jin/Desktop/thesis/code/wandb2numpy/wandb_data/table_tennis/tt_noise"
    # data_path = "/home/zeqi_jin/Desktop/thesis/code/wandb2numpy/wandb_data/table_tennis/tt_mask_vel"
    data_path = "/home/zeqi_jin/Desktop/thesis/code/wandb2numpy/wandb_data/table_tennis/tt_mask_entry"
    reward_dict, success_dict, hitting_dict, steps_dict, names = read_tt_data(data_path)

    smooth_reshaped_reward_dict = {}
    smooth_reshaped_hitting_dict = {}
    smooth_reshaped_success_dict = {}
    reshaped_steps_dict = {}

    for name in names:
        # draw the iqm curve
        reward = reward_dict[name]
        reshaped_reward = np.reshape(reward, (-1, reward.shape[-1]))
        smooth_reshaped_reward = util.smooth(reshaped_reward, 0.6)
        smooth_reshaped_reward_dict[name] = smooth_reshaped_reward

        hitting = hitting_dict[name]
        reshaped_hitting = np.reshape(hitting, (-1, hitting.shape[-1]))
        smooth_reshaped_hitting = util.smooth(reshaped_hitting, 0.1)
        smooth_reshaped_hitting_dict[name] = smooth_reshaped_hitting

        success = success_dict[name]
        reshaped_success = np.reshape(success, (-1, success.shape[-1]))
        smooth_reshaped_success = util.smooth(reshaped_success, 0.6)
        smooth_reshaped_success_dict[name] = smooth_reshaped_success

        steps = steps_dict[name]
        reshaped_steps = np.reshape(steps, (-1, steps.shape[-1]))
        reshaped_steps_dict[name] = reshaped_steps

    # env_type = data_path.split('/')[-1].split('_')[-1]
    # draw_tt_iqm(smooth_reshaped_reward, reshaped_global_steps, None)
    # draw_tt_iqm(smooth_reshaped_hitting, reshaped_global_steps, None)
    draw_tt_iqm(smooth_reshaped_success_dict, reshaped_steps_dict, names, env='mask_entry', label='success')
