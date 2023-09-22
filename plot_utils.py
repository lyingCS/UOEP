from argparse import Namespace
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def smooth(values, window = 3):
    left = window // 2
    new_values = [np.mean(values[max(0,idx-left):min(idx-left+window,len(values))]) for idx in range(len(values))]
    return new_values

def get_training_info(log_path):
    episode = []
    average_total_reward, reward_variance, max_total_reward, min_total_reward, average_n_step, step = [], [], [], [], [], []
    actor_loss, critic_loss = [], []
    with open(log_path, 'r') as infile:
        args = eval(infile.readline())
        for line in tqdm(infile):
            split = line.split('@')
            # episode
            episode.append(eval(split[0].split(':')[1]))
            # episode report
            episode_report = eval(split[1].strip()[len("episode report:"):])
            average_total_reward.append(episode_report['average_total_reward'])
            reward_variance.append(episode_report['reward_variance'])
            max_total_reward.append(episode_report['max_total_reward'])
            min_total_reward.append(episode_report['min_total_reward'])
            average_n_step.append(episode_report['average_n_step'])
            step.append(episode_report['step'])
            # loss report
            loss_report = eval(split[2].strip()[len("step loss (actor,critic):"):])
            actor_loss.append(loss_report[0])
            critic_loss.append(loss_report[1])
    return {
        "episode": episode,
        "average_total_reward": average_total_reward,
        "reward_variance": reward_variance,
        "max_total_reward": max_total_reward,
        "min_total_reward": min_total_reward,
        "average_depth_per_episode": average_n_step,
        "max_depth_per_episode": step,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss
    }

def get_superddpg_training_info(log_path):
    episode = []
    average_total_reward, reward_variance, max_total_reward, min_total_reward, average_n_step, max_n_step, min_n_step \
            = [], [], [], [], [], [], []
    actor_loss, critic_loss, behavior_loss = [], [], []
    with open(log_path, 'r') as infile:
        args = eval(infile.readline())
        for line in tqdm(infile):
            split = line.split('@')
            # episode
            episode.append(eval(split[0].split(':')[1]))
            # episode report
            episode_report = eval(split[1].strip()[len("episode report:"):])
            average_total_reward.append(episode_report['average_total_reward'])
            reward_variance.append(episode_report['reward_variance'])
            max_total_reward.append(episode_report['max_total_reward'])
            min_total_reward.append(episode_report['min_total_reward'])
            average_n_step.append(episode_report['average_n_step'])
            max_n_step.append(episode_report['max_n_step'])
            min_n_step.append(episode_report['min_n_step'])
            # loss report
            loss_report = eval(split[2].strip()[len("step loss:"):])
            actor_loss.append(loss_report['actor_loss'])
            critic_loss.append(loss_report['critic_loss'])
            behavior_loss.append(loss_report['behavior_loss'])
    return {
        "episode": episode,
        "average_total_reward": average_total_reward,
        "reward_variance": reward_variance,
        "max_total_reward": max_total_reward,
        "min_total_reward": min_total_reward,
        "average_depth_per_episode": average_n_step,
        "max_depth_per_episode": max_n_step,
        "min_depth_per_episode": min_n_step,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
        "behavior_loss": behavior_loss
    }

def get_rl_training_info(log_path, training_losses = ['actor_loss', 'critic_loss']):
    episode = []
    average_total_reward, reward_variance, max_total_reward, min_total_reward, average_n_step, max_n_step, min_n_step \
            = [], [], [], [], [], [], []
    training_loss_records = {k: [] for k in training_losses}
    with open(log_path, 'r') as infile:
        args = eval(infile.readline())
        for line in tqdm(infile):
            split = line.split('@')
            # episode
            episode.append(eval(split[0].split(':')[1]))
            # episode report
            episode_report = eval(split[1].strip()[len("episode report:"):])
            average_total_reward.append(episode_report['average_total_reward'])
            reward_variance.append(episode_report['reward_variance'])
            max_total_reward.append(episode_report['max_total_reward'])
            min_total_reward.append(episode_report['min_total_reward'])
            average_n_step.append(episode_report['average_n_step'])
            max_n_step.append(episode_report['max_n_step'])
            min_n_step.append(episode_report['min_n_step'])
            # loss report
            loss_report = eval(split[2].strip()[len("step loss:"):])
            for k in training_losses:
                training_loss_records[k].append(loss_report[k])
    info = {
        "episode": episode,
        "average_total_reward": average_total_reward,
        "reward_variance": reward_variance,
        "max_total_reward": max_total_reward,
        "min_total_reward": min_total_reward,
        "average_depth_per_episode": average_n_step,
        "max_depth_per_episode": max_n_step,
        "min_depth_per_episode": min_n_step
    }
    for k in training_losses:
        info[k] = training_loss_records[k]
    return info
    

def get_offlinesl_training_info(log_path, training_losses = ['training_loss']):
    episode = []
    average_total_reward, reward_variance, max_total_reward, min_total_reward, average_n_step, max_n_step, min_n_step \
            = [], [], [], [], [], [], []
    training_loss_records = {k: [] for k in training_losses}
    with open(log_path, 'r') as infile:
        args = eval(infile.readline())
        for line in tqdm(infile):
            split = line.split('@')
            # episode
            episode.append(eval(split[0].split(':')[1]))
            # episode report
            episode_report = eval(split[1].strip()[len("episode report:"):])
            average_total_reward.append(episode_report['average_total_reward'])
            reward_variance.append(episode_report['reward_variance'])
            max_total_reward.append(episode_report['max_total_reward'])
            min_total_reward.append(episode_report['min_total_reward'])
            average_n_step.append(episode_report['average_n_step'])
            max_n_step.append(episode_report['max_n_step'])
            min_n_step.append(episode_report['min_n_step'])
            # loss report
            loss_report = eval(split[2].strip()[len("step loss:"):])
            for k in training_losses:
                training_loss_records[k].append(loss_report[k])
    info = {
        "episode": episode,
        "average_total_reward": average_total_reward,
        "reward_variance": reward_variance,
        "max_total_reward": max_total_reward,
        "min_total_reward": min_total_reward,
        "average_depth_per_episode": average_n_step,
        "max_depth_per_episode": max_n_step,
        "min_depth_per_episode": min_n_step
    }
    for k in training_losses:
        info[k] = training_loss_records[k]
    return info


def get_model_training_info(prefix, info_getter, smoothness = 100,
                            observe = ['actor_loss', 'critic_loss'],
                            SEED_list = [11,13,17,19,23]):
    seed_info = []
    mean_result = {}
    for SEED in SEED_list:
        expe = prefix + str(SEED)
        log_path = "output/rl4rs/agents/" + expe  + "/model.report"
        info = info_getter(log_path, observe)
        for k in list(info.keys()):
            v = info[k]
            info[k] = smooth(np.array(v), smoothness)

            if k not in mean_result:
                mean_result[k] = info[k][-1]
            else:
                mean_result[k] += info[k][-1]
        seed_info.append(info)
    print('\t'.join([k for k,v in mean_result.items()]))
    print('\t'.join([str(v/len(SEED_list)) for k,v in mean_result.items()]))
    return seed_info


def get_env_training_info(log_path):
    bce_loss, l2_loss = [], []
    with open(log_path, 'r') as infile:
        for line in tqdm(infile):
            try:
                if "loss" in line and "grad_fn" in line:
                    # print(line.split(" "))
                    tmp_split = line.split(" ")
                    bce_loss.append(eval(tmp_split[2].split("(")[1].split(",")[0]))
                    l2_loss.append(eval(tmp_split[6].split("(")[1].split(",")[0]))
            except:
                print(line)
    bce_loss = smooth(bce_loss, 10)
    l2_loss = smooth(l2_loss, 10)
    return {
        "step":np.arange(len(bce_loss)),
        "bce_loss": bce_loss,
        "l2_loss": l2_loss
    }

# draw auc in env
def get_env_auc_training_info(log_path):
    auc_list = []
    with open(log_path, 'r') as infile:
        for line in tqdm(infile):
            try:
                if "auc" in line and "validation" not in line:
                    # print(line.split(" "))
                    tmp_split = line.split(" ")
                    auc_list.append(eval(tmp_split[5][:8]))
            except:
                print(line)
    auc_list = smooth(auc_list, 5)
    return {
        "step":np.arange(len(auc_list)),
        "auc_list": auc_list
    }

def plot_multiple_line(legend_names, list_of_stats, x_name, ncol = 2, row_height = 4):
    '''
    @input:
    - legend_names: [legend]
    - list_of_stats: [{field_name: [values]}]
    - x_name: x-axis field_name
    - ncol: number of subplots in each row
    '''
    plt.rcParams.update({'font.size': 14})
    assert ncol > 0
    features = list(list_of_stats[0].keys())
    features.remove(x_name)
    N = len(features)
    fig_height = 12 // ncol if len(features) == 1 else row_height*((N-1)//ncol+1)
    plt.figure(figsize = (16, fig_height))
    for i,field in enumerate(features):
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        minY,maxY = float('inf'),float('-inf')
        for j,L in enumerate(legend_names):
            X = list_of_stats[j][x_name]
            value_list = list_of_stats[j][field]
            minY,maxY = min(minY,min(value_list)),max(maxY,max(value_list))
            plt.plot(X[:len(value_list)], value_list, label = L)
        plt.ylabel(field)
        plt.xlabel(x_name)
        scale = 1e-4 + maxY - minY
        plt.ylim(minY - scale * 0.05, maxY + scale * 0.05)
        plt.legend()
    plt.show()
    
def plot_multiple_line_with_smooth(legend_names, list_of_stats, x_name, ncol = 2, row_height = 4, window = None):
    '''
    @input:
    - legend_names: [legend]
    - list_of_stats: [{field_name: [values]}]
    - x_name: x-axis field_name
    - ncol: number of subplots in each row
    '''
    if window:
        for dic in list_of_stats:
            for k, v in dic.items():
                dic.update(k, smooth(v, window))
        # list_of_stats = [dic.update(k, smooth(v, window)) for k,v in dic.items() for dic in list_of_stats]
    plt.rcParams.update({'font.size': 14})
    assert ncol > 0
    features = list(list_of_stats[0].keys())
    features.remove(x_name)
    N = len(features)
    fig_height = 12 // ncol if len(features) == 1 else row_height*((N-1)//ncol+1)
    plt.figure(figsize = (16, fig_height))
    X = list_of_stats[0][x_name]
    for i,field in enumerate(features):
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        minY,maxY = float('inf'),float('-inf')
        for j,L in enumerate(legend_names):
            value_list = list_of_stats[j][field]
            minY,maxY = min(minY,min(value_list)),max(maxY,max(value_list))
            plt.plot(X[:len(value_list)], value_list, label = L)
        plt.ylabel(field)
        plt.xlabel(x_name)
        scale = 1e-4 + maxY - minY
        plt.ylim(minY - scale * 0.05, maxY + scale * 0.05)
        plt.legend()
    plt.show()


def plot_mean_var_line(legend_names, list_of_stats, x_name, ncol = 2, row_height = 4, 
                       window = None, with_x = True, font_size = 20, legend_idx = -1):        # draw mean in different seed
    '''
    @input:
    - legend_names: [legend]
    - list_of_stats: [[{field_name: [values]}]]
    - x_name: x-axis field_name
    - ncol: number of subplots in each row
    '''
    color_lib = [("r", "salmon"), ("g", "springgreen"), ("b", "dodgerblue"), ("y", "lightyellow"), ('black', 'lightgrey'), ('purple', 'magenta')]
    plt.rcParams.update({'font.size': 14})
    features = list(list_of_stats[0][0].keys())
    features.remove(x_name)
    seeds_len = list(list_of_stats[0])
    N = len(features)
    X = list_of_stats[0][0][x_name]
    fig_height = 12 // ncol if len(features) == 1 else row_height*((N-1)//ncol+1)
    plt.figure(figsize = (16, fig_height))
    for i,field in enumerate(features):
        plt.subplot((N-1)//ncol+1,ncol,i+1)
        minY,maxY = float('inf'),float('-inf')
        for j,L in enumerate(legend_names):
            mean_map = [[] for _ in range(len(X))]
            for seed in range(len(list_of_stats[j])):
                for k, v in enumerate(list_of_stats[j][seed][field]):
                    mean_map[k].append(v)
            mean_curve = []
            up_curve = []
            down_curve = []
            half = len(list_of_stats[0]) // 2
            if len(list_of_stats[0]) != 1:
                for v in mean_map:
                    mean_curve.append(np.mean(v))
                    down_curve.append(np.mean(sorted(v)[:half]))
                    up_curve.append(np.mean(sorted(v)[len(list_of_stats[0]) - half:]))
            else:
                for v in mean_map:
                    mean_curve.append(np.mean(v))
                    down_curve = mean_curve
                    up_curve = mean_curve
            if window:
                mean_curve = smooth(mean_curve, window)
                down_curve = smooth(down_curve, window)
                up_curve = smooth(up_curve, window)
            mean_curve = np.array(mean_curve)
            up_curve = np.array(up_curve)
            down_curve = np.array(down_curve)
            minY,maxY = min(down_curve.min(), minY), max(up_curve.max(), maxY)
            plt.plot(X, mean_curve, color=color_lib[j % len(color_lib)][0], linewidth=1.0, label=L)
            plt.fill_between(X, up_curve, down_curve, facecolor=color_lib[j % len(color_lib)][1], alpha=0.3)
        plt.ylabel(field, fontsize = font_size)
        if with_x:
            plt.xlabel(x_name, fontsize = font_size)
        scale = 1e-4 + maxY - minY
        try:
            plt.ylim(minY - scale * 0.05, maxY + scale * 0.05)
        except:
            print('ylim:', minY, maxY, scale)
        if legend_idx < 0 or i == legend_idx:
            plt.legend(fontsize = font_size)
    plt.show()

