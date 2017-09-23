import os
import numpy as np
import matplotlib.pyplot as plt
import functions
import importlib.util

cur_dir = os.path.dirname(__file__)
os.chdir(cur_dir)
np.random.seed(123)

# Loading functions from Chapter 6 relating to SARSA and Q-Learning algorithms
split_dir = cur_dir.split('\\')
above_dir = '\\'.join(split_dir[:-2])
loc_ch6 = importlib.util.spec_from_file_location("functions", above_dir + r"\Chapter6\Wind\functions.py")
module_ch6 = importlib.util.module_from_spec(loc_ch6)
loc_ch6.loader.exec_module(module_ch6)


if __name__ == '__main__':
    """
    ONLY UP, DOWN, RIGHT, LEFT ACTIONS
    """
    height = 7
    width = 10
    start = (0, 3)
    end = (7, 3)
    max_iter = 1000
    epsilon = 0.1
    alpha = 0.5
    gamma = 1
    reward = -1
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    n = [1, 2, 3, 4, 5, 10]
    methods = ['TREE BACKUP ' + str(i) for i in n] + ['Q-LEARNING']

    episode_length = {i: [] for i in methods}
    state_action = {i: [] for i in methods}

    for i in n:
        episode_length['TREE BACKUP ' + str(i)], state_action['TREE BACKUP ' + str(i)] = \
            functions.tree_backup(height=height, width=width, start=start,
                                  end=end, max_iter=max_iter, epsilon=epsilon,
                                  alpha=alpha, gamma=gamma, reward=reward,
                                  actions=actions, n=i)

    episode_length['Q-LEARNING'], state_action['Q-LEARNING'] = \
        module_ch6.q_learn(height=height, width=width, start=start, end=end,
                           max_iter=max_iter, epsilon=epsilon, alpha=alpha,
                           reward=reward, actions=actions)

    grid = 50
    n_points = int(np.ceil(max_iter / grid))
    episode_mean = {i: np.zeros(n_points, dtype=float) for i in methods}
    for i in range(n_points):
        for met in methods:
            episode_mean[met][i] = episode_length[met][n_points * i:n_points * i + grid].mean()

    x = np.arange(1, n_points + 1)
    plt.figure(figsize=(16, 12))
    plt.title("Time to find the end of the episode for {} iterations, by {} points.".format(max_iter, grid))
    cmap = plt.get_cmap('Paired')
    for i, method in enumerate(episode_length.keys()):
        plt.plot(x, episode_mean[method], color=cmap(i),
                 label="{} - Mean: {}".format(method, episode_length[method].mean()),
                 linewidth=1.0, linestyle="-")
        plt.legend()
    plt.savefig("Episode_length.pdf", dpi=100, format='pdf')


