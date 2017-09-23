import os
import numpy as np
import matplotlib.pyplot as plt
import functions

cur_dir = os.path.dirname(__file__)
os.chdir(cur_dir)
np.random.seed(123)


if __name__ == '__main__':
    """
    ONLY UP, DOWN, RIGHT, LEFT ACTIONS
    """
    height = 7
    width = 10
    start = (0, 3)
    end = (7, 3)
    max_iter = 100000
    epsilon = 0.1
    alpha = 0.5
    reward = -1
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    methods = ['SARSA', 'Q-LEARNING', '2 Q-LEARNING']

    episode_length = {i: [] for i in methods}
    state_action = {i: [] for i in methods}

    episode_length['SARSA'], state_action['SARSA'] = functions.sarsa(height=height, width=width,
                                                           start=start, end=end, max_iter=max_iter,
                                                           epsilon=epsilon, alpha=alpha,
                                                           reward=reward, actions=actions)

    episode_length['Q-LEARNING'], state_action['Q-LEARNING'] = functions.q_learn(height=height, width=width,
                                                                       start=start, end=end, max_iter=max_iter,
                                                                       epsilon=epsilon, alpha=alpha,
                                                                       reward=reward, actions=actions)

    episode_length['2 Q-LEARNING'], state_action['2 Q-LEARNING'], _ = functions.double_q_learn(height=height, width=width,
                                                                                 start=start, end=end,
                                                                                 max_iter=max_iter,
                                                                                 epsilon=epsilon, alpha=alpha,
                                                                                 reward=reward, actions=actions)

    grid = 1000
    n_points = int(np.ceil(max_iter / grid))
    episode_mean = {i: np.zeros(n_points, dtype=float) for i in methods}
    for i in range(n_points):
        for met in methods:
            episode_mean[met][i] = episode_length[met][n_points * i:n_points * i + grid].mean()

    x = np.arange(1, n_points + 1)
    plt.figure(figsize=(16, 12))
    plt.title("Time to find the end of the episode for  {} iterations, by {} points.".format(max_iter, grid))
    cmap = plt.get_cmap('Paired')
    for i, method in enumerate(episode_length.keys()):
        plt.plot(x, episode_mean[method], color=cmap(i),
                 label="{} - Mean: {}".format(method, episode_length[method].mean()),
                 linewidth=1.0, linestyle="-")
        plt.legend()
    plt.savefig("Episode_length_base.pdf", dpi=100, format='pdf')

    # The rectangle world
    track_x = []
    track_y = []
    for i in range(width):
        for j in range(height):
            track_x.append(i)
            track_y.append(j)

    for method in state_action.keys():
        new_state = start
        traj_x = []
        traj_y = []
        # k there just to make sure the loop ends,
        # if there would be no optimal path
        k = 0
        while k < 25:
            k += 1
            state = new_state
            traj_x.append(new_state[0])
            traj_y.append(new_state[1])
            if state == end:
                break

            # Avoiding cases where the new state might be outside of the environment borders
            try:
                # Finding the deterministic max action from the state
                fut_action_i = np.argmax(state_action[method][state[0], state[1], :])
                new_action = actions[fut_action_i]
            except IndexError:
                break

            new_state = (state[0] + new_action[0], state[1] + new_action[1])

            # Effect of winds on the new state
            if state[0] in (3, 4, 5, 8):
                new_state = (new_state[0], min(new_state[1] + 1, height - 1))
            elif state[0] in (6, 7):
                new_state = (new_state[0], min(new_state[1] + 2, height - 1))

        # Plot of the optimal path
        winds = [1, 1, 1, 2, 2, 1]
        plt.figure(figsize=(16, 12))
        plt.scatter(track_x, track_y, color="blue", label="Track")
        plt.plot(traj_x, traj_y, color="red", linewidth=1.0, linestyle="-")
        plt.title("Determinsitic optimal policy for {} after {} iterations.".format(method, max_iter))
        plt.xticks(np.arange(0, width, 1))
        plt.text(-1, -0.8, 'WIND', fontsize=10, color='navy')
        for i, wind in enumerate(winds):
            plt.text(3 + i, -0.8, wind, fontsize=10, color='navy')
        plt.text(0, 3.1, r'S', fontsize=10)
        plt.text(7, 3.1, r'E', fontsize=10)
        plt.savefig("{}_optimal_policy_base.pdf".format(method), dpi=100, format='pdf')