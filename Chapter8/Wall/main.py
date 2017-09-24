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
    width = 9
    start = (3, 0)
    end = (8, 6)
    max_iter = 6000
    epsilon = 0.1
    alpha = 0.5
    gamma = 1
    reward = -1
    actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    n = 10
    kappa_b = [0.05, 0.01, 0.001]
    kappa_as = [0.1, 0.05, 0.01]
    methods = ['DYNA-Q+ B ' + str(i) for i in kappa_b] + \
              ['DYNA-Q+ AS ' + str(i) for i in kappa_as]
    wall_change = 3000

    episode_length = {i: [] for i in methods}
    state_action = {i: [] for i in methods}

    for i in kappa_b:
        episode_length['DYNA-Q+ B ' + str(i)], state_action['DYNA-Q+ B ' + str(i)] = \
            functions.dynaq_b(height=height, width=width, start=start,
                              end=end, max_iter=max_iter,
                              epsilon=epsilon, alpha=alpha,
                              gamma=gamma, reward=reward,
                              actions=actions, n=n, kappa=i,
                              wall_change=wall_change)

    for i in kappa_as:
        episode_length['DYNA-Q+ AS ' + str(i)], state_action['DYNA-Q+ AS ' + str(i)] = \
            functions.dynaq_as(height=height, width=width, start=start,
                               end=end, max_iter=max_iter,
                               epsilon=epsilon, alpha=alpha,
                               gamma=gamma, reward=reward,
                               actions=actions, n=n, kappa=i,
                               wall_change=wall_change)
    grid = 100
    n_points = int(np.ceil(max_iter / grid))
    episode_mean = {i: np.zeros(n_points, dtype=float) for i in methods}
    for i in range(n_points):
        for met in methods:
            episode_mean[met][i] = episode_length[met][grid * i:grid * i + grid].mean()

    x = np.arange(1, n_points + 1)
    plt.figure(figsize=(16, 12))
    plt.title("Time to find the end of the episode for  {} iterations, by {} "
              "points.".format(max_iter, grid))
    cmap = plt.get_cmap('Paired')
    for i, method in enumerate(episode_length.keys()):
        episode_mean_all = np.round(episode_length[method].mean(), 2)
        plt.plot(x, episode_mean[method], color=cmap(i),
                 label="{} - Mean: {}".format(method, episode_mean_all),
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

            try:
                # Finding the deterministic max action from the state
                fut_action_i = np.argmax(state_action[method][state[0], state[1], :])
                new_action = actions[fut_action_i]
            except IndexError:
                break

            # Effect of wall on the new state
            if state[0] + new_action[0] in (1, 2, 3, 4, 5, 6, 7) and \
               state[1] + new_action[1] == 2:
                new_state = state
            else:
                new_state = (state[0] + new_action[0], state[1] + new_action[1])

        # Plot of the optimal path
        wall = [1, 2, 3, 4, 5, 6, 7, 8]
        plt.figure(figsize=(16, 12))
        plt.scatter(track_x, track_y, color="blue", label="Track")
        plt.scatter(wall, [2] * len(wall), color="magenta",
                    label="Wall after 3000 iterations")
        plt.plot(traj_x, traj_y, color="red", linewidth=1.0, linestyle="-")
        plt.title("Determinsitic optimal policy for {} after "
                  "{} iterations.".format(method, max_iter))
        plt.xticks(np.arange(0, width, 1))
        plt.text(3, 0.1, r'S', fontsize=10)
        plt.text(8, 6.1, r'E', fontsize=10)
        plt.savefig("{}_optimal_policy_base.pdf".format(method), dpi=100,
                    format='pdf')
