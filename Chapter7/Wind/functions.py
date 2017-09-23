import os
import numpy as np
import math

cur_dir = os.path.dirname(__file__)
os.chdir(cur_dir)

"""
Tree back-up algorithm
"""


def tree_backup(height, width, start, end, max_iter, epsilon, alpha, gamma, reward, actions, n):
    l_actions = len(actions)
    episode_length = np.zeros(max_iter, dtype=int)
    state_actions = np.zeros((width, height, l_actions), dtype=float)  # keeps Q(s,a) for all possible moves

    for i in range(max_iter):
        td_values = np.zeros(n, dtype=float)  # holds the values of \delta_t
        q_values = np.zeros(n + 1, dtype=float)  # holds te values of Q(S_{t+1}, A_{t+1})
        pi_values = np.zeros(n + 1, dtype=float)  # holds values of \pi(A_{t+1}, S_{t+1})
        state_values = [(0, 0) for _ in range(n + 1)]  # holds values of S visited
        action_values = np.zeros(n + 1, dtype=int)  # holds indices of A taken
        tracker = 0
        tau = 0
        T = np.inf
        t = 0
        steps = 0
        state = start
        if np.random.binomial(1, epsilon) == 1:
            action_i = np.random.choice(np.arange(0, l_actions))
            action = actions[action_i]
            q_values[tracker] = state_actions[state[0], state[1], action_i]
            pi_values[tracker] = epsilon / l_actions
        else:
            action_i = np.argmax(state_actions[state[0], state[1], :])
            action = actions[action_i]
            q_values[tracker] = state_actions[state[0], state[1], action_i]
            pi_values[tracker] = 1 - epsilon + epsilon / l_actions

        state_values[tracker] = state
        action_values[tracker] = actions.index(action)

        tracker += 1
        td_tracker = 0

        while tau < (T - 1):
            if t < T:
                steps -= 1

                # Checking whether next action is not moving the state outside the borders
                if state[0] + action[0] < 0 or state[0] + action[0] > width - 1 or \
                   state[1] + action[1] < 0 or state[1] + action[1] > height - 1:
                    new_state = state
                else:
                    new_state = (state[0] + action[0], state[1] + action[1])

                # Effect of winds on the new state
                if state[0] in (3, 4, 5, 8):
                    new_state = (new_state[0], min(new_state[1] + 1, height - 1))
                elif state[0] in (6, 7):
                    new_state = (new_state[0], min(new_state[1] + 2, height - 1))

                state_values[tracker] = new_state

                if new_state == end:
                    T = t + 1
                    td_values[td_tracker] = reward - q_values[tracker - 1]
                else:
                    cur_max = np.max(state_actions[new_state[0], new_state[1], :])
                    td_values[td_tracker] = reward + gamma * (epsilon / l_actions) * \
                                            np.sum(state_actions[new_state[0], new_state[1], :]) + \
                                            gamma * (1 - epsilon) * cur_max - q_values[tracker - 1]

                    if np.random.binomial(1, epsilon) == 1:
                        fut_action_i = np.random.choice(np.arange(0, l_actions))
                        action = actions[fut_action_i]
                        pi_values[tracker] = epsilon / l_actions
                    else:
                        fut_action_i = np.argmax(state_actions[new_state[0], new_state[1], :])
                        action = actions[fut_action_i]
                        pi_values[tracker] = 1 - epsilon + epsilon / l_actions

                    q_values[tracker] = state_actions[new_state[0], new_state[1],
                                                      fut_action_i]
                    action_values[tracker] = fut_action_i

                    state = new_state

            # n-step bootstrapping
            tracker += 1
            td_tracker += 1
            tracker %= (n + 1)
            td_tracker %= n
            tau = t - n + 1
            if tau >= 0:
                e = 1
                G = q_values[tracker]
                tau_end = min(t, T - 1) - tau
                for j in range(tau_end + 1):
                    tracker_i = (tracker + j) % (n + 1)
                    tracker_j = (td_tracker + j) % n
                    G += e * td_values[tracker_j]
                    e *= gamma * pi_values[tracker_i]

                old_value = state_actions[state_values[tracker][0],
                                          state_values[tracker][1],
                                          action_values[tracker]]

                state_actions[state_values[tracker][0],
                              state_values[tracker][1],
                              action_values[tracker]] += alpha * (G - old_value)

            t += 1

        episode_length[i] = - steps

    return episode_length, state_actions