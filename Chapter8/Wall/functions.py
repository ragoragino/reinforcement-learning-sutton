import os
import numpy as np
import math


cur_dir = os.path.dirname(__file__)
os.chdir(cur_dir)


"""
DYNA-Q+ with back-up adjustment
"""


def dynaq_b(height, width, start, end, max_iter, epsilon, alpha,
             gamma, reward, actions, n, kappa, wall_change):
    l_actions = len(actions)
    episode_length = np.zeros(max_iter, dtype=int)
    state_actions = np.zeros((width, height, l_actions), dtype=float)  # keeps Q(s,a) for all possible moves
    model = {}  # model (S, A) values
    visits = np.zeros((width, height, l_actions), dtype=float)  # keeps track for how long
                                                                # the state/action was not visited

    for i in range(max_iter):
        state = start
        steps = 0
        while state != end:
            steps -= 1

            # Finding the max subsequent action
            # from the state according to the soft policy
            if np.random.binomial(1, epsilon) == 1:
                action_i = np.random.choice(np.arange(0, l_actions))
                action = actions[action_i]
            else:
                action_i = np.argmax(state_actions[state[0], state[1], :])
                action = actions[action_i]

            if i < wall_change:
                if state[0] + action[0] < 0 or state[0] + action[0] > width - 1 or \
                   state[1] + action[1] < 0 or state[1] + action[1] > height - 1 or \
                   ((state[0] + action[0]) in (1, 2, 3, 4, 5, 6, 7, 8) and
                        (state[1] + action[1]) == 2):  # BLOCKING WALL
                    new_state = state
                else:
                    new_state = (state[0] + action[0], state[1] + action[1])
            else:
                if state[0] + action[0] < 0 or state[0] + action[0] > width - 1 or \
                   state[1] + action[1] < 0 or state[1] + action[1] > height - 1 or \
                   ((state[0] + action[0]) in (1, 2, 3, 4, 5, 6, 7) and
                        (state[1] + action[1]) == 2):  # BLOCKING WALL
                    new_state = state
                else:
                    new_state = (state[0] + action[0], state[1] + action[1])

            # Policy evaluation step
            new_state_v = np.max(state_actions[new_state[0], new_state[1], :])
            td = alpha * (reward + gamma * new_state_v - state_actions[state[0],
                                                                       state[1],
                                                                       action_i])
            state_actions[state[0], state[1], action_i] += td

            model[(state, action_i)] = new_state  # updating model -
                                                  # assuming deterministic environment

            # Keeping the track of how long ago state/action pairs were tested
            visits += 1
            visits[state[0], state[1], action_i] -= 1

            # Update
            state = new_state

            # Model planning
            for _ in range(n):
                pick = np.random.randint(0, len(model))
                model_state, model_action_i = list(model.keys())[pick]
                model_new_state = model[(model_state, model_action_i)]
                model_new_state_max = np.max(state_actions[model_new_state[0],
                                             model_new_state[1], :])
                special_reward = reward + kappa * np.sqrt(visits[model_state[0],
                                                                 model_state[1],
                                                                 model_action_i])
                td = alpha * (special_reward + gamma * model_new_state_max
                              - state_actions[model_state[0], model_state[1],
                                              model_action_i])
                state_actions[model_state[0], model_state[1], model_action_i] += td

        episode_length[i] = - steps
        print("EP: {} - STEPS DQB: {}".format(i, steps))
    return episode_length, state_actions


"""
DYNA-Q+ with action selection adjustment
"""


def dynaq_as(height, width, start, end, max_iter, epsilon, alpha,
              gamma, reward, actions, n, kappa, wall_change):
    l_actions = len(actions)
    episode_length = np.zeros(max_iter, dtype=int)
    state_actions = np.zeros((width, height, l_actions), dtype=float)  # keeps Q(s,a) for all possible moves
    model = {}  # model (S, A) values
    visits = np.zeros((width, height, l_actions), dtype=float)  # keeps for how long the state/action

    for i in range(max_iter):
        state = start
        steps = 0
        while state != end:
            steps -= 1

            # Finding the max subsequent action
            # from the state according to the soft policy
            if np.random.binomial(1, epsilon) == 1:
                action_i = np.random.choice(np.arange(0, l_actions))
                action = actions[action_i]
            else:
                # Action selection is adjusted
                action_i = np.argmax(state_actions[state[0], state[1], :] + kappa * np.sqrt(visits[state[0], state[1], :]))
                action = actions[action_i]

            if i < wall_change:
                if state[0] + action[0] < 0 or state[0] + action[0] > width - 1 or \
                   state[1] + action[1] < 0 or state[1] + action[1] > height - 1 or \
                   ((state[0] + action[0]) in (1, 2, 3, 4, 5, 6, 7, 8) and
                        (state[1] + action[1]) == 2):  # BLOCKING WALL
                    new_state = state
                else:
                    new_state = (state[0] + action[0], state[1] + action[1])
            else:
                if state[0] + action[0] < 0 or state[0] + action[0] > width - 1 or \
                   state[1] + action[1] < 0 or state[1] + action[1] > height - 1 or \
                   ((state[0] + action[0]) in (1, 2, 3, 4, 5, 6, 7) and
                        (state[1] + action[1]) == 2):  # BLOCKING WALL
                    new_state = state
                else:
                    new_state = (state[0] + action[0], state[1] + action[1])

            # Policy evaluation step
            new_state_v = np.max(state_actions[new_state[0], new_state[1], :])
            td = alpha * (reward + gamma * new_state_v - state_actions[state[0],
                                                                       state[1],
                                                                       action_i])
            state_actions[state[0], state[1], action_i] += td

            model[(state, action_i)] = new_state  # assuming deterministic environment

            # Keeping the track of how long ago state/action pairs were tested
            visits += 1
            visits[state[0], state[1], action_i] -= 1

            # Update
            state = new_state

            # Model planning
            for _ in range(n):
                pick = np.random.randint(0, len(model))
                model_state, model_action_i = list(model.keys())[pick]
                model_new_state = model[(model_state, model_action_i)]
                model_new_state_max = np.max(state_actions[model_new_state[0],
                                             model_new_state[1], :])
                td = alpha * (reward + gamma * model_new_state_max -
                              state_actions[model_state[0], model_state[1],
                                            model_action_i])
                state_actions[model_state[0], model_state[1], model_action_i] += td

        episode_length[i] = - steps
        print("EP: {} - STEPS DQB2: {}".format(i, steps))

    return episode_length, state_actions
