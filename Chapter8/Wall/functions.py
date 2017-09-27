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
    # state_actions keeps Q(s,a) for all possible moves
    state_actions = np.zeros((width, height, l_actions), dtype=float)
    model = {}  # model (S, A) values
    # visits keeps track for how long the state/action was not visited
    visits = np.zeros((width, height, l_actions), dtype=float)

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

            model[(state, action_i)] = new_state  # update of the model

            # Keeping track of how long ago state/action pairs were tested
            visits += 1
            visits[state[0], state[1], action_i] = 0

            # Update
            state = new_state

            # Model planning
            for _ in range(n):
                pick_0 = np.random.randint(0, width)
                pick_1 = np.random.randint(0, height)
                pick_a = np.random.randint(0, 4)
                model_state = (pick_0, pick_1)
                model_action_i = pick_a
                try:
                    model_new_state = model[(model_state, model_action_i)]
                    special_reward = reward + kappa * np.sqrt(visits[model_state[0],
                                                                     model_state[1],
                                                                     model_action_i])
                except KeyError:
                    model_new_state = model_state
                    special_reward = 0
                model_new_state_max = np.max(state_actions[model_new_state[0],
                                             model_new_state[1], :])
                td = alpha * (special_reward + gamma * model_new_state_max -
                              state_actions[model_state[0], model_state[1],
                                            model_action_i])
                state_actions[model_state[0], model_state[1], model_action_i] += td

        episode_length[i] = - steps

    return episode_length, state_actions


"""
DYNA-Q+ with action selection adjustment
"""


def dynaq_as(height, width, start, end, max_iter, epsilon, alpha,
              gamma, reward, actions, n, kappa, wall_change):
    l_actions = len(actions)
    episode_length = np.zeros(max_iter, dtype=int)
    # state_actions keeps Q(s,a) for all possible moves
    state_actions = np.zeros((width, height, l_actions), dtype=float)
    model = {}  # model (S, A) values
    # visits keeps for how long the state/action was not visited
    visits = np.zeros((width, height, l_actions), dtype=float)

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
                # Action selection is adjusted by the visits metric
                action_i = np.argmax(state_actions[state[0], state[1], :] +
                                     kappa * np.sqrt(visits[state[0],
                                                            state[1], :]))
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

            model[(state, action_i)] = new_state  # model update

            # Keeping track of how long ago state/action pairs were visited
            visits += 1
            visits[state[0], state[1], action_i] = 0

            # Update
            state = new_state

            # Model planning
            for _ in range(n):
                pick_0 = np.random.randint(0, width)
                pick_1 = np.random.randint(0, height)
                pick_a = np.random.randint(0, 4)
                model_state = (pick_0, pick_1)
                model_action_i = pick_a
                try:
                    model_new_state = model[(model_state, model_action_i)]
                    special_reward = reward
                except KeyError:
                    model_new_state = model_state
                    special_reward = 0
                model_new_state_max = np.max(state_actions[model_new_state[0],
                                             model_new_state[1], :])
                td = alpha * (special_reward + gamma * model_new_state_max -
                              state_actions[model_state[0], model_state[1],
                                            model_action_i])
                state_actions[model_state[0], model_state[1], model_action_i] += td

        episode_length[i] = - steps

    return episode_length, state_actions
