import numpy as np

"""
Epsilon-soft SARSA optimal control algorithm
"""


def sarsa(height, width, start, end, max_iter, epsilon, alpha, reward, actions):
    l_actions = len(actions)
    episode_length = np.zeros(max_iter, dtype=int)
    state_actions = np.zeros((width, height, l_actions))  # keeps Q(s,a) for all possible moves

    for i in range(max_iter):
        steps = 0
        state = start
        if np.random.binomial(1, epsilon) == 1:
            action_i = np.random.choice(np.arange(0, l_actions))
            action = actions[action_i]
        else:
            action_i = np.argmax(state_actions[state[0], state[1], :])
            action = actions[action_i]

        while state != end:
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

            # Finding the max subsequent action
            # from the new_state according to the epsilon-soft policy
            if np.random.binomial(1, epsilon) == 1:
                fut_action_i = np.random.choice(np.arange(0, l_actions))
                new_action = actions[fut_action_i]
            else:
                fut_action_i = np.argmax(state_actions[new_state[0], new_state[1], :])
                new_action = actions[fut_action_i]

            # Policy evaluation step
            new_state_v = state_actions[new_state[0], new_state[1], fut_action_i]
            cur_action_i = actions.index(action)
            td = alpha * (reward + new_state_v - state_actions[state[0],
                                                               state[1],
                                                               cur_action_i])
            state_actions[state[0], state[1], cur_action_i] += td

            # Update state and action
            state = new_state
            action = new_action

        episode_length[i] = - steps

    return episode_length, state_actions


"""
Epsilon-soft Q-learning optimal control algorithm
"""


def q_learn(height, width, start, end, max_iter, epsilon, alpha, reward, actions):
    l_actions = len(actions)
    episode_length = np.zeros(max_iter, dtype=int)
    state_actions = np.zeros((width, height, l_actions))  # keeps Q(s,a) for all possible moves

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

            # Policy evaluation step
            new_state_v = np.max(state_actions[new_state[0], new_state[1], :])
            td = alpha * (reward + new_state_v - state_actions[state[0],
                                                               state[1],
                                                               action_i])
            state_actions[state[0], state[1], action_i] += td

            # Update
            state = new_state

        episode_length[i] = - steps

    return episode_length, state_actions


def double_q_learn(height, width, start, end, max_iter, epsilon, alpha, reward, actions):
    l_actions = len(actions)
    episode_length = np.zeros(max_iter, dtype=int)
    vers = ['Q1', 'Q2']
    state_actions = {i: np.zeros((width, height, l_actions)) for i in vers}

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
                action_i = np.argmax(state_actions['Q1'][state[0], state[1], :] + \
                                     state_actions['Q2'][state[0], state[1], :])
                action = actions[action_i]

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

            if np.random.binomial(1, 0.5) == 1:
                ver_update = 'Q1'
                ver_max = 'Q2'
            else:
                ver_update = 'Q2'
                ver_max = 'Q1'

            # Policy evaluation step
            new_state_v = np.max(state_actions[ver_max][new_state[0], new_state[1], :])
            td = alpha * (reward + new_state_v - state_actions[ver_update][state[0],
                                                               state[1],
                                                               action_i])
            state_actions[ver_update][state[0], state[1], action_i] += td

            # Update
            state = new_state

        episode_length[i] = - steps

    return episode_length, state_actions['Q1'], state_actions['Q2']