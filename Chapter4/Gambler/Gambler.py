import os
import math
import numpy as np
import matplotlib.pyplot as plt

cur_dir = os.path.dirname(__file__)
os.chdir(cur_dir)
size = 99  # size of the state set (excluding 0 a 100 states)


def value_teration(probability, threshold):
    it = 0  # iteration counter
    old_v = np.zeros(size + 2)  # old v(s)
    new_v = np.zeros(size + 2)  # new v(s)
    actions = np.zeros(size + 2)  # final policies
    old_v[size + 1] = 1
    new_v[size + 1] = 1

    # Value iteration
    while True:
        it += 1
        for i in range(1, size + 1):
            old_v[i] = new_v[i]
            new_v[i] = 0

        indicator = True
        for i in range(1, size + 1):
            if i < (100 - i):
                current_index = i
            else:
                current_index = 100 - i
            for j in range(1, current_index + 1):
                current_value = (1 - probability) * old_v[i - j] + \
                                probability * old_v[i + j]
                if new_v[i] < current_value:
                    new_v[i] = current_value
            if abs(new_v[i] - old_v[i]) > threshold:
                indicator = False

        if indicator:
            break

    for i in range(1, size + 1):
        old_v[i] = 0
        if i < (100 - i):
            current_index = i
        else:
            current_index = 100 - i
        for j in range(1, current_index + 1):
            current_value = (1 - probability) * new_v[i - j] + \
                            probability * new_v[i + j]
            if old_v[i] < current_value:
                old_v[i] = current_value
                actions[i] = j

    return it, new_v, actions


def policy_teration(probability, threshold):
    it = 0  # iteration counter
    old_v = np.zeros(size + 2)  # old v(s)
    new_v = np.zeros(size + 2)  # new v(s)
    action_v = np.zeros(size + 2)  # final policies
    new_actions = np.zeros(size + 2, dtype=int)  # keeps old optimal actions
    old_actions = np.zeros(size + 2, dtype=int)  # keeps updated optimal actions
    old_v[size + 1] = 1
    new_v[size + 1] = 1
    action_v[size + 1] = 1  # default action is to bet 1

    for i in range(1, size + 1):
        old_actions[i] = 1

    # Policy iteration
    while True:
        it += 1

        # Policy evaluation
        while True:
            for i in range(1, size + 1):
                old_v[i] = new_v[i]
                new_v[i] = 0
                old_actions[i] = new_actions[i]

            indicator = True
            for i in range(1, size + 1):
                current_value = (1 - probability) * old_v[i - old_actions[i]] + \
                                probability * old_v[i + old_actions[i]]
                new_v[i] = current_value
                if abs(new_v[i] - old_v[i]) >= threshold:
                    indicator = False

            if indicator:
                break

        # Policy improvement
        indicator = True
        for i in range(1, size + 1):
            action_v[i] = 0

        for i in range(1, size + 1):
            if i < (100 - i):
                current_index = i
            else:
                current_index = 100 - i
            for j in range(1, current_index + 1):
                current_value = (1 - probability) * (new_v[i - j]) \
                                + probability * (new_v[i + j])
                if action_v[i] < current_value:
                    action_v[i] = current_value
                    new_actions[i] = j

            if old_actions[i] != new_actions[i]:
                indicator = False

        if indicator:
            break

    return it, new_v, new_actions


if __name__ == '__main__':
    prob_h = [0.25, 0.4, 0.55]
    threshold = 0.00001
    for i in prob_h:
        iter_v, value_v, policy_v = value_teration(i, threshold)
        iter_p, value_p, policy_p = policy_teration(i, threshold)
        x = np.arange(0, size + 2, 1)
        fig = plt.figure(figsize=(16, 12), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(x, value_v, color="blue", linewidth=1.0,
                 linestyle="-", label="Number of events")
        plt.title("Values of states for probability of head {} "
                  "for value iteration".format(i))
        plt.subplot(1, 2, 2)
        plt.plot(x, value_p, color="blue", linewidth=1.0,
                 linestyle="-", label="Number of events")
        plt.title("Values of states for probability of head {} "
                  "for policy iteration".format(i))
        plt.savefig("values_" + str(i) + ".pdf", bbox_inches='tight',
                    dpi=100, format='pdf')

        fig = plt.figure(figsize=(16, 12), dpi=100)
        plt.subplot(1, 2, 1)
        plt.plot(x, policy_v, color="blue", linewidth=1.0,
                 linestyle="-", label="Number of events")
        plt.title("Optimal policies for probability of head {} "
                  "for value iteration".format(i))
        plt.subplot(1, 2, 2)
        plt.plot(x, policy_p, color="blue", linewidth=1.0,
                 linestyle="-", label="Number of events")
        plt.title("Optimal policies for probability of head {} "
                  "for policy iteration".format(i))
        plt.savefig("policies_" + str(i) + ".pdf", bbox_inches='tight',
                    dpi=100, format='pdf')

