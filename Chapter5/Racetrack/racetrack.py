import matplotlib.pyplot as plt
import numpy as np
import os

cur_dir = os.path.dirname(__file__)
os.chdir(cur_dir)
np.random.seed(123)

height = 32  # height of the track
width = 17  # width of the track
length = 9  # length of the track
# states_actions keeps n and average rewards
states_actions = np.zeros(height * width * length * 2,
                          dtype=float)
states_actions.shape = (width, height, 3, 3, 2)
# states keeps an indicator for viability of the track,
# indicator for a finish and height and width of max
# actions for each state
states = np.ones(height * width * 4, dtype=float)
states.shape = (width, height, 4)
states[:, :, 1] = 0
# Initial \pi(a|s) is (0, 1), i.e. increasing velocity by one
# in the vertical direction
states[:, :, 2] = 0

track_x = []
track_y = []
max_iter = 500000  # Number of iterations
epsilon = 0.4  # Exploration parameter
max_steps = -9  # How many steps to end an episode

# Creating the viable positions on the xy axis
# defined by the grid width x height
for i in range(width):
    for j in range(height):
        if i == 0 and (j < 18 or j >= 28):
            states[i, j, 0] = 0
        if i == 1 and (j < 10 or j >= 29):
            states[i, j, 0] = 0
        if i == 2 and (j < 3 or j >= 31):
            states[i, j, 0] = 0
        if i == 9 and j < 24:
            states[i, j, 0] = 0
        if i > 9 and j < 25:
            states[i, j, 0] = 0

        if i == 16 and j >= 26:  # Finish positions
            states[i, j, 1] = 1

        if states[i, j, 0] != 0:
            track_x.append(i)
            track_y.append(j)

# Starting positions
starting_positions = [j for j in range(width)
                      if states[j, 0, 0] != 0]

# Function that returns the actions according to the
# e-soft policy with regard to velocity constraints
def cdf(velocity, action, epsilon):
    if velocity[0] == 0 or velocity[1] == 0:
        if velocity[0] == 0:
            if velocity[1] == 0:
                actions = [(0, 1), (1, 0), (1, 1)]
            elif velocity[1] == 1:
                actions = [(0, 1), (1, -1), (1, 0), (1, 1)]
            elif velocity[1] == 5:
                actions = [(0, -1), (1, -1), (1, 0)]
            else:
                actions = [(0, -1), (0, 1), (1, -1), (1, 0),
                           (1, 1)]
        else:
            if velocity[0] == 1:
                actions = [(-1, 1), (0, 1), (1, 0), (1, 1)]
            elif velocity[0] == 5:
                actions = [(-1, 0), (-1, 1), (0, 1)]
            else:
                actions = [(-1, 0), (-1, 1), (0, 1), (1, 0),
                           (1, 1)]

    elif velocity[0] == 5 or velocity[1] == 5:
        if velocity[0] == 5:
            if velocity[1] == 5:
                actions = [(-1, -1), (-1, 0), (0, -1)]
            else:
                actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                            (0, 1)]
        else:
            actions = [(-1, -1), (-1, 0), (0, -1),
                       (1, -1), (1, 0)]

    elif velocity[0] == 1 and velocity[1] == 1:
        actions = [(-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)]

    else:
        actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)]

    if action not in actions:
        action = actions[np.random.randint(0, len(actions), 1)[0]]

    pick = np.random.uniform(0, 1)
    actions_index = actions.index(action)
    len_actions = len(actions)
    distribution = [epsilon * i / len_actions
                    for i in range(1, actions_index + 1)]
    distribution.append(actions_index * epsilon / len_actions + \
                        1 - epsilon + epsilon / len_actions)
    distribution += [distribution[-1] + epsilon * i / len_actions
                     for i in range(1, len_actions - actions_index)]

    i = 0
    while distribution[i] < pick:
        i += 1

    return actions[i]

# Main algorithm
for it in range(max_iter):
    trajectory = []
    movements = []
    velocity_l = []
    step = 0
    start = (np.random.choice(starting_positions, 1)[0], 0)
    velocity = (0, 0)
    indicator = False
    while step > max_steps:
        step -= 1
        action = (states[start[0], start[1], 2],
                  states[start[0], start[1], 3])
        movement = cdf(velocity, action, epsilon)
        trajectory.append(start)
        movements.append(movement)
        velocity = tuple(map(sum, zip(movement, velocity)))
        velocity_l.append(velocity)
        new_state = tuple(map(sum, zip(start, velocity)))
        # Not viable
        if new_state[0] < 0 or new_state[0] >= width \
            or new_state[1] < 0 or new_state[1] >= height \
            or states[new_state[0], new_state[1], 0] == 0:
            start = (np.random.choice(starting_positions, 1)[0], 0)
            velocity = (0, 0)
            continue
        # Finish
        elif states[new_state[0], new_state[1], 1] == 1:
            print("ITER: {}".format(it))
            print("TRAJECTORY: {}".format(trajectory))
            print("VELOCITY: {}".format(velocity_l))
            print("MOVEMENT: {}".format(movements))
            indicator = True
            break
        else:
            start = new_state

    state_action_tuple = []
    for i in range(-step):
        # Only first occurrences
        mov_1 = movements[i][0] + 1
        mov_2 = movements[i][1] + 1
        if (trajectory[i], movements[i]) not in state_action_tuple:
            state_action_tuple.append((trajectory[i], movements[i]))
            n = states_actions[trajectory[i][0], trajectory[i][1],
                               mov_1, mov_2, 0]
            states_actions[trajectory[i][0], trajectory[i][1],
                           mov_1, mov_2, 1] = \
                states_actions[trajectory[i][0], trajectory[i][1],
                               mov_1, mov_2, 1] * n / (n + 1) + \
                (i + step) / (n + 1)
            states_actions[trajectory[i][0], trajectory[i][1],
                           mov_1, mov_2, 0] = n + 1
            """
            Without incentivizing finishing/increasing the number of steps/
            changing the max velocity, the agent does not see a point in finishing
            as when finishing he also gets -9. I have decided to implement an
            incentive in order to make him learn that ending in a finish is more optimal
            than other ends.
            """
            if indicator:
                states_actions[trajectory[i][0], trajectory[i][1],
                               mov_1, mov_2, 1] += 0.05

    for i in trajectory:
        run_max = max_steps - 1
        if i[1] == 0:
            allowed_actions = [(0, 1), (1, 0), (1, 1)]
        else:
            allowed_actions = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                                (0, 1), (1, -1), (1, 0), (1, 1)]
        for index in allowed_actions:
            if states_actions[i[0], i[1], index[0] + 1, index[1] + 1, 1] > run_max:
                run_max = states_actions[i[0], i[1], index[0] + 1, index[1] + 1, 1]
                states[i[0], i[1], 2] = index[0]
                states[i[0], i[1], 3] = index[1]

new_state = (5, 0)
traj_30_x = [new_state[0]]
traj_30_y = [new_state[1]]
velocity = (0, 0)
i = 0
while i < -max_steps:
    best_action_1 = states[int(new_state[0]), int(new_state[1]), 2]
    best_action_2 = states[int(new_state[0]), int(new_state[1]), 3]
    velocity = tuple(map(sum, zip((best_action_1, best_action_2), velocity)))
    new_state = tuple(map(sum, zip(new_state, velocity)))
    print(traj_30_x)
    print(traj_30_y)
    traj_30_x.append(new_state[0])
    traj_30_y.append(new_state[1])
    i += 1
    try:
        if states[int(new_state[0]), int(new_state[1]), 0] == 0:
            del traj_30_x[-1]
            del traj_30_y[-1]
            break
    except IndexError:
        del traj_30_x[-1]
        del traj_30_y[-1]
        break

plt.figure(figsize=(16, 12), dpi=100)
plt.scatter(track_x, track_y, color="blue", label="Track")
plt.scatter(traj_30_x, traj_30_y, color="red")
plt.savefig("RACETRACK.pdf", dpi=100, format='pdf')

