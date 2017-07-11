import re
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pylab
os.chdir(r'D:\Materials\Matemathics\ReinforcementLearning\Chapter2')

length = 10000
nrow = 6

bandit = np.empty((length, nrow), dtype=np.float64)
k = 0
with open('bandit.csv') as doc:
    for i in doc:
        split_line = re.split(',', i)
        numeric_vector = [float(i) for i in split_line[:-1]]
        bandit[:, k] = np.array(numeric_vector)
        k += 1

bandit_alpha = np.empty((length, nrow), dtype=np.float64)
k = 0
with open('bandit_alpha.csv') as doc:
    for i in doc:
        split_line = re.split(',', i)
        numeric_vector = [float(i) for i in split_line[:-1]]
        bandit_alpha[:, k] = np.array(numeric_vector)
        k += 1

bandit_classic = np.empty((length, nrow), dtype=np.float64)
k = 0
with open('bandit_classic.csv') as doc:
    for i in doc:
        split_line = re.split(',', i)
        numeric_vector = [float(i) for i in split_line[:-1]]
        bandit_classic[:, k] = np.array(numeric_vector)
        k += 1

bandit_classic_alpha = np.empty((length, nrow), dtype=np.float64)
k = 0
with open('bandit_classic_alpha.csv') as doc:
    for i in doc:
        split_line = re.split(',', i)
        numeric_vector = [float(i) for i in split_line[:-1]]
        bandit_classic_alpha[:, k] = np.array(numeric_vector)
        k += 1


# Plot of average reward for the case of random-walk dynamics of the action values
# Comparison between update parameters alpha = 1 and alpha = 0.1
X = np.arange(1, length + 1, 1)
fig = plt.figure(figsize=(16, 12), dpi=100)
plt.subplot(1, 2, 1)
plt.plot(X, bandit[:, 0], color="blue", linewidth=1.0, linestyle="-", label="epsilon = 0.1")
plt.plot(X, bandit[:, 2], color="red", linewidth=1.0, linestyle="-", label="epsilon = 0.01")
plt.plot(X, bandit[:, 4], color="green", linewidth=1.0, linestyle="-", label="epsilon = 0.0000001")
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.title('Average reward for a 10-bandit algorithm \nwith RW action values, alpha = 1')

plt.subplot(1, 2, 2)
plt.plot(X, bandit_alpha[:, 0], color="blue", linewidth=1.0, linestyle="-", label="epsilon = 0.1")
plt.plot(X, bandit_alpha[:, 2], color="red", linewidth=1.0, linestyle="-", label="epsilon = 0.01")
plt.plot(X, bandit_alpha[:, 4], color="green", linewidth=1.0, linestyle="-", label="epsilon = 0.0000001")
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.title('Average reward for a 10-bandit algorithm \nwith RW action values, alpha = 0.1')
pylab.savefig('average_reward_rw.pdf', bbox_inches='tight', dpi = 100, format = 'pdf')

# Plot of optimal action for the case of random-walk dynamics of the action values
# Comparison between update parameters alpha = 1 and alpha = 0.1
X = np.arange(1, length + 1, 1)
fig = plt.figure(figsize=(16, 12), dpi=100)
plt.subplot(1, 2, 1)
plt.plot(X, bandit[:, 1] * 100, color="blue", linewidth=1.0, linestyle="-", label="epsilon = 0.1")
plt.plot(X, bandit[:, 3] * 100, color="red", linewidth=1.0, linestyle="-", label="epsilon = 0.01")
plt.plot(X, bandit[:, 5] * 100, color="green", linewidth=1.0, linestyle="-", label="epsilon = 0.0000001")
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Optimal action [%]')
plt.title('Optimal action for a 10-bandit algorithm \nwith RW action values, alpha = 1')

plt.subplot(1, 2, 2)
plt.plot(X, bandit_alpha[:, 1] * 100, color="blue", linewidth=1.0, linestyle="-", label="epsilon = 0.1")
plt.plot(X, bandit_alpha[:, 3] * 100, color="red", linewidth=1.0, linestyle="-", label="epsilon = 0.01")
plt.plot(X, bandit_alpha[:, 5] * 100, color="green", linewidth=1.0, linestyle="-", label="epsilon = 0.0000001")
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Optimal action [%]')
plt.title('Optimal action for a 10-bandit algorithm \nwith RW action values, alpha = 0.1')
pylab.savefig('optimal_action_rw.pdf', bbox_inches='tight', dpi = 100, format = 'pdf')

# Plot of average reward for the case of constant action values
# Comparison between update parameters alpha = 1 and alpha = 0.1
X = np.arange(1, length + 1, 1)
fig = plt.figure(figsize=(16, 12), dpi=100)
plt.subplot(1, 2, 1)
plt.plot(X, bandit_classic[:, 0], color="blue", linewidth=1.0, linestyle="-", label="epsilon = 0.1")
plt.plot(X, bandit_classic[:, 2], color="red", linewidth=1.0, linestyle="-", label="epsilon = 0.01")
plt.plot(X, bandit_classic[:, 4], color="green", linewidth=1.0, linestyle="-", label="epsilon = 0.0000001")
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.title('Average reward for a 10-bandit algorithm \nwith constant action values, alpha = 1')

plt.subplot(1, 2, 2)
plt.plot(X, bandit_classic_alpha[:, 0], color="blue", linewidth=1.0, linestyle="-", label="epsilon = 0.1")
plt.plot(X, bandit_classic_alpha[:, 2], color="red", linewidth=1.0, linestyle="-", label="epsilon = 0.01")
plt.plot(X, bandit_classic_alpha[:, 4], color="green", linewidth=1.0, linestyle="-", label="epsilon = 0.0000001")
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.title('Average reward for a 10-bandit algorithm \nwith constant action values, alpha = 0.1')
pylab.savefig('average_reward_c.pdf', bbox_inches='tight', dpi = 100, format = 'pdf')

# Plot of optimal action for the case of constant action values
# Comparison between update parameters alpha = 1 and alpha = 0.1
X = np.arange(1, length + 1, 1)
fig = plt.figure(figsize=(16, 12), dpi=100)
plt.subplot(1, 2, 1)
plt.plot(X, bandit_classic[:, 1] * 100, color="blue", linewidth=1.0, linestyle="-", label="epsilon = 0.1")
plt.plot(X, bandit_classic[:, 3] * 100, color="red", linewidth=1.0, linestyle="-", label="epsilon = 0.01")
plt.plot(X, bandit_classic[:, 5] * 100, color="green", linewidth=1.0, linestyle="-", label="epsilon = 0.0000001")
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Optimal action [%]')
plt.title('Optimal action for a 10-bandit algorithm \nwith constant action values, alpha = 1')

plt.subplot(1, 2, 2)
plt.plot(X, bandit_classic_alpha[:, 1] * 100, color="blue", linewidth=1.0, linestyle="-", label="epsilon = 0.1")
plt.plot(X, bandit_classic_alpha[:, 3] * 100, color="red", linewidth=1.0, linestyle="-", label="epsilon = 0.01")
plt.plot(X, bandit_classic_alpha[:, 5] * 100, color="green", linewidth=1.0, linestyle="-", label="epsilon = 0.0000001")
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Optimal action [%]')
plt.title('Optimal action for a 10-bandit algorithm \nwith constant action values, alpha = 0.1')
pylab.savefig('optimal_action_c.pdf', bbox_inches='tight', dpi = 100, format = 'pdf')
