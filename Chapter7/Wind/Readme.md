As I have not found some interesting exercises for this chapter in the book, I just wanted to see whether 
the n-step tree-backup Q-Learning is better than classical Q-Learning. The simulations were tested
for the same environment as used in Chapter 6, but with only base moves (i.e. up, down, left and right).
However, I have tried different values of n to see how the performance differs. Due to the fact, that 
the peformance for different values of n gets close together, I have plotted the time to find
the goal state for only 1000 episodes. In this plot, we can clearly see that at the beginning,
the methods of n-step, mainly 2-, 3-, 4- and 5-step, bootstrapping are much better than other 
algorithms. However, after a while, the performance of all algorithms get much closer together.
If we would plot the series for more episodes, we could see that they all find the optimal path
as a path of a deterministic policy from the beginning state.

