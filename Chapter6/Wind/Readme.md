## Chapter 6 - Windy Gridworld with King's Moves, Exercise 6.9

### Setup:
Windy Gridworld Shown inset in Figure 6.4 is a standard gridworld, with start and goal states, 
but with one difference: there is a crosswind upward through the middle of the grid. 
The actions are the standard four: up, down, right, and left, but in the middle region 
the resultant next states are shifted upward by a wind, the strength of which varies 
from column to column. The strength of the wind is given below each column, 
in number of cells shifted upward. For example, if you are one cell to the right of the 
goal, then the action left takes you to the cell just above the goal. Let us treat 
this as an undiscounted episodic task, with constant rewards of −1 until the goal state is reached.

Re-solve the windy gridworld task assuming eight possible actions, including the diagonal moves, 
rather than the usual four. How much better can you do with the extra actions? Can you do even
better by including a ninth action that causes no movement at all other than that
caused by the wind?

### Solution:

I have implemented the initial problem and the solution of the exercise for SARSA, 
Q-Learning and Double Q-Learning algorithms. The file implementation of these algorithms
is functions.py, while wind_base.py, wind_diag.py and wind_all.py implement the particular
plotting routines for individual tasks. These tasks consist of an allowance of only four basic 
moves, i.e. left, right, up and down, then allowance of basic and diagonal moves and allowance
of all moves, respectively. The plots resulting from these scripts show firstly the evoluton of
a mean of an episode length, i.e. how many steps on average it took algorithm to find the end in 
the given time frame, for 100,000 iterations. As a second part, the plots of the environment 
with the marks of actions following optimal policy after the last iteration are presented for each
task. 

LEFT, RIGHT, UP, DOWN:
As we can see, all algorithms find a route to the goal, however SARSA is much less efficient, as it stays 
on around 19 steps per episode, with optimal policy taking 17 steps. Q-Learning and Double 
Q-Learning both find the optimal policy in 15 steps at the end, and on average it took them
around 17 steps to finish an episode. 

DIAGONAL:
Yes, the performance of all algorithms gets critically better. 
Here, we can even notice better performance of Double Q-Learning than regular Q-Learning.

ALL:
No, adding the optionality of no-move does not change the performance relative to the results
including only King's moves. The reason might be that there is only a small set of actions
that are possible only with wind moves and are inaccessible in the same number of moves by diagonal,
vertical or horizontal moves.
And again, both Q-Learning and Double Q-Learning perform better than SARSA, however, in this case, 
the difference between the latter two methods is negligible.




 



