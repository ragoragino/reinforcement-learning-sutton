## Chapter 8 - Exercise 8.4

### Setup

The main theme of this chapter are planning methods, i.e. methods that exploit the model of the 
environment in order to increase the learning efficiency. The authors present in this chapter so-called 
Dyna-Q+ algorithm, which is intended to perform well in situations, where the environment might 
change during the learning process. The algorithm keeps track of the time it last visited each 
state/action pair ($\tau (s, a)$) and adds an additional reward, called the exploration bonus, 
derived from this time (with parameter $\kappa$) to the basic reward during the planning back-up. 
This tweak should motivate the agent to not only keep exploring, but to keep 
exploring places in a more systematic fashion, i.e. explore state/action pairs that have not been visited 
for a long time. 

#### Exercise 8.4:
The exploration bonus described above actually changes the estimated values of states and actions. 
Is this necessary? Suppose the bonus $\kappa$ \* $\sqrt{\tau}$ was used not in backups, but solely in action 
selection. That is, suppose the action selected was always that for which $Q(S\_t; a) + 
\kappa \* \sqrt{\tau (S\_t; a)}$ was maximal. Carry out a gridworld experiment that tests and 
illustrates the strengths and weaknesses of this alternate approach.

### Solution
I have implemented the Dyna-Q+ algorithm with backup-adjustments (the original one) and action-selection
adjustments for different $\kappa$ parameters. The environment was a rectangular space (9, 7) 
with a wall in the second row that blocks all cells except the first one from left. 
After 3000 iterations (i.e. episodes), the first cell from the right in this row is also unblocked. 
The algorithms were tested for 6000 iterations.

As we can see from the [Episode_length.pdf](https://github.com/ragoragino/reinforcement-learning-sutton/tree/master/Chapter8/Wall/Episode_length.pdf),
Dyna-Q+ with $\kappa = 0.01$ and backup adjustments performed the best as it found the new route 
quickest and it also kept the lowest overall mean of episode length. For Dyna-Q+ with action 
adjustment, $\kappa = 0.05$ worked the best. "Optimal" $\kappa$ being higher for action selection 
seems reasonable as in this case in order for an action long unvisited to take place and 
update, the maximalization procedure over state actions needs to happen and therefore larger
"additional reward" needs to be given to this state. For longer $\kappa$ parameters we notice 
worsenning of the performance as the algorithm very often visits states that have been unvisited
for a long period of time, thus very often not following the optimal policy. 
