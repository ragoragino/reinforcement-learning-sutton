## Chapter 8 - Exercise 8.4

### Setup

The main theme of this chapter are planning methods, i.e. methods that exploit the model of the 
environment in order to increase the learning efficiency. The authors present in this chapter so-called 
Dyna-Q+ algorithm, which is intended to perform well in situations, where the environment might 
change during the learning process. The algorithm keeps track of the time it visited each state/action
pair and adds an additional reward, called the exploration bonus, derived from this time to the basic
reward. This tweak should motivate the agent to not only keep exploring, but to keep exploring places 
in a more systematic fashion, i.e. state/action pairs that have not been visited for a long time. 

The exploration bonus described above actually
changes the estimated values of states and actions. Is this necessary? Suppose
the bonus \kappa * \sqrt_{\tau} was used not in backups, but solely in action selection. That is,
suppose the action selected was always that for which Q(S_t; a) + \kappa * \sqrt_{\tau (S_t; a)} was
maximal. Carry out a gridworld experiment that tests and illustrates the strengths
and weaknesses of this alternate approach.

### Solution

