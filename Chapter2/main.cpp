#include <chrono>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>


/*
EXERCISE 2.5, pp. 36:
Design and conduct an experiment to demonstrate the difficulties that sample-average 
methods have for nonstationary problems. Use a modified version of the 10-armed testbed
in which all the q∗(a) start out equal and then take independent random walks (say by 
adding a normally distributed increment with mean zero and standard deviation 0.01 to 
all the q∗(a) on each step). Prepare plots like Figure 2.2 for an action-value method 
using sample averages, incrementally computed, and another action-value method using 
a constant step-size parameter, alpha = 0.1. Use epsilon = 0.1 and longer runs, say of 
10,000 steps.
*/


/*
Template function that returns maximum value from an array

@param
    array
@return
    double: maximum value
*/
template <typename T, size_t N>
double max_vector(T(&v)[N]) {
    int v_size = N;
    double max_value = v[0];
    for (int i = 1; i != v_size; i++)
        max_value = std::max(max_value, v[i]);
    return max_value;
}


/*
Template function that returns a vector of indices taking maximum value in an array

@param
    array
@return
    std::vector<int>: with indices referring to the points in the vector taking 
    maximum value
*/
template <typename T, size_t N>
std::vector<int> max_index(T(&v)[N]) {
    std::vector<int> max_index_vector;
    double max_value = max_vector(v);
    int v_size = N;
    for (int i = 0; i != v_size; i++)
    {
        if (v[i] == max_value)
            max_index_vector.push_back(i);
    }
    return max_index_vector;
}


/*
Function that compares strings (i.e. stringVec) of two stringContainer structures

@param
    average_reward: pointer to array containing cumulative average reward for single epsilon run
    optimal action: pointer to array containing cumulative optimal action for single epsilon run
    epsilon: parameter of exploration of the bandit algorithm
    routine_no: current cycle number in a single epsilon run
    steps: parameter specifying the length of the bandit algorithm
    alpha: parameter of update
@return
    updated arrays average_reward and optimal_action
*/
void bandit(double* average_reward, double* optimal_action, const double &, const double &, const int &, const double &);


int main() {
    auto begin = std::chrono::high_resolution_clock::now();
    const double epsilon[] = { 0.1, 0.01, 0.0000001 }; // exploration parameters
    double routine_no = 1.0;
    const double alpha[] = { 1.0, 0.1 };
    const int routine_steps = 1000; // number of routines over which to average
    const int steps = 10000; // number of steps in a single epsilon run
    double *average_reward = new double[steps];
    double *opt_action = new double[steps];

    std::ofstream file1("bandit_classic.csv");
    if (file1.is_open())
    {
        for (const double j : epsilon) {
            for (int k = 0; k != routine_steps; k++)
            {
                bandit(average_reward, opt_action, j, routine_no, steps, alpha[0]);
                routine_no++;
            }

            for (int k = 0; k != steps; k++)
            {
                file1 << average_reward[k] << ",";
            }
            file1 << "\n";
            for (int k = 0; k != steps; k++)
            {
                file1 << opt_action[k] << ",";
            }
            file1 << "\n";
        }
    }
    file1.close();
    delete opt_action;
    delete average_reward;

    routine_no = 1.0;
    average_reward = new double[steps];
    opt_action = new double[steps];
    std::ofstream file2("bandit_classic_alpha.csv");
    if (file2.is_open())
    {
        for (const double j : epsilon) {
            for (int k = 0; k != routine_steps; k++)
            {
                bandit(average_reward, opt_action, j, routine_no, steps, alpha[1]);
                routine_no++;
            }
            for (int k = 0; k != steps; k++)
            {
                file2 << average_reward[k] << ",";
            }
            file2 << "\n";
            for (int k = 0; k != steps; k++)
            {
                file2 << opt_action[k] << ",";
            }
            file2 << "\n";
        }
    }
    file2.close();
    delete opt_action;
    delete average_reward;

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " milliseconds" << std::endl;
    return 0;
}

void bandit(double* average_reward, double* optimal_action, const double &epsilon, const double &routine_no, const int &steps, const double &alpha) {
    const int k = 10; // 10 bandits
    const int inv_epsilon = 1 / epsilon; // probability associated with the exploration paramter
    const double increment_sd = 0.01; // random-walk increment of action values
    const double increment_mean = 0; // random-walk mean of action values
    const int reward_sd = 1; // standard deviation of reward distribution
    std::vector<double> l_average_reward = { 0 }; // variable holding local values of the average reward
    double current_step = 0;
    std::vector<bool> l_optimal_action; // variable holding local values of the optimal action
    double action_estimates[k] = {}; // estimates of the action values
    double action_values[k] = {}; // true action values
    double counter[k] = {}; // counter of the number of selections of individual bandits
    for (int i = 0; i != k; i++)
        counter[i]++;
    std::random_device rd;
    const std::uniform_int_distribution<int> ud1(0, inv_epsilon - 1);
    const std::uniform_int_distribution<int> ud2(0, k - 1);
    const double nd_mean = 0;
    const double nd_sd = 5;
    
    // This loop produces standard k-bandit algorithm without random-walk dynamics of the action values
    // Also the loop after addition of current_step needs to be commented in order for this version to work properly
    // Saved as bandit_classic.csv
    /*for (int j = 0; j != k; j++)
    {
    std::normal_distribution<double> nd(nd_mean,nd_sd);
    action_values[j] = nd(rd);
    }*/
    

    for (int i = 0; i != steps; i++)
    {
        current_step++;
        for (int j = 0; j != k; j++)
        {
            std::normal_distribution<double> nd(increment_mean, increment_sd);
            action_values[j] += nd(rd);
        }
        std::vector<int> current_max_index = max_index(action_values);
        if (ud1(rd) == 0)
        {
            int pick = ud2(rd);
            bool optimal = std::find(current_max_index.begin(), current_max_index.end(), pick) != current_max_index.end();
            l_optimal_action.push_back(optimal);
            std::normal_distribution<double> nd(action_values[pick], reward_sd);
            double reward = nd(rd);
            l_average_reward.push_back(reward);
            action_estimates[pick] += 1.0 / (counter[pick]) * (reward - action_estimates[pick]) *
                (alpha != 1.0 ? counter[pick] * alpha : alpha);
            counter[pick]++;
        }
        else
        {
            std::vector<int> max_index_vec = max_index(action_estimates);
            std::uniform_int_distribution<int> ud3(0, max_index_vec.size() - 1);
            int pick = max_index_vec[ud3(rd)];
            bool optimal = std::find(current_max_index.begin(), current_max_index.end(), pick) != current_max_index.end();
            l_optimal_action.push_back(optimal);
            std::normal_distribution<double> nd(action_values[pick], reward_sd);
            double reward = nd(rd);
            l_average_reward.push_back(reward);
            action_estimates[pick] += (1.0 / (counter[pick]))  * (reward - action_estimates[pick]) *
                (alpha != 1.0 ? counter[pick] * alpha : alpha);
            counter[pick]++;
        }
    }

    for (int i = 0; i != steps; i++)
    {
        if (routine_no < 2)
        {
            average_reward[i] = l_average_reward[i];
            optimal_action[i] = l_optimal_action[i];
        }
        else
        {
            average_reward[i] = l_average_reward[i] * (1.0 / routine_no) + (routine_no - 1.0) / (routine_no) * average_reward[i];
            optimal_action[i] = l_optimal_action[i] * (1.0 / routine_no) + (routine_no - 1.0) / (routine_no) * optimal_action[i];
        }
    }
}
