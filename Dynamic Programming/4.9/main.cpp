/****************
 * Reinforcement Learning: An Introduction by Richard Sutton and Andrew Barto, First Edition, Problem 4.9
 * Value Iteration on the Gambler's Problem
 * Plotted using ROOT
 * Compilation: root -l main.cpp
 * Author: Chandradithya J
 ***************/

#include <cmath>
#include <iostream>
#include <algorithm>
#include <random>
#include <vector>

double theta = 1e-10;

void valueIteration(std::vector<int> policy, std::vector<double>& value, double ph){
    double delta = 0.0;
    do{
        for(int s = 0; s <= 100; s++){
            if(s == 0 || s == 100) continue;
            double v = value[s];
            int bestAction = policy[s]; double bestActionVal = -1.0;
            int maxStake = std::min(s, 100-s);
            for(int a = 0; a <= maxStake; a++){
                double reward = 0.0;
                if(s+a == 100) reward = 1.0; 
                double val = ph * (reward + value[s+a]) + (1-ph) * value[s-a];
                std::cout << s << " " << a << " " << ph * (reward + value[s+a]) << "\n";
                // find the best action
                if (val > bestActionVal){ bestActionVal = val; bestAction = a; }
            }
            // update the value function
            delta = std::max(delta, std::abs(v - value[s]));
        }
    }while(delta >= theta);
}

void printVector(std::vector<double> v){
    for(int i = 0; i < v.size(); i++) std::cout << v[i] << " ";
    std::cout << std::endl;
}

int main(){
    // initialize the policy and value function
    std::vector<int> policy(101, 0);
    std::vector<double> value(101, 0.0);
    // probability of heads
    double ph1 = 0.25, ph2 = 0.55;

    // perform value iteration to compute the optimal policy
    valueIteration(policy, value, ph1);
    // update the policy
    for (int s = 0; s <= 100; s++){
        if (s == 0 || s == 100)
            continue;
        double v = value[s];
        int bestAction = policy[s];
        double bestActionVal = -1.0;
        int maxStake = std::min(s, 100 - s);
        for (int a = 0; a <= maxStake; a++){
            double reward = 0.0;
            if (s + a == 100) reward = 1.0;
            double val = ph1 * (reward + value[s + a]) + (1 - ph1) * value[s - a];
            // find the best action
            if (val > bestActionVal){ bestActionVal = val; bestAction = a; }
        }
        policy[s] = bestAction;
    }

    fill(policy.begin(), policy.end(), 0);
    fill(value.begin(), value.end(), 0.0);

    valueIteration(policy, value, ph2);
    // update the policy
    for (int s = 0; s <= 100; s++){
        if (s == 0 || s == 100)
            continue;
        double v = value[s];
        int bestAction = policy[s];
        double bestActionVal = -1.0;
        int maxStake = std::min(s, 100 - s);
        for (int a = 0; a <= maxStake; a++){
            double reward = 0.0;
            if (s + a == 100)
                reward = 1.0;
            double val = ph2 * (reward + value[s + a]) + (1 - ph2) * value[s - a];
            // find the best action
            if (val > bestActionVal){ bestActionVal = val; bestAction = a; }
        }
        policy[s] = bestAction;
    }
}