/****************
 * Reinforcement Learning: An Introduction by Richard Sutton and Andrew Barto, First Edition, Example 4.2
 * Policy Iteration on Jack's Rental Car Problem
 * Plotted using ROOT
 * Compilation: root -l main.c
 * Author: Chandradithya J
 ***************/

#include <random>
#include <cmath>
#include <utility>
#include <stdlib.h>
#include <algorithm>
#include <unordered_map>

// given in the problem (except theta)
int rentalRequests[2] = {3, 4};
int returns[2] = {2, 3};
int carsOutBeforeMove[2] = {rentalRequests[0] - returns[0],
                            rentalRequests[1] - returns[1]};
int rentReward = 10, moveLoss = 2, maxCars = 20;
double theta = 1e-3, discount = 0.9;

struct State
{
    int loc1, loc2;

    bool operator==(const State &other) const
    {
        return (loc1 == other.loc1 && loc2 == other.loc2);
    }
};

State getNextState(State oldState, int action){
    State s = {1, 0};
    return s;
}

double getReward(State s, int action) { return 10; }

double getStateTransitionProbability(State oldState, int action){
    return 0.0;
}

void policyIteration(double theta, double discount, std::unordered_map<State, int>& policy, std::unordered_map<State, int>& value){
    bool policyStable = false;

    while(!policyStable){
        // policy evaluation
        bool converged = false;
        while (!converged){
            converged = true;
            double delta = 0.0;
            for (int n1 = 0; n1 <= maxCars; n1++){
                for (int n2 = 0; n2 <= maxCars; n2++){
                    State s = {n1, n2};
                    double v = value[s];
                    double valueSum = 0.0;
                    int action = policy[s];
                    for (int ns1 = 0; ns1 <= maxCars; ns1++){
                        for (int ns2 = 0; ns2 <= maxCars; ns2++){
                            State ns = {ns1, ns2};
                            double transitionProb = getStateTransitionProbability(s, action);
                            valueSum += transitionProb * (getReward(s, action) + discount * value[ns]);
                        }
                    }
                    value[s] = valueSum; // update the value function

                    delta = std::max(delta, abs(v - value[s]));
                }
            }

            if (delta > theta) converged = false;
        }

        policyStable = true;

        // policy improvement
        for (int n1 = 0; n1 <= maxCars; n1++){
            for (int n2 = 0; n2 <= maxCars; n2++){
                State s = {n1, n2};
                int oldAction = policy[s];
                double bestActionProb = 0.0;
                int bestAction = oldAction;
                for (int a = -5; a <= 5; a++){            
                    State ns = getNextState(s, a);
                    double actionProb = getStateTransitionProbability(s, a) * (getReward(s, a) + discount * value[ns]);

                    if (actionProb > bestActionProb){
                        bestActionProb = actionProb;
                        bestAction = a;
                    }
                }
                policy[s] = bestAction; // update the policy

                if (oldAction != policy[s]) policyStable = false;
            }
        }
    }
}

int main(){
    // Initialize a policy to do nothing in each state (move zero cars)
    std::unordered_map<State, int> policy;
    for(int n1 = 0; n1 <= maxCars; n1++){
        for (int n2 = 0; n2 <= maxCars; n2++){
            State s = {n1, n2};
            policy[s] = 0;
        }
    }

    // Initialize a value function to have the same reward (expected reward)
    std::unordered_map<State, int> value;
    for (int n1 = 0; n1 <= maxCars; n1++){
        int expectedRentLoc1 = (n1 >= rentalRequests[0]) ? rentalRequests[0]*rentReward : n1*rentReward;
        for (int n2 = 0; n2 <= maxCars; n2++){
            int expectedRentLoc2 = (n2 >= rentalRequests[1]) ? rentalRequests[1]*rentReward : n2*rentReward;
            State s = {n1, n2};
            value[s] = expectedRentLoc1 + expectedRentLoc2;
        }
    }
}