/****************
 * Reinforcement Learning: An Introduction by Richard Sutton and Andrew Barto, First Edition, Problem 4.7
 * Policy Iteration on a modification of Jack's Rental Car Problem
 * Plotted using ROOT
 * Compilation: root -l main.cpp
 * Author: Chandradithya J
 ***************/

#include <cmath>
#include <utility>
#include <stdlib.h>
#include <algorithm>
#include <map>
#include <iostream>

// given in the problem (except theta)
int rentalRequests[2] = {3, 4};
int returns[2] = {2, 3};
double rentReward = 10.0, moveCost = 2.0;
int maxCars = 20;
double theta = 1e-6, discount = 0.9;

struct State{
    int loc1, loc2;

    bool operator==(const State &other) const{
        return (loc1 == other.loc1 && loc2 == other.loc2);
    }

    bool operator<(const State &other) const{
        return loc1 < other.loc1 || (loc1 == other.loc1 && loc2 == other.loc2);
    }
};

bool isValidState(State s){
    if (s.loc1 >= 0 && s.loc1 <= maxCars && s.loc2 >= 0 && s.loc2 <= maxCars) return true;
    return false;
}

State getNextState(State oldState, int action){
    State nextState = {oldState.loc1 - action, oldState.loc2 + action}; // overnight action
    // check if action is possible
    if (isValidState(nextState)){
        // perform the daytime actions of renting and returning cars
        int carsRentedFromLoc1 = std::min(nextState.loc1, rentalRequests[0]);
        int carsRentedFromLoc2 = std::min(nextState.loc2, rentalRequests[1]);

        // calculate the new state at the end of the day (after rentals and returns)
        nextState.loc1 = std::max(0, std::min(20, nextState.loc1 - (carsRentedFromLoc1 - returns[0])));
        nextState.loc2 = std::max(0, std::min(20, nextState.loc2 - (carsRentedFromLoc2 - returns[1])));

        return nextState;
    }
    else return oldState;
}

double getReward(State s, int action){
    State nextMorningState = {s.loc1 - action, s.loc2 + action}; // overnight action
    double movingCost = action <= 0 ? abs(moveCost * action) : abs(moveCost * (action - 1)); // one car shuttled for free from loc1 to loc2
    double overnightParkingFee = nextMorningState.loc2 > 10 ? 4.0 : 0.0;

    // if action is possible, calculate the next day's rental reward
    if (isValidState(nextMorningState)){
        // perform the daytime actions of renting and returning cars
        double rentalRewardFromLoc1 = std::min(nextMorningState.loc1, rentalRequests[0]) * rentReward;
        double rentalRewardFromLoc2 = std::min(nextMorningState.loc2, rentalRequests[1]) * rentReward;

        // calculate the total reward
        return rentalRewardFromLoc1 + rentalRewardFromLoc2 - movingCost - overnightParkingFee;
    }
    else return -1.0 * (movingCost + overnightParkingFee);
}

double getStateTransitionProbability(State oldState, int action){
    int numValidActions = 0;
    for (int a = -5; a <= 5; a++){
        State nextState = {oldState.loc1 - a, oldState.loc2 + a};
        if (isValidState(nextState)) numValidActions++;
        else { if (action == a) return 0.0; }
    }
    return 1.0 / numValidActions;
}

void policyIteration(double theta, double discount, std::map<State, int> &policy, std::map<State, double> &value){
    bool policyStable = false;

    while (!policyStable){
        // policy evaluation
        bool converged = false;
        while (!converged){
            std::cout << "Beginning policy evaluation\n";

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

        std::cout << "Beginning policy improvement\n";

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
    std::map<State, int> policy;
    for (int n1 = 0; n1 <= maxCars; n1++){
        for (int n2 = 0; n2 <= maxCars; n2++){
            State s = {n1, n2};
            policy[s] = 0;
        }
    }

    // Initialize a value function to have the same reward (expected reward)
    std::map<State, double> value;
    for (int n1 = 0; n1 <= maxCars; n1++){
        double expectedRentLoc1 = (n1 >= rentalRequests[0]) ? rentalRequests[0] * rentReward : n1 * rentReward;
        for (int n2 = 0; n2 <= maxCars; n2++){
            double expectedRentLoc2 = (n2 >= rentalRequests[1]) ? rentalRequests[1] * rentReward : n2 * rentReward;
            State s = {n1, n2};
            value[s] = expectedRentLoc1 + expectedRentLoc2;
        }
    }

    // perform policy iteration to compute the optimal policy
    policyIteration(theta, discount, policy, value);
}