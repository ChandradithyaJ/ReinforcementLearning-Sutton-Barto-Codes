/****************
 * Reinforcement Learning: An Introduction by Richard Sutton and Andrew Barto, Second Edition, Problem 2.5
 * Comparision of the Sample-Average, Incremental Implementation and an Action-Value method with a constant 
 * step-size parameter of 0.1 on a Nonstationary Multiarmed Testbed
 * Plotted using ROOT
 * Compilation: root -l main.cpp
 * Author: Chandradithya J
***************/

#include <random>
#include <cmath>
#include <utility>
#include <stdlib.h>
#include <algorithm>

using namespace std;

double sampleGaussian(double mean, double variance) {
  // Use Mersenne Twister engine for random number generation
  random_device rd;
  mt19937 gen(rd());

  normal_distribution<double> distribution;
  double standard_normal_sample = distribution(gen);
  return standard_normal_sample * sqrt(variance) + mean;
}

int sampleBoltzmann(vector<pair<double, int>> actions, double temperature, int numActions){
	if(temperature <= 0 || numActions <= 0){
		return -1.0;
	}

	vector<double> weights(numActions);
	double sumWeights = 0.0;
	for(int i = 0; i < numActions; i++){
		weights[i] = exp(-actions[i].first/temperature);
		sumWeights += weights[i];
	}

	// Normalize the weights
	for(int i = 0; i < numActions; i++) weights[i] /= sumWeights;

	// sample an index
	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<int> distribution(weights.begin(), weights.end());
	return distribution(gen);
}


int selectAction(vector<pair<double, int>> actions, double epsilon, double temperature, int numActions){
	if(epsilon < 0 || epsilon > 1.0) return -1;

	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<double> distribution(0.0, 1.0);

	double random_number = distribution(gen);
	if (random_number < epsilon) { // epsilon-greedy part
		return sampleBoltzmann(actions, temperature, numActions);
	} else { // greedy part
		double bestActionReward = -10000.0;
		int bestActionIndex = -1;

		for(int i = 0; i < numActions; i++){
			if(actions[i].first > bestActionReward){
				bestActionReward = actions[i].first;
				bestActionIndex = i;
			}
		}
		return bestActionIndex;
	}
}

void plot(double averageReward_sampleAverage[], double percentOptimalPlays_sampleAverage[], double averageReward_inc[], double percentOptimalPlays_inc[], double averageReward_constStep[], double percentOptimalPlays_constStep[], int numPlays, string xAxis, string yAxis){
	auto canvas = new TCanvas("canvas","canvas",600, 400);

	auto graph1 = new TGraph(numPlays+1, averageReward_sampleAverage);
	graph1->SetLineColor(kBlue); graph1->SetTitle("Sample-Average");
	auto graph2 = new TGraph(numPlays+1, averageReward_inc);
	graph2->SetLineColor(kRed); graph2->SetTitle("Incremental Implementation");
	auto graph3 = new TGraph(numPlays+1, averageReward_constStep);
	graph3->SetLineColor(kOrange); graph3->SetTitle("Constant Step Size");
	auto multigraph = new TMultiGraph();
	multigraph->Add(graph1);
	multigraph->Add(graph2);
	multigraph->Add(graph3);

	multigraph->SetTitle("Nonstationary Multiarmed Bandit Comparision;Plays;Average Reward");
	multigraph->Draw("apl");

	canvas->BuildLegend();
}

int main(){
	int numActions = 10;
	double epsilon = 0.1;
	double alpha = 0.1;
	double temperature = 1;
	int numPlays = 10000;

	vector<double> trueRewards(numActions, 1.0); // all start out equal
	vector<pair<double, int>> actions(numActions, make_pair(0.0, 0));

	double averageReward_sampleAverage[numPlays+1];
	double averageReward_inc[numPlays+1];
	double averageReward_constStep[numPlays+1];
	double percentOptimalPlays_sampleAverage[numPlays+1];
	double percentOptimalPlays_inc[numPlays+1];
	double percentOptimalPlays_constStep[numPlays+1];
	int optimalAction = -1;

	// Initialize
	averageReward_sampleAverage[0] = 0.0;
	averageReward_inc[0] = 0.0;
	averageReward_constStep[0] = 0.0;
	percentOptimalPlays_sampleAverage[0] = 0.0;
	percentOptimalPlays_inc[0] = 0.0;
	percentOptimalPlays_constStep[0] = 0.0;
	optimalAction = distance(trueRewards.begin(), max_element(trueRewards.begin(), trueRewards.end()));

	// Sample Average
	for(int i = 1; i <= numPlays; i++){
		// select an action
		int selectedAction = selectAction(actions, epsilon, temperature, numActions);
		if(selectedAction == -1){
			cout << "Error";
			break;
		}
		// get a reward for that action
		double reward = sampleGaussian(trueRewards[selectedAction], 1);
		// update the true rewards (nonstationary) and recompute the optimal action
		for(int i = 0; i < numActions; i++) trueRewards[i] += sampleGaussian(0, 0.01);
		optimalAction = distance(trueRewards.begin(), max_element(trueRewards.begin(), trueRewards.end()));
		// update the expected reward value of that action
		actions[selectedAction].first = (actions[selectedAction].first * actions[selectedAction].second + reward)/(actions[selectedAction].second + 1);
		actions[selectedAction].second = actions[selectedAction].second + 1;
		// update the average reward
		averageReward_sampleAverage[i] = (averageReward_sampleAverage[i-1]*(i-1)+reward)/i;
		// update the optimal play percent
		if(selectedAction == optimalAction) percentOptimalPlays_sampleAverage[i] = (percentOptimalPlays_sampleAverage[i-1]*(i-1) + 1)/i;
		else percentOptimalPlays_sampleAverage[i] = (percentOptimalPlays_sampleAverage[i-1]*(i-1))/i;
	}

	// reset common variables
	fill(trueRewards.begin(), trueRewards.end(), 1.0);
	fill(actions.begin(), actions.end(), make_pair(0.0, 0));
	optimalAction = distance(trueRewards.begin(), max_element(trueRewards.begin(), trueRewards.end()));

	// Incremental Implementation
	for(int i = 1; i <= numPlays; i++){
		// select an action
		int selectedAction = selectAction(actions, epsilon, temperature, numActions);
		if(selectedAction == -1){
			cout << "Error";
			break;
		}
		// get a reward for that action
		double reward = sampleGaussian(trueRewards[selectedAction], 1);
		// update the true rewards (nonstationary) and recompute the optimal action
		for(int i = 0; i < numActions; i++) trueRewards[i] += sampleGaussian(0, 0.01);
		optimalAction = distance(trueRewards.begin(), max_element(trueRewards.begin(), trueRewards.end()));
		// update the expected reward value of that action
		actions[selectedAction].first = actions[selectedAction].first + (1/(actions[selectedAction].second+1))*(reward - actions[selectedAction].first);
		actions[selectedAction].second = actions[selectedAction].second + 1;
		// update the average reward
		averageReward_inc[i] = (averageReward_inc[i-1]*(i-1)+reward)/i;
		// update the optimal play percent
		if(selectedAction == optimalAction) percentOptimalPlays_inc[i] = (percentOptimalPlays_inc[i-1]*(i-1) + 1)/i;
		else percentOptimalPlays_inc[i] = (percentOptimalPlays_inc[i-1]*(i-1))/i;
	}

	// reset common variables
	fill(trueRewards.begin(), trueRewards.end(), 1.0);
	fill(actions.begin(), actions.end(), make_pair(0.0, 0));
	optimalAction = distance(trueRewards.begin(), max_element(trueRewards.begin(), trueRewards.end()));

	// Action-Value Method with a Constant Step-Size
	for(int i = 1; i <= numPlays; i++){
		// select an action
		int selectedAction = selectAction(actions, epsilon, temperature, numActions);
		if(selectedAction == -1){
			cout << "Error";
			break;
		}
		// get a reward for that action
		double reward = sampleGaussian(trueRewards[selectedAction], 1);
		// update the true rewards (nonstationary) and recompute the optimal action
		for(int i = 0; i < numActions; i++) trueRewards[i] += sampleGaussian(0, 0.01);
		optimalAction = distance(trueRewards.begin(), max_element(trueRewards.begin(), trueRewards.end()));
		// update the expected reward value of that action
		actions[selectedAction].first = actions[selectedAction].first + alpha*(reward - actions[selectedAction].first);
		actions[selectedAction].second = actions[selectedAction].second + 1;
		// update the average reward
		averageReward_constStep[i] = (averageReward_constStep[i-1]*(i-1)+reward)/i;
		// update the optimal play percent
		if(selectedAction == optimalAction) percentOptimalPlays_constStep[i] = (percentOptimalPlays_constStep[i-1]*(i-1) + 1)/i;
		else percentOptimalPlays_constStep[i] = (percentOptimalPlays_constStep[i-1]*(i-1))/i;
	}

	plot(averageReward_sampleAverage, percentOptimalPlays_sampleAverage, averageReward_inc, percentOptimalPlays_inc, averageReward_constStep, percentOptimalPlays_constStep, numPlays, "Plays", "Average Reward");
}