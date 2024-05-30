/****************
 * Reinforcement Learning: An Introduction by Richard Sutton and Andrew Barto, First Edition, Problem 2.2
 * Multiarmed Bandit Problem with an epsilon-greedy approach
 * Softmax Action Selector using a Boltzmann Distribution
 * The true rewards are sampled from a Standard Normal Distribution
 * Plotted using ROOT
 * Compilation: root -l main.c
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

void plot(double averageReward_eps1[], double percentOptimalPlays_eps1[], double averageReward_eps2[], double percentOptimalPlays_eps2[], int numPlays, string xAxis, string yAxis){
	auto canvas = new TCanvas("canvas","canvas",600, 400);

	auto graph1 = new TGraph(numPlays+1, averageReward_eps1);
	graph1->SetLineColor(kBlue); graph1->SetTitle("epsilon = 0.1");
	auto graph2 = new TGraph(numPlays+1, averageReward_eps2);
	graph2->SetLineColor(kRed); graph2->SetTitle("epsilon = 0.01");
	auto multigraph = new TMultiGraph();
	multigraph->Add(graph1);
	multigraph->Add(graph2);

	multigraph->SetTitle("10-armed testbed with Softmax Action Selection using the Boltzmann Distribution;Plays;Average Reward");
	multigraph->Draw("apl");

	canvas->BuildLegend();
}

int main(){
	int numActions = 10;
	double epsilon1 = 0.1, epsilon2 = 0.01;
	double temperature = 1;
	int numPlays = 2000;

	vector<double> trueRewards(numActions);
	vector<pair<double, int>> actions(numActions);

	double averageReward_eps1[numPlays+1];
	double averageReward_eps2[numPlays+1];
	double percentOptimalPlays_eps1[numPlays+1];
	double percentOptimalPlays_eps2[numPlays+1];
	int optimalAction = -1;

	// Initialize
	for(int i = 0; i < numActions; i++){
		trueRewards[i] = sampleGaussian(0, 1);
		actions[i] = make_pair(0.0, 0);
	}
	averageReward_eps1[0] = 0.0;
	averageReward_eps2[0] = 0.0;
	percentOptimalPlays_eps1[0] = 0.0;
	percentOptimalPlays_eps2[0] = 0.0;
	optimalAction = distance(trueRewards.begin(), max_element(trueRewards.begin(), trueRewards.end()));

	// Epsilon 1
	for(int i = 1; i <= numPlays; i++){
		// select an action
		int selectedAction = selectAction(actions, epsilon1, temperature, numActions);
		if(selectedAction == -1){
			cout << "Error";
			break;
		}
		// get a reward for that action
		double reward = sampleGaussian(trueRewards[selectedAction], 1);
		// update the expected reward value of that action
		actions[selectedAction].first = (actions[selectedAction].first * actions[selectedAction].second + reward)/(actions[selectedAction].second + 1);
		actions[selectedAction].second = actions[selectedAction].second + 1;
		// update the average reward
		averageReward_eps1[i] = (averageReward_eps1[i-1]*(i-1)+reward)/i;
		// update the optimal play percent
		if(selectedAction == optimalAction) percentOptimalPlays_eps1[i] = (percentOptimalPlays_eps1[i-1]*(i-1) + 1)/i;
		else percentOptimalPlays_eps1[i] = (percentOptimalPlays_eps1[i-1]*(i-1))/i;
	}

	fill(actions.begin(), actions.end(), make_pair(0.0, 0));
	// Epsilon 2
	for(int i = 1; i <= numPlays; i++){
		// select an action
		int selectedAction = selectAction(actions, epsilon2, temperature, numActions);
		if(selectedAction == -1){
			cout << "Error";
			break;
		}
		// get a reward for that action
		double reward = sampleGaussian(trueRewards[selectedAction], 1);
		// update the expected reward value of that action
		actions[selectedAction].first = (actions[selectedAction].first * actions[selectedAction].second + reward)/(actions[selectedAction].second + 1);
		actions[selectedAction].second = actions[selectedAction].second + 1;
		// update the average reward
		averageReward_eps2[i] = (averageReward_eps2[i-1]*(i-1)+reward)/i;
		// update the optimal play percent
		if(selectedAction == optimalAction) percentOptimalPlays_eps2[i] = (percentOptimalPlays_eps2[i-1]*(i-1) + 1)/i;
		else percentOptimalPlays_eps2[i] = (percentOptimalPlays_eps2[i-1]*(i-1))/i;
	}

	plot(averageReward_eps1, percentOptimalPlays_eps1, averageReward_eps2, percentOptimalPlays_eps2, numPlays, "Plays", "Average Reward");
}