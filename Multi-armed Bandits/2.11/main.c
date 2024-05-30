/****************
 * Reinforcement Learning: An Introduction by Richard Sutton and Andrew Barto, Second Edition, Problem 2.11
 * Comparision of the Upper-Confidence-Bound, Gradient Bandit, epsilon-greedy method with a constant 
 * step-size parameter of 0.1, epsilon-greedy with sample average, and epsilon-greedy with optimistic
 * initialization on a Nonstationary Multiarmed Testbed
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

void plot(double averageReward_ucb[], double averageReward_gba[], double averageReward_cs[], double averageReward_sa[], double averageReward_oi[], vector<double> hyperparameters){
	gErrorIgnoreLevel = kError;

	double hyperparametersArray[hyperparameters.size()];
	for(int i = 0; i < hyperparameters.size(); i++) hyperparametersArray[i] = hyperparameters[i];

	int numHyperparameters = hyperparameters.size();

	auto canvas = new TCanvas("canvas","canvas", 600, 400);

	auto graph1 = new TGraph(numHyperparameters, hyperparametersArray, averageReward_ucb);
	graph1->SetLineColor(kBlue); graph1->SetTitle("Upper-Confidence-Bound (c)");
	auto graph2 = new TGraph(numHyperparameters, hyperparametersArray, averageReward_gba);
	graph2->SetLineColor(kRed); graph2->SetTitle("Gradient Bandit Algorithm (alpha)");
	auto graph3 = new TGraph(numHyperparameters, hyperparametersArray, averageReward_cs);
	graph3->SetLineColor(kOrange); graph3->SetTitle("Constant Step Size (epsilon)");
	auto graph4 = new TGraph(numHyperparameters, hyperparametersArray, averageReward_sa);
	graph4->SetLineColor(kGreen); graph4->SetTitle("Sample-Average (epsilon)");
	auto graph5 = new TGraph(numHyperparameters, hyperparametersArray, averageReward_oi);
	graph5->SetLineColor(kYellow); graph5->SetTitle("Optimistic Initialization (initial value)");
	auto multigraph = new TMultiGraph();
	multigraph->Add(graph1);
	multigraph->Add(graph2);
	multigraph->Add(graph3);
	multigraph->Add(graph4);
	multigraph->Add(graph5);

	multigraph->SetTitle("Nonstationary Multiarmed Bandit Comparision;Hyperparameters;Average Reward for the last 100,000 steps");
	multigraph->Draw("apl");

	canvas->BuildLegend();
}

int selectActionUCB(vector<pair<double, int>> actions, int numActions, double confidence, int t){
	int bestActionIndex = 0;
	double maxPotential = actions[0].first;
	for(int a = 0; a < numActions; a++){
		if (actions[a].second == 0){
			bestActionIndex = a;
			break;
		} else{
			double potential = actions[a].first + confidence*sqrt(log(t)/actions[a].second);
			if(potential > maxPotential){
				maxPotential = potential;
				bestActionIndex = a;
			}
		}
	}
	return bestActionIndex;
}

int selectActionGBA(int numActions, vector<double> pref, vector<double>& prob){
	// calculate probabilities for actions
	double sumWeights = 0.0;
	for(int a = 0; a < numActions; a++){
		prob[a] = exp(pref[a]);
		sumWeights += prob[a];
	}
	for(int a = 0; a < numActions; a++) prob[a] /= sumWeights;

	// sample an index
	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<int> distribution(prob.begin(), prob.end());
	return distribution(gen);
}

int selectActionEps(vector<pair<double, int>> actions, double epsilon, double temperature, int numActions){
	if(epsilon < 0) return -1;

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

void updateRewards(vector<double>& trueRewards, vector<pair<double, int>>& actions, int selectedAction, double reward){
	// update the true rewards (nonstationary)
	for(int i = 0; i < actions.size(); i++) trueRewards[i] += sampleGaussian(0, 0.01);
	// update the expected reward value of that action
	actions[selectedAction].first = (actions[selectedAction].first * actions[selectedAction].second + reward)/(actions[selectedAction].second + 1);
	actions[selectedAction].second = actions[selectedAction].second + 1;
}

void reset(vector<double>& trueRewards, vector<pair<double, int>>& actions){
	fill(trueRewards.begin(), trueRewards.end(), 1.0);
	fill(actions.begin(), actions.end(), make_pair(0.0, 0));
}

int main(){
	int numActions = 10;
	const vector<double> epsilon{1.0/128, 1.0/64, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2, 1.0, 2.0, 4.0};
	const vector<double> alpha{1.0/128, 1.0/64, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2, 1.0, 2.0, 4.0};
	const vector<double> Q0{1.0/128, 1.0/64, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2, 1.0, 2.0, 4.0};
	const vector<double> confidence{1.0/128, 1.0/64, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2, 1.0, 2.0, 4.0};
	const double temperature = 1;
	const double stepSize = 0.1;
	int numPlays = 200000;

	vector<double> trueRewards(numActions, 1.0); // all start out equal
	vector<pair<double, int>> actions(numActions, make_pair(0.0, 0));

	double averageReward_ucb[confidence.size()];
	memset(averageReward_ucb, 0.0, sizeof(averageReward_ucb));
	double averageReward_gba[alpha.size()];
	memset(averageReward_gba, 0.0, sizeof(averageReward_gba));
	double averageReward_cs[epsilon.size()];
	memset(averageReward_cs, 0.0, sizeof(averageReward_cs));
	double averageReward_sa[epsilon.size()];
	memset(averageReward_sa, 0.0, sizeof(averageReward_sa));
	double averageReward_oi[Q0.size()];
	memset(averageReward_oi, 0.0, sizeof(averageReward_oi));

	// Upper-Confidence-Bound Algorithm
	int idx = 0;
	for(float c: confidence){
		// reset common variables
		reset(trueRewards, actions);
		for(int i = 1; i <= numPlays; i++){
			// select an action
			int selectedAction = selectActionUCB(actions, numActions, c, i);
			// get a reward for that action and update the true and expected rewards
			double reward = sampleGaussian(trueRewards[selectedAction], 1);
			updateRewards(trueRewards, actions, selectedAction, reward);
			// update the average reward (calculate only for the second half)
			if(i >= numPlays/2 + 1) averageReward_ucb[idx] = (averageReward_ucb[idx] * (i - numPlays/2 - 1) + reward) / (i - numPlays/2);
		}
		idx++;
	}

	// Gradient Bandit Algorithm
	idx = 0;
	for(float alp: alpha){
		vector<double> preference(numActions, 0.0);
		vector<double> probability(numActions, 1/numActions);
		double avgReward = 0.0;
		// reset common variables
		reset(trueRewards, actions);
		for(int i = 1; i <= numPlays; i++){
			// select an action
			int selectedAction = selectActionGBA(numActions, preference, probability);
			// get a reward for that action and update the true and expected rewards
			double reward = sampleGaussian(trueRewards[selectedAction], 1);
			updateRewards(trueRewards, actions, selectedAction, reward);
			// update the numerical preferences
			for(int a = 0; a < numActions; a++){
				if(a == selectedAction) preference[a] += alp * (reward - avgReward) * (1 - probability[a]);
				else preference[a] -= alp * (reward - avgReward) * probability[a];
			}
			// update the average reward received
			avgReward = (avgReward * (i-1) + reward) / i;
			// update the average reward (calculate only for the second half)
			if(i >= numPlays/2 + 1) averageReward_gba[idx] = (averageReward_gba[idx] * (i - numPlays/2 - 1) + reward) / (i - numPlays/2);
		}
		idx++;
	}

	// Epsilon-Greedy Method with a constant step size parameter of 0.1
	idx = 0;
	for(double e: epsilon){
		// reset common variables
		reset(trueRewards, actions);
		for(int i = 1; i <= numPlays; i++){
			// select an action
			int selectedAction = selectActionEps(actions, e, temperature, numActions);
			// get a reward for that action and update the true and expected rewards
			double reward = sampleGaussian(trueRewards[selectedAction], 1);
			// update the true rewards (nonstationary)
			for(int i = 0; i < numActions; i++) trueRewards[i] += sampleGaussian(0, 0.01);
			// update the expected reward value of that action
			actions[selectedAction].first = actions[selectedAction].first + stepSize*(reward - actions[selectedAction].first);
			actions[selectedAction].second = actions[selectedAction].second + 1;
			// update the average reward (calculate only for the second half)
			if(i >= numPlays/2 + 1) averageReward_cs[idx] = (averageReward_cs[idx] * (i - numPlays/2 - 1) + reward) / (i - numPlays/2);
		}
		idx++;
	}

	// Epsilon-Greedy Method with Sample-Average
	idx = 0;
	for(double e: epsilon){
		// reset common variables
		reset(trueRewards, actions);
		for(int i = 1; i <= numPlays; i++){
			// select an action
			int selectedAction = selectActionEps(actions, e, temperature, numActions);
			// get a reward for that action and update the true and expected rewards
			double reward = sampleGaussian(trueRewards[selectedAction], 1);
			updateRewards(trueRewards, actions, selectedAction, reward);
			// update the average reward (calculate only for the second half)
			if(i >= numPlays/2 + 1) averageReward_sa[idx] = (averageReward_sa[idx] * (i - numPlays/2 - 1) + reward) / (i - numPlays/2);
		}
		idx++;
	}

	// Epsilon-Greedy Method with Optimistic Initialization
	idx = 0;
	for(double o: Q0){
		// reset common variables
		reset(trueRewards, actions);
		fill(actions.begin(), actions.end(), make_pair(o, 0));
		for(int i = 1; i <= numPlays; i++){
			// select an action
			int selectedAction = selectActionEps(actions, 0, temperature, numActions);
			// get a reward for that action and update the true and expected rewards
			double reward = sampleGaussian(trueRewards[selectedAction], 1);
			// update the true rewards (nonstationary)
			for(int i = 0; i < numActions; i++) trueRewards[i] += sampleGaussian(0, 0.01);
			// update the expected reward value of that action
			actions[selectedAction].first = actions[selectedAction].first + stepSize*(reward - actions[selectedAction].first);
			actions[selectedAction].second = actions[selectedAction].second + 1;
			// update the average reward (calculate only for the second half)
			if(i >= numPlays/2 + 1) averageReward_oi[idx] = (averageReward_oi[idx] * (i - numPlays/2 - 1) + reward) / (i - numPlays/2);
		}
		idx++;
	}

	plot(averageReward_ucb, averageReward_gba, averageReward_cs, averageReward_sa, averageReward_oi, epsilon);
}