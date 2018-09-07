import pandas as pd
import numpy as np
import math
from collections import defaultdict

class RandomSurvivalForest():

	
	def __init__(self, n_trees = 10, max_features = 2, max_depth = 5, min_samples_split = 2, split = "auto"):
		self.n_trees = n_trees
		self.max_depth = max_depth
		self.min_samples_split = min_samples_split
		self.split = split
		self.max_features = max_features

	
	def _logrank(self, x, feature):
		c = x[feature].median()
		if x[x[feature] <= c].shape[0] < self.min_samples_split or x[x[feature] > c].shape[0] <self.min_samples_split:
			return 0
		t = list(set(x["time"]))
		get_time = {t[i]:i for i in range(len(t))}
		N = len(t)
		y = np.zeros((3,N))
		d = np.zeros((3,N))
		feature_inf = x[x[feature] <= c]
		feature_sup = x[x[feature] > c]
		count_sup = np.zeros((N,1))
		count_inf = np.zeros((N,1))
		for _, r in feature_sup.iterrows():
			t_idx = get_time[r["time"]]
			count_sup[t_idx] = count_sup[t_idx] + 1
			if r["event"]:
				d[2][t_idx] = d[2][t_idx] + 1
		for _, r in feature_inf.iterrows():
			t_idx = get_time[r["time"]]
			count_inf[t_idx] = count_inf[t_idx] + 1
			if r["event"]:
				d[1][t_idx] = d[1][t_idx] + 1
		nb_inf = feature_inf.shape[0]
		nb_sup = feature_sup.shape[0]
		for i in range(N):
			y[1][i] = nb_inf
			y[2][i] = nb_sup
			y[0][i] = y[1][i] + y[2][i]
			d[0][i] = d[1][i] + d[2][i]
			nb_inf = nb_inf - count_inf[i]
			nb_sup = nb_sup - count_sup[i]
		num = 0
		den = 0
		for i in range(N):
			if y[0][i] > 0:
				num = num + d[1][i] - y[1][i] * d[0][i] / float(y[0][i])
			if y[0][i] > 1:
				den = den + (y[1][i] / float(y[0][i])) * y[2][i] * ((y[0][i] - d[0][i]) / (y[0][i] - 1)) * d[0][i]
		L = num / math.sqrt(den)
		return abs(L)

	
	def _find_best_feature(self, x):
		split_func = {"auto" : self._logrank}
		features = [f for f in x.columns if f not in ["time", "event"]]
		information_gains = [split_func[self.split](x, feature) for feature in features]
		highest_ig = max(information_gains)
		if highest_ig == 0:
			return None
		else:
			return features[information_gains.index(highest_ig)]

	
	def _compute_leaf(self, x, tree):
		tree["type"] = "leaf"
		# for each distinct death time, count the number of death and
		# individuals at risk at that time.
		# this will be used to calculate the cumulative hazard estimate
		unique_observed_times = np.array(x["time"].unique())
		sorted_unique_times = np.sort(unique_observed_times)
		death_and_risk_at_time = defaultdict(lambda:{"death":0, "risky":0})
		# used to calculate survival probability
		# written by the original author
		count = {}
		for _, r in x.iterrows():
			# count the number of death and individuals at risk at each 
			# observed time
			for observed_time in sorted_unique_times:
				if observed_time < r["time"]:
					death_and_risk_at_time[observed_time]["risky"] += 1.
				elif observed_time == r["time"]:
					if r["event"] == 1:
						death_and_risk_at_time[observed_time]["death"] += 1.
					else:
						death_and_risk_at_time[observed_time]["risky"] += 1.
				else:
					break
			# update count
			count.setdefault((r["time"], 0), 0)
			count.setdefault((r["time"], 1), 0)
			count[(r["time"], r["event"])] = count[(r["time"], r["event"])] + 1

		# used to calculate survival probability 
		# written by the original author
		total = x.shape[0]
		tree["count"] = count
		tree["t"] = sorted_unique_times
		tree["total"] = total

		# calculate the cumulative hazard function
		# on the observed time
		#
		# the cumulative hazard function for the node is defined as:
		# H(t) = sum_{t_l < t} d_l / r_l
		# where {t_l} is the distinct death times in this leaf node,
		# and d_l and r_l equal the number of deaths and individuals
		# at risk at time t_l.
		cumulated_hazard = 0
		cumulative_hazard_function = {}
		for observed_time in sorted_unique_times:
			cumulated_hazard += \
				float(death_and_risk_at_time[observed_time]["death"]) / \
				float(death_and_risk_at_time[observed_time]["risky"])
			cumulative_hazard_function[observed_time] = cumulated_hazard
		tree["cumulative_hazard_function"] = cumulative_hazard_function

	
	def _build(self, x, tree, depth):
		unique_targets = pd.unique(x["time"])

		if len(unique_targets) == 1 or depth == self.max_depth:
			self._compute_leaf(x, tree)
			return
	
		best_feature = self._find_best_feature(x)

		if best_feature == None:
			self._compute_leaf(x, tree)
			return

		feature_median = x[best_feature].median()
		
		tree["type"] = "node"
		tree["feature"] = best_feature
		tree["median"] = feature_median

		left_split_x = x[x[best_feature] <= feature_median]
		right_split_x = x[x[best_feature] > feature_median]
		split_dict = [["left", left_split_x], ["right", right_split_x]]
	
		for name, split_x in split_dict:
			tree[name] = {}
			self._build(split_x, tree[name], depth + 1)

	
	def _compute_survival(self, row, tree):
		count = tree["count"]
		t = tree["t"]
		total = tree["total"]
		h = 1
		survivors = float(total)
		for ti in t:
			if ti <= row[self.time_column]:
				h = h * (1 - count[(ti,1)] / survivors)
			survivors = survivors - count[(ti,1)] - count[(ti,0)]
		return h
	
	
	def _predict_row(self, tree, row, target="survival"):
		if tree["type"] == "leaf":
			if target == "hazard":
				return tree["cumulative_hazard_function"]
			else:
				return self._compute_survival(row, tree)
		else:
			if row[tree["feature"]] > tree["median"]:
				return self._predict_row(tree["right"], row)
			else:
				return self._predict_row(tree["left"], row)


	def _get_ensemble_cumulative_hazard(self, row):
		hazard_functions = [self._predict_row(tree, row, target="hazard")
				   for tree in self.trees]
		supplement_hazard_value = np.zeros(len(hazard_functions))
		# for each observed time, make sure every hazard_function generated
		# by different tree will have a cumulative hazard value.
		for observed_time in np.sort(self.times):
			for index, hazard_function in enumerate(hazard_functions):
				if observed_time not in hazard_function:
					hazard_function[observed_time] = \
						supplement_hazard_value[index]
				else:
					supplement_hazard_value[index] = \
						hazard_function[observed_time]
		# calculate average to generate the ensemble hazard function
		ensemble_cumulative_hazard = {}
		for observed_time in self.times:
			ensemble_cumulative_hazard[observed_time] = \
				np.avg([hazard_function[observed_time] 
						for hazard_function in hazard_functions])
		return ensemble_cumulative_hazard


	def _estimate_median_time(self, hazard_function, avg_flag):
		log_two = np.log(2)
		prev_time = 0
		final_time = np.inf
		for i in range(len(self.times)):
			if hazard_function[self.times[i]] == log_two:
				return self.times[i]
			elif hazard_function[self.times[i]] < log_two:
				prev_time = self.times[i]
			else:
				if avg_flag:
					return (prev_time + self.times[i]) / 2.
				else:
					return self.times[i]
		return (prev_time + final_time) / 2.


	def _print_with_depth(self, string, depth):
		print("{0}{1}".format("    " * depth, string))

	
	def _print_tree(self, tree, depth = 0):
		if tree["type"] == "leaf":
			self._print_with_depth(tree["t"], depth)
			return
		self._print_with_depth("{0} > {1}".format(tree["feature"], tree["median"]), depth)
		self._print_tree(tree["left"], depth + 1)
		self._print_tree(tree["right"], depth + 1)

	
	def fit(self, x, event):
		self.trees = [{} for i in range(self.n_trees)]
		event.columns = ["time", "event"]
		self.times = np.sort(list(event["time"].unique()))
		features = list(x.columns)
		x = pd.concat((x, event), axis=1)
		x = x.sort_values(by="time")
		x.index = range(x.shape[0])
		for i in range(self.n_trees):
			sampled_x = x.sample(frac = 1, replace = True)
			sampled_x.index = range(sampled_x.shape[0])
			sampled_features = list(np.random.permutation(features))[:self.max_features] + ["time","event"]
			self._build(sampled_x[sampled_features], self.trees[i], 0)

	
	def predict_survival_probability(self, x):
		self.time_column = list(x.columns)[-1]
		compute_trees = [x.apply(lambda u: self._predict_row(self.trees[i], u), axis=1) for i in range(self.n_trees)]
		return sum(compute_trees) / self.n_trees

	
	def predict_median_survival_times(self, x_test, average_to_get_median=True):
		result = []
		for _, row in x.iterrows():
			cumulative_hazard = self._get_ensemble_cumulative_hazard(row)
			result.append(self._estimate_median_time(cumulative_hazard, average_to_get_median))
		return result


	def predict_cumulative_hazard(self, x):
		result = [self._get_ensemble_cumulative_hazard(row) 
				  for _, row in x.iterrows()]
		return result
	

	def draw(self):
		for i in range(len(self.trees)):
			print("==========================================\nTree", i)
			self._print_tree(self.trees[i])



