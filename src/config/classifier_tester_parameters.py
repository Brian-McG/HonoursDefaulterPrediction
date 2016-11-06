import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

import classifiers as clf

generic_classifier_parameter_dict = {}

# Generic classifiers
# Clustering-launched classification
clc_parameter_dict = {"parameters": {"d": np.linspace(0.01, 1, 20)}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.clustering_launched_classifier_description] = clc_parameter_dict

# Extreme learning machine
elm_parameter_dict = {"parameters": {
    "layers": [[(20, "sigm"), (2, "rbf_l2")], [(20, "lin"), (20, "lin")], [(20, "sigm"), (20, "sigm")], [(50, "sigm")], [(50, "lin")], [(50, "tanh")], [(50, "rbf_l1")],
               [(50, "rbf_l2")], [(50, "rbf_linf")], [(20, "sigm")], [(20, "lin")], [(20, "tanh")], [(20, "rbf_l1")], [(20, "rbf_l2")], [(20, "rbf_linf")], [(75, "sigm")],
               [(75, "lin")], [(75, "tanh")], [(75, "rbf_l1")], [(75, "rbf_l2")], [(75, "rbf_linf")], [(100, "sigm")], [(100, "lin")], [(100, "tanh")], [(100, "rbf_l1")],
               [(100, "rbf_l2")], [(100, "rbf_linf")], [(5, "sigm")], [(5, "lin")], [(5, "tanh")], [(5, "rbf_l1")], [(5, "rbf_l2")], [(5, "rbf_linf")], [(5, "sigm"), (5, "sigm")],
               [(5, "lin"), (5, "lin")], [(5, "tanh"), (5, "tanh")], [(5, "rbf_l1"), (5, "rbf_l1")], [(5, "rbf_l2"), (5, "rbf_l2")], [(5, "rbf_linf"), (5, "rbf_linf")],
               [(25, "sigm"), (25, "sigm")], [(25, "lin"), (25, "lin")], [(25, "tanh"), (25, "tanh")], [(25, "rbf_l1"), (25, "rbf_l1")], [(25, "rbf_l2"), (25, "rbf_l2")],
               [(25, "rbf_linf"), (25, "rbf_linf")], [(50, "sigm"), (50, "sigm")], [(50, "lin"), (50, "lin")], [(50, "tanh"), (50, "tanh")], [(50, "rbf_l1"), (50, "rbf_l1")],
               [(50, "rbf_l2"), (50, "rbf_l2")], [(50, "rbf_linf"), (50, "rbf_linf")], [(75, "sigm"), (75, "sigm")], [(75, "lin"), (75, "lin")], [(75, "tanh"), (75, "tanh")],
               [(75, "rbf_l1"), (75, "rbf_l1")], [(75, "rbf_l2"), (75, "rbf_l2")], [(75, "rbf_linf"), (75, "rbf_linf")], [(100, "sigm"), (100, "sigm")],
               [(100, "lin"), (100, "lin")], [(100, "tanh"), (100, "tanh")], [(100, "rbf_l1"), (100, "rbf_l1")], [(100, "rbf_l2"), (100, "rbf_l2")],
               [(100, "rbf_linf"), (100, "rbf_linf")], [(5, "sigm"), (50, "sigm")], [(5, "lin"), (50, "lin")], [(5, "tanh"), (50, "tanh")], [(5, "rbf_l1"), (50, "rbf_l1")],
               [(5, "rbf_l2"), (50, "rbf_l2")], [(5, "rbf_linf"), (50, "rbf_linf")], [(50, "sigm"), (5, "sigm")], [(50, "lin"), (5, "lin")], [(50, "tanh"), (5, "tanh")],
               [(50, "rbf_l1"), (5, "rbf_l1")], [(50, "rbf_l2"), (5, "rbf_l2")], [(50, "rbf_linf"), (5, "rbf_linf")], [(50, "sigm"), (100, "sigm")], [(50, "lin"), (100, "lin")],
               [(50, "tanh"), (100, "tanh")], [(50, "rbf_l1"), (100, "rbf_l1")], [(50, "rbf_l2"), (100, "rbf_l2")], [(50, "rbf_linf"), (100, "rbf_linf")],
               [(100, "sigm"), (5, "sigm")], [(100, "lin"), (5, "lin")], [(100, "tanh"), (5, "tanh")], [(100, "rbf_l1"), (5, "rbf_l1")], [(100, "rbf_l2"), (5, "rbf_l2")],
               [(100, "rbf_linf"), (5, "rbf_linf")]]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.elm_description] = elm_parameter_dict

# Artificial neural network
ann_parameter_dict = {"parameters": {"hidden_layer_sizes": [(5,), (50, 50), (100,), (5, 5, 5), (50, 5), (4, 4)], "activation": ["identity", "logistic", "tanh", "relu"],
                                     "solver": ["lbgfs", "sgd", "adam"], "max_iter": [1000]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.ann_description] = ann_parameter_dict

# Support Vector Machines (RBF)
rbf_svm_parameter_dict = {"parameters": {"cache_size": [2048], "decision_function_shape": ["ovr"], "kernel": ["rbf"], "C": np.linspace(0.01, 10, 20).tolist() + [1],
                                         "gamma": ["auto"] + np.linspace(0.0001, 1, 20).tolist(), 'class_weight': [None, 'balanced'], "max_iter": [10000]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.svm_rdf_description] = rbf_svm_parameter_dict

# Support Vector Machines (linear)
linear_svm_parameter_dict = {"parameters": {"cache_size": [2048], "decision_function_shape": ["ovr"], "kernel": ["linear"], "C": np.linspace(0.01, 10, 20).tolist() + [1],
                                            'class_weight': [None, 'balanced'], "max_iter": [10000]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.svm_linear_description] = linear_svm_parameter_dict

# Support Vector Machines (poly)
poly_svm_parameter_dict = {"parameters": {"cache_size": [2048], "decision_function_shape": ["ovr"], "kernel": ["poly"], "C": np.linspace(0.01, 2, 5).tolist() + [1],
                                          "gamma": ["auto"] + np.linspace(0.01, 1, 5).tolist(), 'class_weight': [None, 'balanced'], 'degree': [2, 3, 4], "max_iter": [10000]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.svm_poly_description] = poly_svm_parameter_dict

# Logistic regression
logistic_regression_dict = {"parameters": {"C": [1] + np.linspace(0.01, 3, 5).tolist(), "fit_intercept": [True, False], "intercept_scaling": [1] + np.linspace(0.01, 10, 10).tolist(),
                                           "class_weight": [None, "balanced"], "solver": ["liblinear", "sag"]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.logistic_regression_description] = logistic_regression_dict

# Decision tree
decision_tree_dict = {"parameters": {"criterion": ["gini", "entropy"], "splitter": ["best", "random"], "max_features": ["auto"], "class_weight": [None, "balanced"],
                                     "max_depth": [None] + [3, 5], "min_samples_split": [2,3], "min_samples_leaf": [2,3],
                                     "max_leaf_nodes": [None] + [3, 6]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.decision_tree_description] = decision_tree_dict

# AdaBoost
adaboost_dict = {"parameters": {
    "base_estimator": [BernoulliNB(), DecisionTreeClassifier(), ExtraTreeClassifier(), Perceptron(), SGDClassifier()], "n_estimators": [5, 20, 50, 75, 100],
    "learning_rate": np.linspace(0.01, 1, 3).tolist(), "algorithm": ["SAMME", "SAMME.R"]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.adaboost_description] = adaboost_dict

# Random forest
random_forest_dict = {"parameters": {"n_estimators": list(range(5, 120, 10)), "criterion": ["gini", "entropy"], "max_features": ["auto", "sqrt", "log2", None],
                                     "max_depth": [None] + list(range(5, 120, 20)), "class_weight": [None, "balanced"]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.random_forest_description] = random_forest_dict

# K-nearest neighbours
k_nearest_dict = {"parameters": {"n_neighbors": list(range(5, 120, 15)), "weights": ["uniform", "distance"], "algorithm": ["auto"],
                                 "leaf_size": list(range(20, 50, 10)), "p": list(range(1, 4))}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.k_nearest_description] = k_nearest_dict

# Gaussian Naive Bayes
gaussian_dict = {"parameters": {}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.gaussian_naive_bayes_description] = gaussian_dict

# Bernoulli Naive Bayes
bernoulli_dict = {"parameters": {"alpha": np.linspace(0.1, 1, 10).tolist(), "fit_prior": [True, False], "binarize": [None] + np.linspace(0, 10, 10).tolist()},
                  "requires_random_state": False}
generic_classifier_parameter_dict[clf.bernoulli_naive_bayes_description] = bernoulli_dict
