import numpy as np
from multiprocessing import Manager
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

import classifiers as clf

generic_classifier_parameter_dict = {}

# Generic classifiers
# Clustering-launched classification
clc_parameter_dict = {"parameters": {"d": np.linspace(0.01, 1, 20)}, "requires_random_state": True}
generic_classifier_parameter_dict[clf.clustering_launched_classifier_description] = clc_parameter_dict

# Extreme learning machine
elm_parameter_dict = None
generic_classifier_parameter_dict[clf.elm_description] = elm_parameter_dict

# Artificial neural network
ann_parameter_dict = None
generic_classifier_parameter_dict[clf.ann_description] = ann_parameter_dict

# Support Vector Machines (RBF)
rbf_svm_parameter_dict = {"parameters": {"cache_size": [2048], "decision_function_shape": ["ovr"], "kernel": ["rbf"], "C": np.linspace(0.01, 10, 20).tolist() + [1],
                          "gamma": ["auto"] + np.linspace(0.0001, 1, 20).tolist(), 'class_weight': [None, 'balanced']}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.svm_rdf_description] = rbf_svm_parameter_dict

# Support Vector Machines (linear)
linear_svm_parameter_dict = {"parameters": {"cache_size": [2048], "decision_function_shape": ["ovr"], "kernel": ["linear"], "C": np.linspace(0.01, 10, 20).tolist() + [1],
                             'class_weight': [None, 'balanced']}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.svm_linear_description] = linear_svm_parameter_dict

# Support Vector Machines (poly)
poly_svm_parameter_dict = {"parameters": {"cache_size": [2048], "decision_function_shape": ["ovr"], "kernel": ["poly"], "C": np.linspace(0.01, 5, 10).tolist() + [1],
                           "gamma": ["auto"] + np.linspace(0.0001, 1, 10).tolist(), 'class_weight': [None, 'balanced'], 'degree': [2, 3, 4, 5],
                           'coef0': [0, 5, 10]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.svm_poly_description] = poly_svm_parameter_dict

# Logistic regression
logistic_regression_dict = {"parameters": {"C": np.linspace(0.01, 10, 20).tolist(), "fit_intercept": [True, False], "intercept_scaling": np.linspace(0.01, 10, 20).tolist(),
                            "class_weight": [None, "balanced"], "solver": ["newton-cg", "lbfgs", "liblinear", "sag"]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.logistic_regression_description] = logistic_regression_dict

# Decision tree
decision_tree_dict = {"parameters": {"criterion": ["gini", "entropy"], "splitter": ["best", "random"], "max_features": ["auto", "sqrt", "log2"], "class_weight": [None, "balanced"],
                      "max_depth": ([None] + list(range(3, 9))), "min_samples_split": list(range(1, 4)), "min_samples_leaf": list(range(1, 3)),
                      "max_leaf_nodes": [None] + list(range(5, 10))}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.decision_tree_description] = decision_tree_dict

# AdaBoost
adaboost_dict = {"parameters": {
    "base_estimator": [BernoulliNB(), DecisionTreeClassifier(), ExtraTreeClassifier(), Perceptron(), SGDClassifier()], "n_estimators": [5, 20, 50, 75, 100],
    "learning_rate": np.linspace(0.01, 1, 3).tolist(), "algorithm": ["SAMME", "SAMME.R"]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.adaboost_description] = adaboost_dict

# Random forest
random_forest_dict = {"parameters": {"n_estimators": list(range(5, 120, 10)), "criterion": ["gini", "entropy"], "max_features": ["auto", "sqrt", "log2", None],
                      "max_depth": [None] + list(range(5, 120, 20)), "oob_score": [True, False], "class_weight": [None, "balanced"]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.random_forest_description] = random_forest_dict

# K-nearest neighbours
k_nearest_dict = {"parameters": {"n_neighbors": list(range(5, 120, 10)), "weights": ["uniform", "distance"], "algorithm": ["ball_tree", "kd_tree", "brute", "auto"],
                  "leaf_size": list(range(20, 50, 5)), "p": list(range(1, 5))}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.k_nearest_description] = k_nearest_dict

# Bernoulli Naive Bayes
bernoulli_dict = {"parameters": {"alpha": np.linspace(0, 1, 10).tolist(), "fit_prior": [True, False], "binarize": [None] + np.linspace(0, 10, 10).tolist()}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.bernoulli_naive_bayes_description] = bernoulli_dict

# Voting classifier
generic_classifier_parameter_dict[clf.voting_classifier_description] = None
