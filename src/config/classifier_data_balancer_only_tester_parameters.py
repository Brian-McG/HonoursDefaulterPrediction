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
clc_parameter_dict = {"parameters": {"d": [0.25]}, "requires_random_state": True}
generic_classifier_parameter_dict[clf.clustering_launched_classifier_description] = clc_parameter_dict

# Extreme learning machine
elm_parameter_dict = {"parameters": {
    "layers": [[(20, "sigm")]]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.elm_description] = elm_parameter_dict

# Artificial neural network
ann_parameter_dict = {"parameters": {}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.ann_description] = ann_parameter_dict

# Support Vector Machines (RBF)
rbf_svm_parameter_dict = {"parameters": {}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.svm_rdf_description] = rbf_svm_parameter_dict

# Support Vector Machines (linear)
linear_svm_parameter_dict = {"parameters": {"max_iter": [10000]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.svm_linear_description] = linear_svm_parameter_dict

# Support Vector Machines (poly)
poly_svm_parameter_dict = {"parameters": {"max_iter": [10000]}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.svm_poly_description] = poly_svm_parameter_dict

# Logistic regression
logistic_regression_dict = {"parameters": {}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.logistic_regression_description] = logistic_regression_dict

# Decision tree
decision_tree_dict = {"parameters": {}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.decision_tree_description] = decision_tree_dict

# AdaBoost
adaboost_dict = {"parameters": {}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.adaboost_description] = adaboost_dict

# Random forest
random_forest_dict = {"parameters": {}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.random_forest_description] = random_forest_dict

# K-nearest neighbours
k_nearest_dict = {"parameters": {}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.k_nearest_description] = k_nearest_dict

# Gaussian Naive Bayes
gaussian_dict = {"parameters": {}, "requires_random_state": False}
generic_classifier_parameter_dict[clf.gaussian_naive_bayes_description] = gaussian_dict

# Bernoulli Naive Bayes
bernoulli_dict = {"parameters": {},
                  "requires_random_state": False}
generic_classifier_parameter_dict[clf.bernoulli_naive_bayes_description] = bernoulli_dict
