"""Contains all classifier_parameters used (except ANNs)"""
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

import config.classifiers as clfrs

classifier_parameters = {}


def append_classifier_details(classifier_data_balancer, classifier_parameters_dict, classifier_description, classifier_dict):
    """Adds classifier, data_balancer, status and classifier_description to classifier_arr as a dictionary"""
    classifier_dict[classifier_description] = {"data_balancer": classifier_data_balancer, "classifier_parameters": classifier_parameters_dict}


# Generic classifier_parameters
# Clustering-Launched Classification
clustering_launched_classifier_data_balancer = InstanceHardnessThreshold
clustering_launched_classifier_parameters = {"d": 0.4}
append_classifier_details(clustering_launched_classifier_data_balancer, clustering_launched_classifier_parameters, clfrs.clustering_launched_classifier_description,
                          classifier_parameters)

# Extreme learning machines
elm_data_balancer = TomekLinks
elm_parameters = {"layers": [(20, 'sigm'), (20, 'sigm')]}
append_classifier_details(elm_data_balancer, elm_parameters, clfrs.elm_description, classifier_parameters)

# Artificial Neural network
ann_data_balancer = None
ann_parameters = {"activation": "relu", "hidden_layer_sizes": (50, 5), "max_iter": 1000, "solver": "sgd"}
append_classifier_details(ann_data_balancer, ann_parameters, clfrs.ann_description, classifier_parameters)

# Support Vector Machines (with RDF kernel)
svm_rdf_data_balancer = ADASYN
svm_parameters = {"cache_size": 1000, "gamma": 0.0527263157894736, "C": 3.69052631578947, "kernel": "rbf", "class_weight": None, "decision_function_shape": "ovr", "probability": True}
append_classifier_details(svm_rdf_data_balancer, svm_parameters, clfrs.svm_rdf_description, classifier_parameters)

# Support Vector Machines (with linear kernel)
svm_linear_data_balancer = TomekLinks
svm_linear_parameters = {"C": 1.06157894736842, "decision_function_shape": "ovr", "cache_size": 1000, "kernel": "linear", "probability": True, "class_weight": "balanced"}
append_classifier_details(svm_linear_data_balancer, svm_linear_parameters, clfrs.svm_linear_description, classifier_parameters)

# Support Vector Machines (with polynomial kernel)
svm_poly_data_balancer = ADASYN
svm_poly_parameters = {"C": 5, "cache_size": 1000, "coef0": 0, "gamma": "auto", "kernel": "poly", "degree": 3, "class_weight": None, "probability": True}
append_classifier_details(svm_poly_data_balancer, svm_poly_parameters, clfrs.svm_poly_description, classifier_parameters)

# Logistic Regression
logistic_regression_data_balancer = TomekLinks
logistic_regression_parameters = {"C": 8.94842105263157, "class_weight": None, "fit_intercept": True, "solver": "liblinear", "intercept_scaling": 0.01}
append_classifier_details(logistic_regression_data_balancer, logistic_regression_parameters, clfrs.logistic_regression_description, classifier_parameters)

# Decision Tree
decision_tree_data_balancer = NeighbourhoodCleaningRule
decision_tree_parameters = {"criterion": "gini", "max_depth": 8, "max_features": "sqrt", "max_leaf_nodes": None, "min_samples_leaf": 2, "min_samples_split": 2, "splitter": "random", "class_weight": "balanced"}
append_classifier_details(decision_tree_data_balancer, decision_tree_parameters, clfrs.decision_tree_description, classifier_parameters)

# AdaBoost
adaboost_data_balancer = InstanceHardnessThreshold
adaboost_parameters = {"n_estimators": 5, "learning_rate": 0.01, "base_estimator": BernoulliNB(), "algorithm": "SAMME.R"}
append_classifier_details(adaboost_data_balancer, adaboost_parameters, clfrs.adaboost_description, classifier_parameters)

# Random forest
random_forest_data_balancer = None
random_forest_parameters = {"n_estimators": 115, "class_weight": "balanced", "criterion": "gini", "max_depth": 5, "max_features": "sqrt", "oob_score": True}
append_classifier_details(random_forest_data_balancer, random_forest_parameters, clfrs.random_forest_description, classifier_parameters)

# K-nearest neighbours
k_nearest_data_balancer = InstanceHardnessThreshold
k_nearest_parameters = {"weights": "distance", "p": 1, "n_neighbors": 65, "leaf_size": 20, "algorithm": "ball_tree"}
append_classifier_details(k_nearest_data_balancer, k_nearest_parameters, clfrs.k_nearest_description, classifier_parameters)

# Gaussian Naive Bayes
gaussian_naive_bayes_data_balancer = ADASYN
gaussian_naive_bayes_parameters = {}
append_classifier_details(gaussian_naive_bayes_data_balancer, gaussian_naive_bayes_parameters, clfrs.gaussian_naive_bayes_description, classifier_parameters)

# Bernoulli Naive Bayes
bernoulli_naive_bayes_data_balancer = ADASYN
bernoulli_naive_bayes_parameters = {"alpha": 0.2, "binarize": 0, "fit_prior": True}
append_classifier_details(bernoulli_naive_bayes_data_balancer, bernoulli_naive_bayes_parameters, clfrs.bernoulli_naive_bayes_description, classifier_parameters)
