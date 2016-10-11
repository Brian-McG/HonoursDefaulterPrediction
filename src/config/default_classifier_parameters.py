"""Contains all classifier_parameters used (except ANNs)"""
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

import config.classifiers as clfrs

classifier_parameters = {}


def append_classifier_details(classifier_data_balancer, classifier_parameters_dict, classifier_description, classifier_dict):
    """Adds classifier, data_balancer, status and classifier_description to classifier_arr as a dictionary"""
    classifier_dict[classifier_description] = {"data_balancer": classifier_data_balancer, "classifier_parameters": classifier_parameters_dict}


# Generic classifier_parameters
# Clustering-Launched Classification
clustering_launched_classifier_data_balancer = ClusterCentroids
clustering_launched_classifier_parameters = {"d": 0.5}
append_classifier_details(clustering_launched_classifier_data_balancer, clustering_launched_classifier_parameters, clfrs.clustering_launched_classifier_description,
                          classifier_parameters)

# Extreme learning machines
elm_data_balancer = SMOTEENN
elm_parameters = {"layers": [(20, "sigm"), (2, "rbf_l2")]}
append_classifier_details(elm_data_balancer, elm_parameters, clfrs.elm_description, classifier_parameters)

# Artificial Neural network
ann_data_balancer = SMOTEENN
ann_parameters = {"hidden_layer_sizes": (5,), "max_iter": 1000, "solver": "sgd"}
append_classifier_details(ann_data_balancer, ann_parameters, clfrs.ann_description, classifier_parameters)

# Support Vector Machines (with RDF kernel)
svm_rdf_data_balancer = ClusterCentroids
svm_parameters = {"cache_size": 1000, "gamma": "auto", "kernel": "rbf", "class_weight": "balanced", "probability": True, "max_iter": 10000}
append_classifier_details(svm_rdf_data_balancer, svm_parameters, clfrs.svm_rdf_description, classifier_parameters)

# Support Vector Machines (with linear kernel)
svm_linear_data_balancer = ADASYN
svm_linear_parameters = {"C": 7.371053, "decision_function_shape": "ovo", "cache_size": 1000, "gamma": 0.316631579, "kernel": "linear", "probability": True, "max_iter": 100000}
append_classifier_details(svm_linear_data_balancer, svm_linear_parameters, clfrs.svm_linear_description, classifier_parameters)

# Support Vector Machines (with polynomial kernel)
svm_poly_data_balancer = SMOTEENN
svm_poly_parameters = {"cache_size": 1000, "gamma": "auto", "kernel": "poly", "degree": 3, "class_weight": "balanced", "probability": True, "max_iter": 10000}
append_classifier_details(svm_poly_data_balancer, svm_poly_parameters, clfrs.svm_poly_description, classifier_parameters)

# Logistic Regression
logistic_regression_data_balancer = SMOTEENN
logistic_regression_parameters = {"penalty": "l2", "dual": False, "fit_intercept": True, "intercept_scaling": 1, "solver": "newton-cg", "max_iter": 100, "multi_class": "ovr"}
append_classifier_details(logistic_regression_data_balancer, logistic_regression_parameters, clfrs.logistic_regression_description, classifier_parameters)

# Decision Tree
decision_tree_data_balancer = SMOTEENN
decision_tree_parameters = {"max_features": "auto", "class_weight": "balanced"}
append_classifier_details(decision_tree_data_balancer, decision_tree_parameters, clfrs.decision_tree_description, classifier_parameters)

# AdaBoost
adaboost_data_balancer = SMOTEENN
adaboost_parameters = {"n_estimators": 100, "learning_rate": 0.01}
append_classifier_details(adaboost_data_balancer, adaboost_parameters, clfrs.adaboost_description, classifier_parameters)

# Random forest
random_forest_data_balancer = SMOTEENN
random_forest_parameters = {"n_estimators": 10, "n_jobs": -1, "class_weight": "balanced"}
append_classifier_details(random_forest_data_balancer, random_forest_parameters, clfrs.random_forest_description, classifier_parameters)

# K-nearest neighbours
k_nearest_data_balancer = SMOTEENN
k_nearest_parameters = {"n_neighbors": 100}
append_classifier_details(k_nearest_data_balancer, k_nearest_parameters, clfrs.k_nearest_description, classifier_parameters)

# Bernoulli Naive Bayes
bernoulli_naive_bayes_data_balancer = SMOTEENN
bernoulli_naive_bayes_parameters = {"alpha": 0.1, "binarize": None, "class_prior": None, "fit_prior": True}
append_classifier_details(bernoulli_naive_bayes_data_balancer, bernoulli_naive_bayes_parameters, clfrs.bernoulli_naive_bayes_description, classifier_parameters)