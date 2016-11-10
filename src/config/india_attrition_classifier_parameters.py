"""Contains all classifier_parameters used (except ANNs)"""
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import ExtraTreeClassifier

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
elm_data_balancer = SMOTETomek
elm_parameters = {"layers": [(100, 'rbf_l2'), (100, 'rbf_l2')]}
append_classifier_details(elm_data_balancer, elm_parameters, clfrs.elm_description, classifier_parameters)

# Artificial Neural network
ann_data_balancer = ADASYN
ann_parameters = {"activation": "relu", "hidden_layer_sizes": (100,), "max_iter": 1000, "solver": "adam"}
append_classifier_details(ann_data_balancer, ann_parameters, clfrs.ann_description, classifier_parameters)

# Support Vector Machines (with RDF kernel)
svm_rdf_data_balancer = ADASYN
svm_parameters = {"cache_size": 1000, "gamma": 0.0527263157894736, "C": 8.94842105263157, "kernel": "rbf", "class_weight": "balanced", "decision_function_shape": "ovr", "probability": True}
append_classifier_details(svm_rdf_data_balancer, svm_parameters, clfrs.svm_rdf_description, classifier_parameters)

# Support Vector Machines (with linear kernel)
svm_linear_data_balancer = SMOTE
svm_linear_parameters = {"C": 2.11315789473684, "decision_function_shape": "ovr", "cache_size": 1000, "kernel": "linear", "class_weight": None}
append_classifier_details(svm_linear_data_balancer, svm_linear_parameters, clfrs.svm_linear_description, classifier_parameters)

# Support Vector Machines (with polynomial kernel)
svm_poly_data_balancer = RandomUnderSampler
svm_poly_parameters = {"kernel": "poly", "gamma": 0.1, "C": 8.9}
append_classifier_details(svm_poly_data_balancer, svm_poly_parameters, clfrs.svm_poly_description, classifier_parameters)

# Logistic Regression
logistic_regression_data_balancer = ADASYN
logistic_regression_parameters = {"class_weight": "balanced"}
append_classifier_details(logistic_regression_data_balancer, logistic_regression_parameters, clfrs.logistic_regression_description, classifier_parameters)

# Decision Tree
decision_tree_data_balancer = ADASYN
decision_tree_parameters = {"class_weight": "balanced"}
append_classifier_details(decision_tree_data_balancer, decision_tree_parameters, clfrs.decision_tree_description, classifier_parameters)

# AdaBoost
adaboost_data_balancer = ADASYN
adaboost_parameters = {"n_estimators": 100, "learning_rate": 0.505, "base_estimator": ExtraTreeClassifier(), "algorithm": "SAMME"}
append_classifier_details(adaboost_data_balancer, adaboost_parameters, clfrs.adaboost_description, classifier_parameters)

# Random forest
random_forest_data_balancer = ADASYN
random_forest_parameters = {"n_estimators": 115, "max_features": "log2", "max_depth": 45, "criterion": "entropy", "class_weight": "balanced"}
append_classifier_details(random_forest_data_balancer, random_forest_parameters, clfrs.random_forest_description, classifier_parameters)

# K-nearest neighbours
k_nearest_data_balancer = TomekLinks
k_nearest_parameters = {"weights": "distance", "algorithm": "auto", "leaf_size": 30, "n_neighbours": 20, "p": 3}
append_classifier_details(k_nearest_data_balancer, k_nearest_parameters, clfrs.k_nearest_description, classifier_parameters)

# Gaussian Naive Bayes
gaussian_naive_bayes_data_balancer = SMOTEENN
gaussian_naive_bayes_parameters = {}
append_classifier_details(gaussian_naive_bayes_data_balancer, gaussian_naive_bayes_parameters, clfrs.gaussian_naive_bayes_description, classifier_parameters)

# Bernoulli Naive Bayes
bernoulli_naive_bayes_data_balancer = ADASYN
bernoulli_naive_bayes_parameters = {}
append_classifier_details(bernoulli_naive_bayes_data_balancer, bernoulli_naive_bayes_parameters, clfrs.bernoulli_naive_bayes_description, classifier_parameters)
