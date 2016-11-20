"""Contains all classifier parameters for Lima TB dataset only using balancers and default parameters for rest"""
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
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

import config.classifiers as clfrs

classifier_parameters = {}


def append_classifier_details(classifier_data_balancer, classifier_parameters_dict, classifier_description, classifier_dict):
    """Adds classifier, data_balancer, status and classifier_description to classifier_arr as a dictionary"""
    classifier_dict[classifier_description] = {"data_balancer": classifier_data_balancer, "classifier_parameters": classifier_parameters_dict}


# Generic classifier_parameters
# Clustering-Launched Classification
clustering_launched_classifier_data_balancer = InstanceHardnessThreshold
clustering_launched_classifier_parameters = {"d": 0.3}
append_classifier_details(clustering_launched_classifier_data_balancer, clustering_launched_classifier_parameters, clfrs.clustering_launched_classifier_description,
                          classifier_parameters)

# Extreme learning machines
elm_data_balancer = RandomUnderSampler
elm_parameters = {"layers": [(100, 'sigm')]}
append_classifier_details(elm_data_balancer, elm_parameters, clfrs.elm_description, classifier_parameters)

# Artificial Neural network
ann_data_balancer = ADASYN
ann_parameters = {}
append_classifier_details(ann_data_balancer, ann_parameters, clfrs.ann_description, classifier_parameters)

# Support Vector Machines (with RDF kernel)
svm_rdf_data_balancer = RandomUnderSampler
svm_parameters = {"kernel": "rbf", "max_iter": 1000000}
append_classifier_details(svm_rdf_data_balancer, svm_parameters, clfrs.svm_rdf_description, classifier_parameters)

# Support Vector Machines (with linear kernel)
svm_linear_data_balancer = RandomOverSampler
svm_linear_parameters = {"kernel": "linear", "max_iter": 1000000}
append_classifier_details(svm_linear_data_balancer, svm_linear_parameters, clfrs.svm_linear_description, classifier_parameters)

# Support Vector Machines (with polynomial kernel)
svm_poly_data_balancer = RandomOverSampler
svm_poly_parameters = {"kernel": "poly", "max_iter": 1000000}
append_classifier_details(svm_poly_data_balancer, svm_poly_parameters, clfrs.svm_poly_description, classifier_parameters)

# Logistic Regression
logistic_regression_data_balancer = ADASYN
logistic_regression_parameters = {}
append_classifier_details(logistic_regression_data_balancer, logistic_regression_parameters, clfrs.logistic_regression_description, classifier_parameters)

# Decision Tree
decision_tree_data_balancer = RandomUnderSampler
decision_tree_parameters = {}
append_classifier_details(decision_tree_data_balancer, decision_tree_parameters, clfrs.decision_tree_description, classifier_parameters)

# AdaBoost
adaboost_data_balancer = RandomOverSampler
adaboost_parameters = {}
append_classifier_details(adaboost_data_balancer, adaboost_parameters, clfrs.adaboost_description, classifier_parameters)

# Random forest
random_forest_data_balancer = RandomUnderSampler
random_forest_parameters = {}
append_classifier_details(random_forest_data_balancer, random_forest_parameters, clfrs.random_forest_description, classifier_parameters)

# K-nearest neighbours
k_nearest_data_balancer = SMOTEENN
k_nearest_parameters = {}
append_classifier_details(k_nearest_data_balancer, k_nearest_parameters, clfrs.k_nearest_description, classifier_parameters)

# Gaussian Naive Bayes
gaussian_naive_bayes_data_balancer = ADASYN
gaussian_naive_bayes_parameters = {}
append_classifier_details(gaussian_naive_bayes_data_balancer, gaussian_naive_bayes_parameters, clfrs.gaussian_naive_bayes_description, classifier_parameters)

# Bernoulli Naive Bayes
bernoulli_naive_bayes_data_balancer = TomekLinks
bernoulli_naive_bayes_parameters = {}
append_classifier_details(bernoulli_naive_bayes_data_balancer, bernoulli_naive_bayes_parameters, clfrs.bernoulli_naive_bayes_description, classifier_parameters)
