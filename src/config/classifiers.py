"""Configures classifiers and their status"""
from collections import OrderedDict

from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from classifier_wrappers.clustering_launched_classification import ClusteringLaunchedClassifier
from classifier_wrappers.extreme_learning_machines import ExtremeLearningMachine

classifiers = OrderedDict()


def append_classifier_details(classifier_class, status, classifier_description, classifier_dict):
    """Adds classifier, data_balancer, status and classifier_description to classifier_arr as a dictionary"""
    classifier_dict[classifier_description] = {"classifier": classifier_class, "status": status}


# Generic Classifiers
# Clustering-Launched Classification
clustering_launched_classifier = ClusteringLaunchedClassifier
clustering_launched_classifier_description = "Clustering-Launched Classification"
clustering_launched_classifier_enabled = True
append_classifier_details(clustering_launched_classifier, clustering_launched_classifier_enabled, clustering_launched_classifier_description, classifiers)

# Extreme learning machines
elm = ExtremeLearningMachine
elm_description = "Extreme Learning Machine"
elm_enabled = True
append_classifier_details(elm, elm_enabled, elm_description, classifiers)

# Artificial Neural network
ann = MLPClassifier
ann_description = "Artificial neural network"
ann_enabled = True
append_classifier_details(ann, ann_enabled, ann_description, classifiers)

# Support Vector Machines (with RBF kernel)
svm_rdf = svm.SVC
svm_rdf_description = "SVM (RBF)"
svm_rdf_enabled = True
append_classifier_details(svm_rdf, svm_rdf_enabled, svm_rdf_description, classifiers)

# Support Vector Machines (with linear kernel)
svm_linear = svm.SVC
svm_linear_description = "SVM (linear)"
svm_linear_enabled = True
append_classifier_details(svm_linear, svm_linear_enabled, svm_linear_description, classifiers)

# Support Vector Machines (with polynomial kernel)
svm_poly = svm.SVC
svm_poly_description = "SVM (polynomial)"
svm_poly_enabled = True
append_classifier_details(svm_poly, svm_poly_enabled, svm_poly_description, classifiers)

# Logistic Regression
logistic_regression = LogisticRegression
logistic_regression_description = "Logistic regression"
logistic_regression_enabled = True
append_classifier_details(logistic_regression, logistic_regression_enabled, logistic_regression_description, classifiers)

# Decision Tree
decision_tree = DecisionTreeClassifier
decision_tree_description = "Decision Tree"
decision_tree_enabled = True
append_classifier_details(decision_tree, decision_tree_enabled, decision_tree_description, classifiers)

# AdaBoost
adaboost = AdaBoostClassifier
adaboost_description = "AdaBoost"
adaboost_enabled = True
append_classifier_details(adaboost, adaboost_enabled, adaboost_description, classifiers)

# Random forest
random_forest = RandomForestClassifier
random_forest_description = "Random forest"
random_forest_enabled = True
append_classifier_details(random_forest, random_forest_enabled, random_forest_description, classifiers)

# K-nearest neighbours
k_nearest = KNeighborsClassifier
k_nearest_description = "K-nearest neighbours"
k_nearest_enabled = True
append_classifier_details(k_nearest, k_nearest_enabled, k_nearest_description, classifiers)

# Gaussian Naive Bayes
gaussian_naive_bayes = GaussianNB
gaussian_naive_bayes_description = "Gaussian Naive Bayes"
gaussian_naive_bayes_enabled = True
append_classifier_details(gaussian_naive_bayes, gaussian_naive_bayes_enabled, gaussian_naive_bayes_description, classifiers)

# Bernoulli Naive Bayes
bernoulli_naive_bayes = BernoulliNB
bernoulli_naive_bayes_description = "Bernoulli Naive Bayes"
bernoulli_naive_bayes_enabled = True
append_classifier_details(bernoulli_naive_bayes, bernoulli_naive_bayes_enabled, bernoulli_naive_bayes_description, classifiers)
