"""Contains all classifiers used (except ANNs)"""
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import ClusterCentroids
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from classifier_wrappers.clustering_launched_classification import ClusteringLaunchedClassifier
from classifier_wrappers.extreme_learning_machines import ExtremeLearningMachine

classifiers = []


def append_classifier_details(data_balancer, classifier_class, classifier_parameters, status, classifier_description, classifier_arr):
    """Adds classifier, data_balancer, status and classifier_description to classifier_arr as a dictionary"""
    classifier_arr.append({"classifier": classifier_class, "classifier_parameters": classifier_parameters, "data_balancer": data_balancer, "status": status,
                           "classifier_description": classifier_description})


# Generic Classifiers
# Clustering-Launched Classification
clustering_launched_classifier_data_balancer = ClusterCentroids
clustering_launched_classifier = ClusteringLaunchedClassifier
clustering_launched_classifier_parameters = {"d": 0.5}
clustering_launched_classifier_enabled = False
append_classifier_details(clustering_launched_classifier_data_balancer, clustering_launched_classifier, clustering_launched_classifier_parameters,
                          clustering_launched_classifier_enabled, "Clustering-Launched Classification", classifiers)

# Extreme learning machines
elm_data_balancer = SMOTEENN
elm = ExtremeLearningMachine
elm_parameters = {"defaulter_set": None, "layers": [(20, "sigm"), (2, "rbf_l2")]}
elm_enabled = True
append_classifier_details(elm_data_balancer, elm, elm_parameters, elm_enabled, "Extreme Learning Machine", classifiers)

# Artificial Neural network
ann_data_balancer = SMOTEENN
ann = MLPClassifier
ann_parameters = {"hidden_layer_sizes": (5,), "max_iter": 1000, "solver": "sgd"}
ann_enabled = False
append_classifier_details(ann_data_balancer, ann, ann_parameters, ann_enabled, "Artificial neural network", classifiers)

# Support Vector Machines (with RDF kernel)
svm_rdf_data_balancer = ClusterCentroids
svc_rdf = svm.SVC
svc_parameters = {"cache_size": 1000, "gamma": "auto", "kernel": "rbf", "class_weight": "balanced", "probability": True}
svc_rdf_enabled = False
append_classifier_details(svm_rdf_data_balancer, svc_rdf, svc_parameters, svc_rdf_enabled, "SVM (RDF)", classifiers)

# Support Vector Machines (with linear kernel)
svm_linear_data_balancer = ADASYN
svm_linear = svm.SVC
svm_linear_parameters = {"C": 7.371053, "decision_function_shape": "ovo", "cache_size": 1000, "gamma": 0.316631579, "kernel": "linear", "probability": True}
svm_linear_enabled = False
append_classifier_details(svm_linear_data_balancer, svm_linear, svm_linear_parameters, svm_linear_enabled, "SVM (linear)", classifiers)

# Support Vector Machines (with polynomial kernel)
svm_poly_data_balancer = SMOTEENN
svm_poly = svm.SVC
svm_poly_parameters = {"cache_size": 1000, "gamma": "auto", "kernel": "poly", "degree": 3, "class_weight": "balanced", "probability": True}
svm_poly_enabled = False
append_classifier_details(svm_poly_data_balancer, svm_poly, svm_poly_parameters, svm_poly_enabled, "SVM (polynomial)", classifiers)

# Logistic Regression
logistic_regression_data_balancer = SMOTEENN
logistic_regression = LogisticRegression
logistic_regression_parameters = {"penalty": "l2", "dual": False, "fit_intercept": True, "intercept_scaling": 1, "solver": "newton-cg", "max_iter": 100, "multi_class": "ovr"}
logistic_regression_enabled = False
append_classifier_details(logistic_regression_data_balancer, logistic_regression, logistic_regression_parameters, logistic_regression_enabled, "Logistic regression",
                          classifiers)

# Decision Tree
decision_tree_data_balancer = SMOTEENN
decision_tree = DecisionTreeClassifier
decision_tree_parameters = {"max_features": "auto", "class_weight": "balanced"}
decision_tree_enabled = False
append_classifier_details(decision_tree_data_balancer, decision_tree, decision_tree_parameters, decision_tree_enabled, "Decision Tree", classifiers)

# AdaBoost
adaboost_data_balancer = SMOTEENN
adaboost = AdaBoostClassifier
adaboost_parameters = {"n_estimators": 3000, "learning_rate": 0.01}
adaboost_enabled = False
append_classifier_details(adaboost_data_balancer, adaboost, adaboost_parameters, adaboost_enabled, "AdaBoost", classifiers)

# Random forest
random_forest_data_balancer = SMOTEENN
random_forest = RandomForestClassifier
random_forest_parameters = {"n_estimators": 10, "n_jobs": -1, "class_weight": "balanced"}
random_forest_enabled = False
append_classifier_details(random_forest_data_balancer, random_forest, random_forest_parameters, random_forest_enabled, "Random forest", classifiers)

# K-nearest neighbours
k_nearest_data_balancer = SMOTEENN
k_nearest = KNeighborsClassifier
k_nearest_parameters = {"n_neighbors": 100}
k_nearest_enabled = False
append_classifier_details(k_nearest_data_balancer, k_nearest, k_nearest_parameters, k_nearest_enabled, "K-nearest neighbours", classifiers)

# Bernoulli Naive Bayes
bernoulli_naive_bayes_data_balancer = SMOTEENN
bernoulli_naive_bayes = BernoulliNB
bernoulli_naive_bayes_parameters = {}
bernoulli_naive_bayes_enabled = False
append_classifier_details(bernoulli_naive_bayes_data_balancer, bernoulli_naive_bayes, bernoulli_naive_bayes_parameters, bernoulli_naive_bayes_enabled, "Bernoulli Naive Bayes",
                          classifiers)

# Voting classifier
voting_classifier_data_balancer = SMOTEENN
classifier_one = AdaBoostClassifier(n_estimators=3000, learning_rate=0.01)
classifier_two = BernoulliNB()
classifier_three = RandomForestClassifier(n_estimators=10, n_jobs=-1)
classifier_four = KNeighborsClassifier(n_neighbors=100)
voting_classifier = VotingClassifier
voting_classifier_parameters = {
    "estimators": [("classifier_one", classifier_one), ("classifier_two", classifier_two), ("classifier_three", classifier_three), ("classifier_four", classifier_four)],
    "voting": "hard"}
voting_classifier_enabled = False
append_classifier_details(voting_classifier_data_balancer, voting_classifier, voting_classifier_parameters, voting_classifier_enabled, "Voting classifier", classifiers)
