"""Contains all classifiers used (except ANNs)"""
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# List of classifier information
generic_classifiers = []
non_generic_classifiers = []


def append_classifier_details(data_balancer, classifier, status, classifier_description, classifier_arr):
    """Adds classifier, data_balancer, status and classifier_description to classifier_arr as a dictionary"""
    classifier_arr.append({'classifier': classifier, 'data_balancer': data_balancer, 'status': status,
                           'classifier_description': classifier_description})


# Non-generic Classifiers
# Artificial Neural network - is added in main.py due to use of Processes which requires it be declared in the main method
ann_enabled = True
ann_data_balancer = SMOTEENN()

# Generic Classifiers
# Support Vector Machines (with RDF kernel)
svm_rdf_data_balancer = ClusterCentroids()
svc_rdf = svm.SVC(cache_size=1000, gamma='auto', kernel='rbf', class_weight='balanced', probability=True)
svc_rdf_enabled = False
append_classifier_details(svm_rdf_data_balancer, svc_rdf, svc_rdf_enabled, "SVM (RDF)", generic_classifiers)

# Support Vector Machines (with linear kernel)
svm_linear_data_balancer = None
svm_linear = svm.SVC(cache_size=1000, gamma='auto', kernel='linear', class_weight='balanced', probability=True)
svm_linear_enabled = False
append_classifier_details(svm_linear_data_balancer, svm_linear, svm_linear_enabled, "SVM (linear)", generic_classifiers)

# Support Vector Machines (with polynomial kernel)
svm_poly_data_balancer = SMOTEENN()
svm_poly = svm.SVC(cache_size=1000, gamma='auto', kernel='poly', degree=3, class_weight='balanced', probability=True)
svm_poly_enabled = False
append_classifier_details(svm_poly_data_balancer, svm_poly, svm_poly_enabled, "SVM (polynomial)", generic_classifiers)

# Logistic Regression
logistic_regression_data_balancer = SMOTEENN()
logistic_regression = LogisticRegression(penalty='l2', dual=False, fit_intercept=True, intercept_scaling=1,
                                         solver='newton-cg', max_iter=100, multi_class='ovr')
logistic_regression_enabled = False
append_classifier_details(logistic_regression_data_balancer, logistic_regression, logistic_regression_enabled,
                          "Logistic regression", generic_classifiers)

# Decision Tree
decision_tree_data_balancer = SMOTEENN()
decision_tree = DecisionTreeClassifier(max_features='auto', class_weight='balanced')
decision_tree_enabled = True
append_classifier_details(decision_tree_data_balancer, decision_tree, decision_tree_enabled, "Decision Tree",
                          generic_classifiers)

# AdaBoost
adaboost_data_balancer = SMOTEENN()
adaboost = AdaBoostClassifier(n_estimators=3000, learning_rate=0.01)
adaboost_enabled = False
append_classifier_details(adaboost_data_balancer, adaboost, adaboost_enabled, "AdaBoost", generic_classifiers)

# Random forest
random_forest_data_balancer = SMOTEENN()
random_forest = RandomForestClassifier(n_estimators=10, n_jobs=-1, class_weight='balanced')
random_forest_enabled = False
append_classifier_details(random_forest_data_balancer, random_forest, random_forest_enabled, "Random forest",
                          generic_classifiers)

# K-nearest neighbours
k_nearest_data_balancer = SMOTEENN()
k_nearest = KNeighborsClassifier(n_neighbors=100)
k_nearest_enabled = False
append_classifier_details(k_nearest_data_balancer, k_nearest, k_nearest_enabled, "K-nearest neighbours",
                          generic_classifiers)

# Bernoulli Naive Bayes
bernoulli_naive_bayes_data_balancer = SMOTEENN()
bernoulli_naive_bayes = BernoulliNB()
bernoulli_naive_bayes_enabled = False
append_classifier_details(bernoulli_naive_bayes_data_balancer, bernoulli_naive_bayes, bernoulli_naive_bayes_enabled,
                          "Bernoulli Naive Bayes", generic_classifiers)

# Voting classifier
voting_classifier_data_balancer = SMOTEENN()
classifier_one = AdaBoostClassifier(n_estimators=3000, learning_rate=0.01)
classifier_two = BernoulliNB()
classifier_three = RandomForestClassifier(n_estimators=10, n_jobs=-1)
classifier_four = KNeighborsClassifier(n_neighbors=100)
voting_classifier = VotingClassifier(estimators=[('classifier_one', classifier_one), ('classifier_two', classifier_two),
                                                 ('classifier_three', classifier_three),
                                                 ('classifier_four', classifier_four)], voting='hard')
voting_classifier_enabled = False
append_classifier_details(voting_classifier_data_balancer, voting_classifier, voting_classifier_enabled,
                          "Voting classifier", generic_classifiers)
