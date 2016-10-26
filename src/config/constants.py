import sys

"""Constants used throughout the project"""
NUMBER_OF_FOLDS = 5
VERBOSE_MODE = False
RECORD_RESULTS = True
CUTOFF_RATE = 0.7
RETRY_COUNT = 3
TEST_REPEAT = 3
DATA_BALANCER_STR = "Data balancer"
RANDOM_RANGE = (0, 4294967294)
ANOVA_CHI2 = "ANOVA_CHI2"
LOGISTIC_REGRESSION = "Logistic regression"
BERNOULLI_NAIVE_BAYES = "Bernoulli Naive Bayes"
SVM_LINEAR = "SVM (linear)"
DECISION_TREE = "Decision Tree"
RANDOM_FOREST = "Random forest"

BASE_METRICS = ["Matthews correlation coefficient", "Cohen Kappa Score", "Balanced Accuracy", "Average true positive rate", "Average true negative rate", "Average false positive rate",
                "Average false negative rate"]
TITLE_ROW = ["Classifier description"] + BASE_METRICS + ["initialisation_values"]
TITLE_ROW_WITH_DEFAULT_TIME_RANGE = ["Default time range"] + TITLE_ROW
TITLE_ROW_WITH_TIME_TO_FIT = ["Classifier description"] + BASE_METRICS + ["Average time to fit each fold", "initialisation_values"]
TITLE_ROW_PARAMETER_TESTER = BASE_METRICS + ["Average true positive with cutoff",
                                             "Average true negative rate with cutoff", "Average false positive rate with cutoff", "Average false negative rate with cutoff",
                                             "Average unclassified from cutoff"]

TITLE_ROW_BALANCER_RESULT = ["Data balancer"] + TITLE_ROW
