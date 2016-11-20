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

BASE_METRICS = ["Matthews correlation coefficient", "Brier Score", "Balanced Accuracy", "AUC Score", "H-measure", "Average true positive rate", "Average true negative rate", "Average false positive rate",
                "Average false negative rate", "Standard Deviation of MCC", "Standard Deviation of BS", "Standard Deviation of BACC", "Standard Deviation of AUC", "Standard Deviation of H-measure", "Standard Deviation of TPR", "Standard Deviation of TNR"]
TITLE_ROW = ["Classifier description"] + BASE_METRICS + ["initialisation_values"]
TITLE_ROW_WITH_DEFAULT_TIME_RANGE = ["Default time range"] + TITLE_ROW
TITLE_ROW_WITH_TIME_TO_FIT = ["Classifier description"] + BASE_METRICS + ["Average time to fit each fold", "Standard Deviation of time to fit", "initialisation_values"]
TITLE_ROW_PARAMETER_TESTER = BASE_METRICS

TITLE_ROW_BALANCER_RESULT = ["Data balancer"] + TITLE_ROW
TITLE_ROW_PARAMETER_COMPARISON = ["Parameter selection", "Classifier description"] + BASE_METRICS + ["initialisation_values"]
TITLE_ROW_FEATURE_SELECTION = ["Feature selection strategy", "Classifier description"] + BASE_METRICS + ["initialisation_values", "features_selected", "feature_summary"]
