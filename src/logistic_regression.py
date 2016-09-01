import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer
from sklearn.svm import NuSVC
from sklearn import svm
from sklearn import preprocessing

import constants as const
from data_preprocessing import apply_preprocessing
from ml_statistics import MLStatistics
from ml_technique import MLTechnique, train_and_evaluate_fold


class AdaBoost(MLTechnique):
    """Contains functionality to train and evaluate a logistic regression classifier."""

    def __init__(self):
        self.ml_stats = MLStatistics()

    def train_and_evaluate(self, defaulter_set):
        """Applies k-fold cross validation to train and evaluate the logistic regression classifier"""
        for i in range(const.NUMBER_OF_FOLDS):
            data_balancer = SMOTEENN()
            logistic_regression = LogisticRegression(penalty='l2', dual=False, fit_intercept=True, intercept_scaling=1, solver='newton-cg', max_iter=100, multi_class='ovr')
            train_and_evaluate_fold(self, defaulter_set, i, logistic_regression, data_balancer=data_balancer)

        # Error rates
        avg_accuracy_dict = self.ml_stats.calculate_average_predictive_accuracy()

        print("\nAverage true positive rate:", avg_accuracy_dict["avg_true_positive_rate"])
        print("Average true negative rate:", avg_accuracy_dict["avg_true_negative_rate"])
        print("Average false positive rate:", avg_accuracy_dict["avg_false_positive_rate"])
        print("Average false negative rate:", avg_accuracy_dict["avg_false_negative_rate"])


if __name__ == "__main__":
    input_defaulter_set = pd.DataFrame.from_csv("../data/lima_tb/Lima-TB-Treatment-base.csv", index_col=None, encoding="UTF-8")
    #input_defaulter_set = pd.DataFrame.from_csv("../data/german_finance/german_dataset_numberised.csv", index_col=None, encoding="UTF-8")
    #input_defaulter_set = pd.DataFrame.from_csv("../data/australian_finance/australian.csv", index_col=None, encoding="UTF-8")
    #input_defaulter_set = pd.DataFrame.from_csv("../data/credit_screening/credit_screening.csv", index_col=None, encoding="UTF-8")

    input_defaulter_set = apply_preprocessing(input_defaulter_set)
    svm_imp = AdaBoost()
    svm_imp.train_and_evaluate(input_defaulter_set)