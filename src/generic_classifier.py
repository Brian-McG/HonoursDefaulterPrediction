from sklearn.model_selection import StratifiedKFold

import constants as const
from constants import verbose_print
from ml_statistics import MLStatistics
from ml_technique import train_and_evaluate_fold, MLTechnique


class GenericClassifier(MLTechnique):
    """Contains functionality to train and evaluate a classifier."""

    def __init__(self, classifier_class, classifier_parameters, data_balancer):
        self.ml_stats = MLStatistics()
        self.classifier_class = classifier_class
        self.classifier_parameters = classifier_parameters
        self.data_balancer = data_balancer

    def train_and_evaluate(self, defaulter_set, state):
        """Applies k-fold cross validation to train and evaluate a classifier"""

        if "defaulter_set" in self.classifier_parameters and self.classifier_parameters["defaulter_set"] is None:
            self.classifier_parameters["defaulter_set"] = defaulter_set

        classifier = self.classifier_class(**self.classifier_parameters)

        self.ml_stats.errors = []
        self.ml_stats.roc_list = []
        if self.data_balancer is not None:
            self.data_balancer = self.data_balancer(random_state=state)

        kf = StratifiedKFold(n_splits=const.NUMBER_OF_FOLDS, shuffle=True, random_state=state)
        index = 0

        for train, test in kf.split(defaulter_set.iloc[:, :-1].as_matrix(), defaulter_set.iloc[:, -1:].as_matrix().flatten()):
            train_and_evaluate_fold(self, defaulter_set, train, test, classifier, index, data_balancer=self.data_balancer)
            index += 1

        # Error rates
        avg_accuracy_dict = self.ml_stats.calculate_average_predictive_accuracy()

        verbose_print("\nAverage true rate: {0}".format((avg_accuracy_dict["avg_true_positive_rate"] + avg_accuracy_dict["avg_true_negative_rate"]) / 2.0))
        verbose_print("Average true positive rate: {0}".format(avg_accuracy_dict["avg_true_positive_rate"]))
        verbose_print("Average true negative rate: {0}".format(avg_accuracy_dict["avg_true_negative_rate"]))
        verbose_print("Average false positive rate: {0}".format(avg_accuracy_dict["avg_false_positive_rate"]))
        verbose_print("Average false negative rate: {0}".format(avg_accuracy_dict["avg_false_negative_rate"]))

        verbose_print("\nAverage true positive rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_true_positive_rate_with_prob_cutoff"]))
        verbose_print("Average true negative rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_true_negative_rate_with_prob_cutoff"]))
        verbose_print("Average false positive rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_false_positive_rate_with_prob_cutoff"]))
        verbose_print("Average false negative rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_false_negative_rate_with_prob_cutoff"]))
        verbose_print("Average unclassified (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_unclassified_with_prob_cutoff"]))

        return avg_accuracy_dict
