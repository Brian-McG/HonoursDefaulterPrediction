import constants as const
from constants import verbose_print
from ml_statistics import MLStatistics
from ml_technique import MLTechnique, train_and_evaluate_fold


class GenericClassifier(MLTechnique):
    """Contains functionality to train and evaluate a classifier."""

    def __init__(self, classifier, data_balancer):
        self.ml_stats = MLStatistics()
        self.classifier = classifier
        self.data_balancer = data_balancer

    def train_and_evaluate(self, defaulter_set):
        """Applies k-fold cross validation to train and evaluate a classifier"""

        self.ml_stats.errors.clear()

        for i in range(const.NUMBER_OF_FOLDS):
            train_and_evaluate_fold(self, defaulter_set, i, self.classifier, data_balancer=self.data_balancer)

        # Error rates
        avg_accuracy_dict = self.ml_stats.calculate_average_predictive_accuracy()

        verbose_print("\nAverage true positive rate: {0}".format(avg_accuracy_dict["avg_true_positive_rate"]))
        verbose_print("Average true negative rate: {0}".format(avg_accuracy_dict["avg_true_negative_rate"]))
        verbose_print("Average false positive rate: {0}".format(avg_accuracy_dict["avg_false_positive_rate"]))
        verbose_print("Average false negative rate: {0}".format(avg_accuracy_dict["avg_false_negative_rate"]))

        verbose_print("\nAverage true positive rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_true_positive_rate_with_prob_cutoff"]))
        verbose_print("Average true negative rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_true_negative_rate_with_prob_cutoff"]))
        verbose_print("Average false positive rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_false_positive_rate_with_prob_cutoff"]))
        verbose_print("Average false negative rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_false_negative_rate_with_prob_cutoff"]))
        verbose_print("Average unclassified (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_unclassified_with_prob_cutoff"]))

        return avg_accuracy_dict
