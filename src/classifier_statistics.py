import numpy as np

from config import constants as const


class ClassifierStatistics:
    """Contains functionality to calculate predictive accuracy rates"""

    def __init__(self, error_list=None, roc_list=None):
        if error_list is None:
            error_list = []
        if roc_list is None:
            roc_list = []
        self.errors = error_list
        self.roc_list = roc_list

    @staticmethod
    def calculate_classification_accuracy(test_classification, actual_outcome, test_probabilities=None, probability_cutoff=const.CUTOFF_RATE):
        """Compares the test_classification and actual_outcome. It returns a dictionary with the true positive,
        true negative, false positive and false negative rate."""
        fold_accuracy_dict = {}
        true_positive_count = 0
        true_negative_count = 0
        false_positive_count = 0
        false_negative_count = 0

        for z in range(len(test_classification)):
            # True positive
            if test_classification[z] == 1 and actual_outcome[z] == 1:
                true_positive_count += 1
            # True negative
            elif test_classification[z] != 1 and actual_outcome[z] != 1:
                true_negative_count += 1
            # False positive (Type I)
            elif test_classification[z] == 1 and actual_outcome[z] != 1:
                false_positive_count += 1
            # False negative (Type II)
            elif test_classification[z] != 1 and actual_outcome[z] == 1:
                false_negative_count += 1
            # Debug - error detection
            else:
                raise RuntimeError("Unexpected classification")

        fold_accuracy_dict["true positive rate"] = true_positive_count / float(np.count_nonzero(actual_outcome == 1))
        fold_accuracy_dict["true negative rate"] = true_negative_count / float(np.count_nonzero(actual_outcome == 0))
        fold_accuracy_dict["false positive rate"] = false_positive_count / float(np.count_nonzero(actual_outcome == 0))
        fold_accuracy_dict["false negative rate"] = false_negative_count / float(np.count_nonzero(actual_outcome == 1))

        # Probabilities
        true_positive_count_with_probability_cutoff = 0
        true_negative_count_with_probability_cutoff = 0
        false_positive_count_with_probability_cutoff = 0
        false_negative_count_with_probability_cutoff = 0
        unclassified = 0

        for z in range(len(test_probabilities)):
            # True positive
            if test_probabilities[z][1] > probability_cutoff and actual_outcome[z] == 1:
                true_positive_count_with_probability_cutoff += 1
            # True negative
            elif test_probabilities[z][0] > probability_cutoff and actual_outcome[z] != 1:
                true_negative_count_with_probability_cutoff += 1
            # False positive
            elif test_probabilities[z][1] > probability_cutoff and actual_outcome[z] != 1:
                false_positive_count_with_probability_cutoff += 1
            # False negative
            elif test_probabilities[z][0] > probability_cutoff and actual_outcome[z] == 1:
                false_negative_count_with_probability_cutoff += 1
            else:
                unclassified += 1

        fold_accuracy_dict["true positive rate with probability cutoff"] = true_positive_count_with_probability_cutoff / float(np.count_nonzero(actual_outcome == 1))
        fold_accuracy_dict["true negative rate with probability cutoff"] = true_negative_count_with_probability_cutoff / float(np.count_nonzero(actual_outcome == 0))
        fold_accuracy_dict["false positive rate with probability cutoff"] = false_positive_count_with_probability_cutoff / float(np.count_nonzero(actual_outcome == 0))
        fold_accuracy_dict["false negative rate with probability cutoff"] = false_negative_count_with_probability_cutoff / float(np.count_nonzero(actual_outcome == 1))
        fold_accuracy_dict["unclassified with probability cutoff"] = unclassified / float(len(actual_outcome))

        return fold_accuracy_dict

    def calculate_and_append_fold_accuracy(self, test_classification, actual_outcome, roc_tpr, roc_fpr, test_probabilities=None):
        """Calculates predictive accuracy for fold data and appends it to errors list"""
        self.errors.append(self.calculate_classification_accuracy(test_classification, actual_outcome, test_probabilities=test_probabilities))
        self.roc_list.append((roc_tpr, roc_fpr))

    def calculate_average_predictive_accuracy(self):
        """Averages true positive, true negative, false positive and false negative rate contained in errors"""

        assert len(self.errors) == const.NUMBER_OF_FOLDS

        avg_accuracy_dict = {}
        avg_true_rate = 0
        avg_true_positive_rate = 0
        avg_true_negative_rate = 0
        avg_false_positive_rate = 0
        avg_false_negative_rate = 0
        avg_true_positive_rate_probability_cutoff = 0
        avg_true_negative_rate_probability_cutoff = 0
        avg_false_positive_rate_probability_cutoff = 0
        avg_false_negative_rate_probability_cutoff = 0
        unclassified = 0
        for error_dict in self.errors:
            avg_true_rate += (error_dict["true positive rate"] + error_dict["true negative rate"]) / 2
            avg_true_positive_rate += error_dict["true positive rate"]
            avg_true_negative_rate += error_dict["true negative rate"]
            avg_false_positive_rate += error_dict["false positive rate"]
            avg_false_negative_rate += error_dict["false negative rate"]
            avg_true_positive_rate_probability_cutoff += error_dict["true positive rate with probability cutoff"]
            avg_true_negative_rate_probability_cutoff += error_dict["true negative rate with probability cutoff"]
            avg_false_positive_rate_probability_cutoff += error_dict["false positive rate with probability cutoff"]
            avg_false_negative_rate_probability_cutoff += error_dict["false negative rate with probability cutoff"]
            unclassified += error_dict["unclassified with probability cutoff"]

        avg_accuracy_dict["avg_true_rate"] = avg_true_rate / float(len(self.errors))
        avg_accuracy_dict["avg_true_positive_rate"] = avg_true_positive_rate / float(len(self.errors))
        avg_accuracy_dict["avg_true_negative_rate"] = avg_true_negative_rate / float(len(self.errors))
        avg_accuracy_dict["avg_false_positive_rate"] = avg_false_positive_rate / float(len(self.errors))
        avg_accuracy_dict["avg_false_negative_rate"] = avg_false_negative_rate / float(len(self.errors))
        avg_accuracy_dict["avg_true_positive_rate_with_prob_cutoff"] = avg_true_positive_rate_probability_cutoff / float(len(self.errors))
        avg_accuracy_dict["avg_true_negative_rate_with_prob_cutoff"] = avg_true_negative_rate_probability_cutoff / float(len(self.errors))
        avg_accuracy_dict["avg_false_positive_rate_with_prob_cutoff"] = avg_false_positive_rate_probability_cutoff / float(len(self.errors))
        avg_accuracy_dict["avg_false_negative_rate_with_prob_cutoff"] = avg_false_negative_rate_probability_cutoff / float(len(self.errors))
        avg_accuracy_dict["avg_unclassified_with_prob_cutoff"] = unclassified / float(len(self.errors))

        return avg_accuracy_dict