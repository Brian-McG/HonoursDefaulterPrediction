import constants as const


class MLStatistics:
    """Contains functionality to calculate predictive accuracy rates"""

    def __init__(self, error_list=list()):
        self.errors = error_list


    @staticmethod
    def calculate_classification_accuracy(test_classification, actual_outcome):
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

        fold_accuracy_dict["true positive rate"] = true_positive_count / float(len(test_classification))
        fold_accuracy_dict["true negative rate"] = true_negative_count / float(len(test_classification))
        fold_accuracy_dict["false positive rate"] = false_positive_count / float(len(test_classification))
        fold_accuracy_dict["false negative rate"] = false_negative_count / float(len(test_classification))

        return fold_accuracy_dict

    def calculate_and_append_fold_accuracy(self, test_classification, actual_outcome):
        """Calculates predictive accuracy for fold data and appends it to errors list"""
        self.errors.append(self.calculate_classification_accuracy(test_classification, actual_outcome))

    def calculate_average_predictive_accuracy(self):
        """Averages true positive, true negative, false positive and false negative rate contained in errors"""
        avg_accuracy_dict = {}
        avg_true_positive_rate = 0
        avg_true_negative_rate = 0
        avg_false_positive_rate = 0
        avg_false_negative_rate = 0
        for error_dict in self.errors:
            avg_true_positive_rate += error_dict["true positive rate"]
            avg_true_negative_rate += error_dict["true negative rate"]
            avg_false_positive_rate += error_dict["false positive rate"]
            avg_false_negative_rate += error_dict["false negative rate"]

        avg_accuracy_dict["avg_true_positive_rate"] = avg_true_positive_rate / float(const.NUMBER_OF_FOLDS)
        avg_accuracy_dict["avg_true_negative_rate"] = avg_true_negative_rate / float(const.NUMBER_OF_FOLDS)
        avg_accuracy_dict["avg_false_positive_rate"] = avg_false_positive_rate / float(const.NUMBER_OF_FOLDS)
        avg_accuracy_dict["avg_false_negative_rate"] = avg_false_negative_rate / float(const.NUMBER_OF_FOLDS)

        return avg_accuracy_dict

