from config import constants as const
import operator

class RunStatistics:
    """Contains functionality to calculate predictive accuracy rates across runs"""

    def __init__(self, run_accuracy_list=None, run_roc_list=None):
        if run_accuracy_list is None:
            run_accuracy_list = []
        if run_roc_list is None:
            run_roc_list = []
        self.errors = run_accuracy_list
        self.roc_list = run_roc_list

    def append_run_result(self, test_dict, roc_result_list):
        self.errors.append(test_dict)
        self.roc_list.append(roc_result_list)

    def calculate_average_run_accuracy(self):
        overall_true_rate, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, true_positive_rate_cutoff, true_negative_rate_cutoff, \
            false_positive_rate_cutoff, false_negative_rate_cutoff, unclassified_cutoff = [0] * 10
        for result_dictionary in self.errors:
            overall_true_rate += result_dictionary["avg_true_rate"]
            true_positive_rate += result_dictionary["avg_true_positive_rate"]
            true_negative_rate += result_dictionary["avg_true_negative_rate"]
            false_positive_rate += result_dictionary["avg_false_positive_rate"]
            false_negative_rate += result_dictionary["avg_false_negative_rate"]
            true_positive_rate_cutoff += result_dictionary["avg_true_positive_rate_with_prob_cutoff"]
            true_negative_rate_cutoff += result_dictionary["avg_true_negative_rate_with_prob_cutoff"]
            false_positive_rate_cutoff += result_dictionary["avg_false_positive_rate_with_prob_cutoff"]
            false_negative_rate_cutoff += result_dictionary["avg_false_negative_rate_with_prob_cutoff"]
            unclassified_cutoff += result_dictionary["avg_false_negative_rate_with_prob_cutoff"]

        avg_run_results = [None] * 10
        avg_run_results[0] = overall_true_rate / float(len(self.errors))
        avg_run_results[1] = true_positive_rate / float(len(self.errors))
        avg_run_results[2] = true_negative_rate / float(len(self.errors))
        avg_run_results[3] = false_positive_rate / float(len(self.errors))
        avg_run_results[4] = false_negative_rate / float(len(self.errors))
        avg_run_results[5] = true_positive_rate_cutoff / float(len(self.errors))
        avg_run_results[6] = true_negative_rate_cutoff / float(len(self.errors))
        avg_run_results[7] = false_positive_rate_cutoff / float(len(self.errors))
        avg_run_results[8] = false_negative_rate_cutoff / float(len(self.errors))
        avg_run_results[9] = unclassified_cutoff / float(len(self.errors))

        return avg_run_results