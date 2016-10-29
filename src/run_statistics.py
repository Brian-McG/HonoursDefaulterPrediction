import math
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef


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
        false_positive_rate_cutoff, false_negative_rate_cutoff, unclassified_cutoff, matthews_correlation_coefficient, cohen_kappa_val, fit_time = [0] * 13
        balanced_accuracy_arr = []

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
            matthews_correlation_coefficient += matthews_corrcoef(result_dictionary["actual_outcome"], result_dictionary["test_classification"])
            cohen_kappa_val += cohen_kappa_score(result_dictionary["actual_outcome"], result_dictionary["test_classification"])
            fit_time += result_dictionary["fit_time"]
            balanced_accuracy_arr += result_dictionary["balanced_accuracy_arr"]

        avg_run_results = [None] * 14
        avg_run_results[0] = matthews_correlation_coefficient / float(len(self.errors))
        avg_run_results[1] = cohen_kappa_val / float(len(self.errors))
        avg_run_results[2] = overall_true_rate / float(len(self.errors))
        avg_run_results[3] = true_positive_rate / float(len(self.errors))
        avg_run_results[4] = true_negative_rate / float(len(self.errors))
        avg_run_results[5] = false_positive_rate / float(len(self.errors))
        avg_run_results[6] = false_negative_rate / float(len(self.errors))
        avg_run_results[7] = true_positive_rate_cutoff / float(len(self.errors))
        avg_run_results[8] = true_negative_rate_cutoff / float(len(self.errors))
        avg_run_results[9] = false_positive_rate_cutoff / float(len(self.errors))
        avg_run_results[10] = false_negative_rate_cutoff / float(len(self.errors))
        avg_run_results[11] = unclassified_cutoff / float(len(self.errors))
        avg_run_results[12] = fit_time / float(len(self.errors))

        avg_balanced_acc = 0
        for balanced_accuracy in balanced_accuracy_arr:
            avg_balanced_acc += balanced_accuracy
        avg_balanced_acc /= len(balanced_accuracy_arr)

        balanced_accuracy_std_deviation = 0
        for balanced_accuracy in balanced_accuracy_arr:
            balanced_accuracy_std_deviation += pow(abs(balanced_accuracy - avg_balanced_acc), 2)
        balanced_accuracy_std_deviation /= len(balanced_accuracy_arr)
        balanced_accuracy_std_deviation = math.sqrt(balanced_accuracy_std_deviation)
        avg_run_results[13] = balanced_accuracy_std_deviation

        return avg_run_results
