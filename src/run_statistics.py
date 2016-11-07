import math

from sklearn.metrics import brier_score_loss
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

    @staticmethod
    def calculate_std_deviation(value_arr):
        avg_balanced_acc = 0
        for value in value_arr:
            avg_balanced_acc += value
        avg_balanced_acc /= len(value_arr)

        balanced_accuracy_std_deviation = 0
        for value in value_arr:
            balanced_accuracy_std_deviation += pow(abs(value - avg_balanced_acc), 2)
        balanced_accuracy_std_deviation /= len(value_arr)
        return math.sqrt(balanced_accuracy_std_deviation)

    def calculate_average_run_accuracy(self):
        overall_true_rate, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, true_positive_rate_cutoff, true_negative_rate_cutoff, \
        false_positive_rate_cutoff, false_negative_rate_cutoff, unclassified_cutoff, matthews_correlation_coefficient, brier_score, auc_score, fit_time, hmeasure = [0] * 15
        balanced_accuracy_arr = []
        auc_arr = []
        hmeasure_arr = []
        brier_score_arr = []
        fit_time_arr = []
        mcc_arr = []
        true_positive_arr = []
        true_negative_arr = []
        false_positive_arr = []
        false_negative_arr = []

        count = 0
        for result_dictionary in self.errors:
            for z in range(len(result_dictionary["balanced_accuracy_arr"])):
                overall_true_rate += result_dictionary["balanced_accuracy_arr"][z]
                true_positive_rate += result_dictionary["true_positive_rate_arr"][z]
                true_negative_rate += result_dictionary["true_negative_rate_arr"][z]
                false_positive_rate += result_dictionary["false_positive_rate_arr"][z]
                false_negative_rate += result_dictionary["false_negative_rate_arr"][z]
                matthews_correlation_coefficient += result_dictionary["mcc_arr"][z]
                auc_score += result_dictionary["auc_arr"][z]
                brier_score += result_dictionary["brier_score_arr"][z]
                fit_time += result_dictionary["fit_time_arr"][z]
                hmeasure += result_dictionary["hmeasure_arr"][z]
                count += 1

            true_positive_rate_cutoff += result_dictionary["avg_true_positive_rate_with_prob_cutoff"]
            true_negative_rate_cutoff += result_dictionary["avg_true_negative_rate_with_prob_cutoff"]
            false_positive_rate_cutoff += result_dictionary["avg_false_positive_rate_with_prob_cutoff"]
            false_negative_rate_cutoff += result_dictionary["avg_false_negative_rate_with_prob_cutoff"]
            unclassified_cutoff += result_dictionary["avg_false_negative_rate_with_prob_cutoff"]
            balanced_accuracy_arr += result_dictionary["balanced_accuracy_arr"]
            hmeasure_arr += result_dictionary["hmeasure_arr"]
            auc_arr += result_dictionary["auc_arr"]
            brier_score_arr += result_dictionary["brier_score_arr"]
            fit_time_arr += result_dictionary["fit_time_arr"]
            mcc_arr += result_dictionary["mcc_arr"]
            true_positive_arr += result_dictionary["true_positive_rate_arr"]
            true_negative_arr += result_dictionary["true_negative_rate_arr"]
            false_positive_arr += result_dictionary["false_positive_rate_arr"]
            false_negative_arr += result_dictionary["false_negative_rate_arr"]


        avg_run_results = [None] * 31
        avg_run_results[0] = matthews_correlation_coefficient / float(count)
        avg_run_results[1] = brier_score / float(count)
        avg_run_results[2] = overall_true_rate / float(count)
        avg_run_results[3] = true_positive_rate / float(count)
        avg_run_results[4] = true_negative_rate / float(count)
        avg_run_results[5] = false_positive_rate / float(count)
        avg_run_results[6] = false_negative_rate / float(count)
        avg_run_results[7] = true_positive_rate_cutoff / float(len(self.errors))
        avg_run_results[8] = true_negative_rate_cutoff / float(len(self.errors))
        avg_run_results[9] = false_positive_rate_cutoff / float(len(self.errors))
        avg_run_results[10] = false_negative_rate_cutoff / float(len(self.errors))
        avg_run_results[11] = unclassified_cutoff / float(len(self.errors))
        avg_run_results[12] = fit_time / float(count)
        avg_run_results[14] = balanced_accuracy_arr
        avg_run_results[15] = auc_score / float(count)
        avg_run_results[16] = auc_arr
        avg_run_results[17] = brier_score_arr
        avg_run_results[18] = fit_time_arr
        avg_run_results[19] = mcc_arr
        avg_run_results[13] = self.calculate_std_deviation(balanced_accuracy_arr)
        avg_run_results[20] = self.calculate_std_deviation(mcc_arr)
        avg_run_results[21] = self.calculate_std_deviation(brier_score_arr)
        avg_run_results[22] = self.calculate_std_deviation(auc_arr)
        avg_run_results[23] = self.calculate_std_deviation(fit_time_arr)
        avg_run_results[24] = self.calculate_std_deviation(true_positive_arr)
        avg_run_results[25] = self.calculate_std_deviation(true_negative_arr)
        avg_run_results[26] = self.calculate_std_deviation(false_positive_arr)
        avg_run_results[27] = self.calculate_std_deviation(false_negative_arr)
        avg_run_results[28] = hmeasure / float(count)
        avg_run_results[29] = self.calculate_std_deviation(hmeasure_arr)
        avg_run_results[30] = hmeasure_arr

        return avg_run_results


