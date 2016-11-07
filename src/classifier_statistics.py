import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from sklearn.metrics import brier_score_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score

import constants as const


class ClassifierStatistics:
    """Contains functionality to calculate predictive accuracy rates"""

    def __init__(self, error_list=None, roc_list=None):
        if error_list is None:
            error_list = []
        if roc_list is None:
            roc_list = []
        self.results = error_list
        self.roc_list = roc_list

    @staticmethod
    def process_results(test_classification, actual_outcome, fit_time, test_probabilities=None, probability_cutoff=const.CUTOFF_RATE):
        """Compares the test_classification and actual_outcome. It returns a dictionary with the true positive,
        true negative, false positive and false negative rate."""

        # print(cohen_kappa_score(test_classification, actual_outcome))

        fold_result_dict = {}
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

        fold_result_dict["true positive rate"] = true_positive_count / float(np.count_nonzero(actual_outcome == 1))
        fold_result_dict["true negative rate"] = true_negative_count / float(np.count_nonzero(actual_outcome == 0))
        fold_result_dict["false positive rate"] = false_positive_count / float(np.count_nonzero(actual_outcome == 0))
        fold_result_dict["false negative rate"] = false_negative_count / float(np.count_nonzero(actual_outcome == 1))
        fold_result_dict["true positives"] = true_positive_count
        fold_result_dict["true negatives"] = true_negative_count
        fold_result_dict["false positives"] = false_positive_count
        fold_result_dict["false negatives"] = false_negative_count
        fold_result_dict["test_classification"] = test_classification
        test_probability_arr = []
        for value in test_probabilities:
            val = value[-1:][0]
            if val < 0 and val != -1:
                val = 0
            elif val > 1:
                val = 1
            test_probability_arr.append(val)
        fold_result_dict["test_probabilities"] = test_probability_arr
        fold_result_dict["actual_outcome"] = actual_outcome

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

        fold_result_dict["true positive rate with probability cutoff"] = true_positive_count_with_probability_cutoff / float(np.count_nonzero(actual_outcome == 1))
        fold_result_dict["true negative rate with probability cutoff"] = true_negative_count_with_probability_cutoff / float(np.count_nonzero(actual_outcome == 0))
        fold_result_dict["false positive rate with probability cutoff"] = false_positive_count_with_probability_cutoff / float(np.count_nonzero(actual_outcome == 0))
        fold_result_dict["false negative rate with probability cutoff"] = false_negative_count_with_probability_cutoff / float(np.count_nonzero(actual_outcome == 1))
        fold_result_dict["unclassified with probability cutoff"] = unclassified / float(len(actual_outcome))
        fold_result_dict["fit_time"] = fit_time

        return fold_result_dict

    def calculate_and_append_fold_accuracy(self, test_classification, actual_outcome, roc_tpr, roc_fpr, fit_time, test_probabilities=None):
        """Calculates predictive accuracy for fold data and appends it to errors list"""
        self.results.append(self.process_results(test_classification, actual_outcome, fit_time, test_probabilities=test_probabilities))
        self.roc_list.append((roc_tpr, roc_fpr))

    def calculate_average_results(self):
        """Averages true positive, true negative, false positive and false negative rate contained in errors"""

        avg_result_dict = {}
        avg_true_rate = 0
        avg_true_positive_rate = 0
        avg_true_negative_rate = 0
        avg_false_positive_rate = 0
        avg_false_negative_rate = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        avg_true_positive_rate_probability_cutoff = 0
        avg_true_negative_rate_probability_cutoff = 0
        avg_false_positive_rate_probability_cutoff = 0
        avg_false_negative_rate_probability_cutoff = 0
        unclassified = 0
        classification_arr = []
        actual_result_arr = []
        classification_probabilities_arr = []
        balanced_accuracy_arr = []
        auc_arr = []
        brier_score_arr = []
        fit_time_arr = []
        mcc_arr = []
        true_positive_rate_arr = []
        true_negative_rate_arr = []
        false_positive_rate_arr = []
        false_negative_rate_arr = []
        hmeasure_arr = []

        avg_mcc = 0
        avg_auc = 0
        avg_brier_score = 0
        avg_fit_time = 0

        for error_dict in self.results:
            balanced_accuracy = (error_dict["true positive rate"] + error_dict["true negative rate"]) / 2
            mcc = matthews_corrcoef(error_dict["actual_outcome"], error_dict["test_classification"])

            brier_score, auc, hmeasure = -9999, -9999, -9999
            if -1 not in error_dict["test_probabilities"]:
                brier_score = brier_score_loss(error_dict["actual_outcome"], error_dict["test_probabilities"])
                auc = roc_auc_score(error_dict["actual_outcome"], error_dict["test_probabilities"])

                # Apply H-measure
                count = 0
                while True:
                    try:
                        hmeasure_r = importr("hmeasure")
                        actual_outcome = robjects.FloatVector(error_dict["actual_outcome"])
                        test_probabilities = robjects.FloatVector(error_dict["test_probabilities"])
                        hmeasure_result = hmeasure_r.HMeasure(actual_outcome, test_probabilities)
                        hmeasure = float(hmeasure_result[0][0][0])
                        break

                    except Exception:
                        if count > 0:
                            raise
                        else:
                            utils = rpackages.importr('utils')
                            utils.chooseCRANmirror(ind=1)
                            utils.install_packages("hmeasure")




            avg_true_rate += balanced_accuracy
            avg_true_positive_rate += error_dict["true positive rate"]
            avg_true_negative_rate += error_dict["true negative rate"]
            avg_false_positive_rate += error_dict["false positive rate"]
            avg_false_negative_rate += error_dict["false negative rate"]
            true_positives += error_dict["true positives"]
            true_negatives += error_dict["true negatives"]
            false_positives += error_dict["false positives"]
            false_negatives += error_dict["false negatives"]
            avg_true_positive_rate_probability_cutoff += error_dict["true positive rate with probability cutoff"]
            avg_true_negative_rate_probability_cutoff += error_dict["true negative rate with probability cutoff"]
            avg_false_positive_rate_probability_cutoff += error_dict["false positive rate with probability cutoff"]
            avg_false_negative_rate_probability_cutoff += error_dict["false negative rate with probability cutoff"]
            unclassified += error_dict["unclassified with probability cutoff"]
            classification_arr.append(error_dict["test_classification"])
            actual_result_arr.append(error_dict["actual_outcome"])
            classification_probabilities_arr.append(error_dict["test_probabilities"])
            avg_fit_time += error_dict["fit_time"]
            avg_mcc += mcc
            avg_auc += auc
            avg_brier_score += brier_score

            balanced_accuracy_arr.append(balanced_accuracy)
            auc_arr.append(auc)
            brier_score_arr.append(brier_score)
            mcc_arr.append(mcc)
            fit_time_arr.append(error_dict["fit_time"])
            hmeasure_arr.append(hmeasure)
            true_positive_rate_arr.append(error_dict["true positive rate"])
            true_negative_rate_arr.append(error_dict["true negative rate"])
            false_positive_rate_arr.append(error_dict["false positive rate"])
            false_negative_rate_arr.append(error_dict["false negative rate"])

        avg_result_dict["avg_true_rate"] = avg_true_rate / float(len(self.results))
        avg_result_dict["avg_true_positive_rate"] = avg_true_positive_rate / float(len(self.results))
        avg_result_dict["avg_true_negative_rate"] = avg_true_negative_rate / float(len(self.results))
        avg_result_dict["avg_false_positive_rate"] = avg_false_positive_rate / float(len(self.results))
        avg_result_dict["avg_false_negative_rate"] = avg_false_negative_rate / float(len(self.results))
        avg_result_dict["true positives"] = true_positives
        avg_result_dict["true negatives"] = true_negatives
        avg_result_dict["false positives"] = false_positives
        avg_result_dict["false negatives"] = false_negatives
        avg_result_dict["avg_true_positive_rate_with_prob_cutoff"] = avg_true_positive_rate_probability_cutoff / float(len(self.results))
        avg_result_dict["avg_true_negative_rate_with_prob_cutoff"] = avg_true_negative_rate_probability_cutoff / float(len(self.results))
        avg_result_dict["avg_false_positive_rate_with_prob_cutoff"] = avg_false_positive_rate_probability_cutoff / float(len(self.results))
        avg_result_dict["avg_false_negative_rate_with_prob_cutoff"] = avg_false_negative_rate_probability_cutoff / float(len(self.results))
        avg_result_dict["avg_unclassified_with_prob_cutoff"] = unclassified / float(len(self.results))
        avg_result_dict["test_classification"] = classification_arr
        avg_result_dict["test_probabilities"] = classification_probabilities_arr
        avg_result_dict["actual_outcome"] = actual_result_arr
        avg_result_dict["fit_time"] = avg_fit_time
        avg_result_dict["avg_auc"] = avg_auc / float(len(self.results))
        avg_result_dict["avg_brier_score"] = avg_brier_score / float(len(self.results))
        avg_result_dict["avg_mcc"] = avg_mcc / float(len(self.results))

        avg_result_dict["balanced_accuracy_arr"] = balanced_accuracy_arr
        avg_result_dict["auc_arr"] = auc_arr
        avg_result_dict["brier_score_arr"] = brier_score_arr
        avg_result_dict["fit_time_arr"] = fit_time_arr
        avg_result_dict["mcc_arr"] = mcc_arr
        avg_result_dict["true_positive_rate_arr"] = true_positive_rate_arr
        avg_result_dict["true_negative_rate_arr"] = true_negative_rate_arr
        avg_result_dict["false_positive_rate_arr"] = false_positive_rate_arr
        avg_result_dict["false_negative_rate_arr"] = false_negative_rate_arr
        avg_result_dict["hmeasure_arr"] = hmeasure_arr

        return avg_result_dict
