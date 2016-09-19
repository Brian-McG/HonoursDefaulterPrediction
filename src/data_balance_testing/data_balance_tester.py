"""Primary script used to execute the defaulter prediction"""
import multiprocessing
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from multiprocessing import Manager

from joblib import Parallel
from joblib import delayed

import config.classifiers as cfr
import visualisation as vis
from config import constants as const
from config import data_sets
from data_balancer_result_recorder import DataBalancerResultRecorder
from data_preprocessing import apply_preprocessing
from generic_classifier import GenericClassifier


def main():
    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")

            # Preprocess data set
            input_defaulter_set = apply_preprocessing(input_defaulter_set)

            data_balancers = [None, ClusterCentroids, EditedNearestNeighbours, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule,
                              OneSidedSelection, RandomUnderSampler, TomekLinks, ADASYN, RandomOverSampler, SMOTE, SMOTEENN, SMOTETomek]

            logical_cpu_count = multiprocessing.cpu_count()
            manager = Manager()
            result_recorder = DataBalancerResultRecorder(result_arr=manager.list())
            data_balance_roc_results = manager.list()

            # Execute enabled classifiers
            for classifier_dict in cfr.classifiers:
                if classifier_dict["status"]:
                    print("== {0} ==".format(classifier_dict["classifier_description"]))
                    Parallel(n_jobs=logical_cpu_count)(
                        delayed(run_test)(classifier_dict, input_defaulter_set, data_balancers[z], data_balance_roc_results, result_recorder) for z in
                        range(len(data_balancers)))
                    # Plot ROC results of each balancer
                    vis.plot_mean_roc_curve_of_balancers(data_balance_roc_results, data_set["data_set_description"], classifier_dict["classifier_description"])

            print(result_recorder.results)
            if const.RECORD_RESULTS is True:
                result_recorder.save_results_to_file(data_set["data_set_description"])


def run_test(classifier_dict, input_defaulter_set, data_balancer, data_balance_roc_results, result_recorder):
    classifier_roc_results = []
    print("=== {0} ===".format(data_balancer.__name__ if data_balancer is not None else "None"))
    overall_true_rate, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, true_positive_rate_cutoff, true_negative_rate_cutoff, \
        false_positive_rate_cutoff, false_negative_rate_cutoff, unclassified_cutoff = [0] * 10

    # Execute classifier TEST_REPEAT number of times
    for i in range(const.TEST_REPEAT):
        print("==== Run {0} ====".format(i+1))
        classifier = GenericClassifier(classifier_dict["classifier"], classifier_dict["classifier_parameters"], data_balancer)
        result_dictionary = classifier.train_and_evaluate(input_defaulter_set, i)

        # Add ROC results
        classifier_roc_results.append(classifier.ml_stats.roc_list)

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

    data_balance_roc_results.append((classifier_roc_results, data_balancer.__name__ if data_balancer is not None else "None"))

    individual_data_balancer_results = [None, None, None, None, None, None, None, None, None, None]
    individual_data_balancer_results[0] = ("overall_true_rate", overall_true_rate / const.TEST_REPEAT)
    individual_data_balancer_results[1] = ("true_positive_rate", true_positive_rate / const.TEST_REPEAT)
    individual_data_balancer_results[2] = ("true_negative_rate", true_negative_rate / const.TEST_REPEAT)
    individual_data_balancer_results[3] = ("false_positive_rate", false_positive_rate / const.TEST_REPEAT)
    individual_data_balancer_results[4] = ("false_negative_rate", false_negative_rate / const.TEST_REPEAT)
    individual_data_balancer_results[5] = ("true_positive_rate_cutoff", true_positive_rate_cutoff / const.TEST_REPEAT)
    individual_data_balancer_results[6] = ("true_negative_rate_cutoff", true_negative_rate_cutoff / const.TEST_REPEAT)
    individual_data_balancer_results[7] = ("false_positive_rate_cutoff", false_positive_rate_cutoff / const.TEST_REPEAT)
    individual_data_balancer_results[8] = ("false_negative_rate_cutoff", false_negative_rate_cutoff / const.TEST_REPEAT)
    individual_data_balancer_results[9] = ("unclassified_cutoff", unclassified_cutoff / const.TEST_REPEAT)
    classifier_tuple = (data_balancer.__name__ if data_balancer is not None else "None", individual_data_balancer_results)
    result_recorder.record_results((classifier_dict["classifier_description"], classifier_tuple))


if __name__ == "__main__":

    # Run main
    main()
