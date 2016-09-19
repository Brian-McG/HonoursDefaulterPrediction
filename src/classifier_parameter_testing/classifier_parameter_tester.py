"""Primary script used to execute the defaulter prediction"""
import multiprocessing
from multiprocessing import Manager

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
from joblib import Parallel
from joblib import delayed
from sklearn.model_selection import ParameterGrid

import config.classifier_tester_parameters as ctp
import config.classifiers as cfr
from classifier_result_recorder import ClassifierResultRecorder
from config import constants as const
from config import data_sets
from data_preprocessing import apply_preprocessing
from generic_classifier import GenericClassifier


def execute_loop(classifier_dict, parameter_dict, defaulter_set_arr, results_recorder, z, parameter_grid_len):
    data_balancers = [None, ClusterCentroids, EditedNearestNeighbours, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule,
                      OneSidedSelection, RandomUnderSampler, TomekLinks, ADASYN, RandomOverSampler, SMOTE, SMOTEENN, SMOTETomek]
    if z % 5 == 0:
        print("==== {0} - {1}% ====".format(classifier_dict['classifier_description'], format((z / parameter_grid_len) * 100, '.2f')))

    for data_balancer in data_balancers:
        generic_classifier = GenericClassifier(classifier_dict['classifier'], classifier_dict['classifier_parameters'], data_balancer)
        overall_true_rate, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, true_positive_rate_cutoff, true_negative_rate_cutoff, \
            false_positive_rate_cutoff, false_negative_rate_cutoff, unclassified_cutoff = [0] * 10
        for x in range(const.TEST_REPEAT):

            result_dictionary = generic_classifier.train_and_evaluate(defaulter_set_arr, x)

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

        individual_results = [None, None, None, None, None, None, None, None, None, None]
        individual_results[0] = overall_true_rate / const.TEST_REPEAT
        individual_results[1] = true_positive_rate / const.TEST_REPEAT
        individual_results[2] = true_negative_rate / const.TEST_REPEAT
        individual_results[3] = false_positive_rate / const.TEST_REPEAT
        individual_results[4] = false_negative_rate / const.TEST_REPEAT
        individual_results[5] = true_positive_rate_cutoff / const.TEST_REPEAT
        individual_results[6] = true_negative_rate_cutoff / const.TEST_REPEAT
        individual_results[7] = false_positive_rate_cutoff / const.TEST_REPEAT
        individual_results[8] = false_negative_rate_cutoff / const.TEST_REPEAT
        individual_results[9] = unclassified_cutoff / const.TEST_REPEAT
        sorted_keys = sorted(parameter_dict)
        values = [parameter_dict.get(k) for k in sorted_keys if k in parameter_dict] + [data_balancer.__name__ if data_balancer is not None else "None"]
        results_recorder.record_results(values + individual_results)


if __name__ == "__main__":
    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")

            # Preprocess data set
            input_defaulter_set = apply_preprocessing(input_defaulter_set)

            logical_cpu_count = multiprocessing.cpu_count()

            for i in range(len(ctp.generic_classifier_parameter_arr)):
                if cfr.classifiers[i]['status'] is True and cfr.classifiers[i] is not None:
                    manager = Manager()
                    result_recorder = ClassifierResultRecorder(result_arr=manager.list())

                    # Execute enabled classifiers
                    parameter_grid = ParameterGrid(ctp.generic_classifier_parameter_arr[i])
                    Parallel(n_jobs=logical_cpu_count)(
                        delayed(execute_loop)(cfr.classifiers[i], parameter_grid[z], input_defaulter_set, result_recorder, z, len(parameter_grid)) for z in
                        range(len(parameter_grid)))

                    print(result_recorder.results)
                    if const.RECORD_RESULTS is True:
                        result_recorder.save_results_to_file(sorted(parameter_grid[0]) + ["Data balancer"], prepend_name_description=data_set["data_set_description"] + "_" +cfr.classifiers[i]['classifier_description'])
