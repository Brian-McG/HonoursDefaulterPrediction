"""Primary script used to execute the classfiier comparison"""
import os
import subprocess
import sys
from multiprocessing import Manager
from random import Random

import pandas as pd
from joblib import Parallel
from joblib import delayed

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config.classifiers as cfr
import constants as const
import visualisation as vis
from config import data_sets
from generic_classifier import GenericClassifier
from result_recorder import ResultRecorder
from run_statistics import RunStatistics
from util import get_number_of_processes_to_use

const.TEST_REPEAT = 15


def execute_classifier_run(input_defaulter_set, classifier_parameters, data_balancer, random_values, classifier_dict, classifier_description, roc_plot, result_recorder, numeric_columns,
                           categorical_columns, binary_columns, classification_label, missing_value_strategy):
    """Executes const.TEST_REPEAT runs of stratified k-fold validation using input classifier with input parameters"""
    if classifier_dict["status"]:
        print("=== Executing {0} - {1} ===".format(classifier_description, classifier_parameters))
        test_stats = RunStatistics()
        for i in range(const.TEST_REPEAT):
            generic_classifier = GenericClassifier(classifier_dict["classifier"], classifier_parameters, data_balancer, random_values[i])
            result_dictionary = generic_classifier.k_fold_train_and_evaluate(input_defaulter_set.copy(), numerical_columns=numeric_columns, categorical_columns=categorical_columns,
                                                                             binary_columns=binary_columns, classification_label=classification_label, missing_value_strategy=missing_value_strategy,
                                                                             apply_preprocessing=True)
            test_stats.append_run_result(result_dictionary, generic_classifier.ml_stats.roc_list)

        avg_results = test_stats.calculate_average_run_accuracy()
        roc_plot.append((test_stats.roc_list, classifier_description))
        result_recorder.record_results(avg_results, classifier_description)
        print("=== Completed {0} ===".format(classifier_description))


def main(random_value_arr):
    """Executes the classification process"""
    data_set_arr = []

    classifier_active_count = 0
    for classifier_description, classifier_dict in cfr.classifiers.iteritems():
        if classifier_dict["status"]:
            classifier_active_count += 1

    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")
            # Remove duplicates
            if data_set["duplicate_removal_column"] is not None:
                input_defaulter_set.drop_duplicates(data_set["duplicate_removal_column"], inplace=True)
            # Only retain the important fields
            input_defaulter_set = input_defaulter_set[
                data_set["numeric_columns"] + data_set["categorical_columns"] + [name for name, _, _ in data_set["binary_columns"]] + data_set["classification_label"]]
            # Remove entries with missing inputs
            input_defaulter_set.dropna(axis=0, inplace=True)
            # Reset index to prevent issues further in the pipeline
            input_defaulter_set.reset_index(drop=True, inplace=True)

            manager = Manager()
            result_recorder = ResultRecorder(result_arr=manager.list())

            roc_plot = manager.list()
            if len(random_value_arr) == 0:
                random_value_arr = []
                random = Random()
                for i in range(const.TEST_REPEAT):
                    while True:
                        random_value = random.randint(const.RANDOM_RANGE[0], const.RANDOM_RANGE[1])
                        if random_value not in random_value_arr:
                            random_value_arr.append(random_value)
                            break
            elif len(random_value_arr) < const.TEST_REPEAT:
                raise RuntimeError("Random value does not match length of test repeat {0} != {1}".format(len(random_value_arr), const.TEST_REPEAT))

            cpu_count = get_number_of_processes_to_use()
            # Execute enabled classifiers
            Parallel(n_jobs=cpu_count)(
                delayed(execute_classifier_run)(input_defaulter_set, data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["classifier_parameters"],
                                                data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["data_balancer"], random_value_arr, classifier_dict,
                                                classifier_description, roc_plot, result_recorder, data_set["numeric_columns"], data_set["categorical_columns"], data_set["binary_columns"],
                                                data_set["classification_label"], data_set["missing_values_strategy"]) for classifier_description, classifier_dict in cfr.classifiers.iteritems())

            roc_plot = sorted(roc_plot, key=lambda tup: tup[1])
            result_recorder.results = sorted(result_recorder.results, key=lambda tup: tup[1])

            # Output results to terminal
            for (result_arr, classifier_description) in result_recorder.results:
                print("\n=== {0} ===".format(classifier_description))
                print("Matthews correlation coefficient: {0} ({1})".format(result_arr[0], result_arr[20]))
                print("Brier Score: {0} ({1})".format(result_arr[1], result_arr[21]))
                print("H-measure: {0} ({1})".format(result_arr[28], result_arr[29]))
                print("AUC: {0} ({1})".format(result_arr[15], result_arr[22]))
                print("Balanced Accuracy: {0} ({1})".format(result_arr[2], result_arr[13]))
                print("Average true positive rate: {0} ({1})".format(result_arr[3], result_arr[24]))
                print("Average true negative rate: {0} ({1})".format(result_arr[4], result_arr[25]))
                print("Average false positive rate: {0} ({1})".format(result_arr[5], result_arr[26]))
                print("Average false negative rate: {0} ({1})".format(result_arr[6], result_arr[27]))

            # Save results and execute statistical tests
            data_set_arr.append((data_set["data_set_description"], result_recorder.results))
            metrics, file_paths = ResultRecorder.save_results_for_multi_dataset(((data_set["data_set_description"], result_recorder.results),), dataset=data_set["data_set_description"])
            script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../external_significance_testing/significance_testing.R")
            if classifier_active_count > 1:
                for i in range(len(metrics)):
                    input_arr = ["Rscript", script_path, os.path.abspath(file_paths[i]), data_set["data_set_description"] + "_" + metrics[i]]
                    print(" ".join(input_arr))
                    if sys.platform == 'win32':
                        subprocess.check_call(input_arr, shell=True)
                    else:
                        subprocess.check_call(input_arr, shell=False)
            if const.RECORD_RESULTS:
                vis.plot_mean_roc_curve_of_classifiers(roc_plot, data_set["data_set_description"])
                if cpu_count == 1:
                    result_recorder.save_results_to_file(random_value_arr, data_set["data_set_description"], display_time_to_fit_results=True)
                else:
                    result_recorder.save_results_to_file(random_value_arr, data_set["data_set_description"], display_time_to_fit_results=False)

    vis.visualise_dataset_classifier_results(data_set_arr)
    metrics, file_paths = ResultRecorder.save_results_for_multi_dataset(data_set_arr)
    script_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../external_significance_testing/significance_testing.R")

    if classifier_active_count > 1:
        for i in range(len(metrics)):
            input_arr = ["Rscript", script_path, os.path.abspath(file_paths[i]), metrics[i]]
            print(" ".join(input_arr))
            if sys.platform == 'win32':
                subprocess.check_call(input_arr, shell=True)
            else:
                subprocess.check_call(input_arr, shell=False)


if __name__ == "__main__":
    # Run main
    random_values = []
    for p in range(1, len(sys.argv)):
        random_values.append(long(sys.argv[p]))
    main(random_values)
