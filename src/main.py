"""Primary script used to execute the defaulter prediction"""

import pandas as pd
from multiprocessing import Manager

from joblib import Parallel
from joblib import delayed

import config.classifiers as cfr
import visualisation as vis
from config import constants as const
from config import data_sets
from data_preprocessing import apply_preprocessing
from feature_selection.select_features import select_features
from generic_classifier import GenericClassifier
from result_recorder import ResultRecorder
from run_statistics import RunStatistics
from util import get_number_of_processes_to_use
import numpy as np


def execute_classifier_run(input_defaulter_set, classifier_parameters, data_balancer, data_set_description, classifier_dict, classifier_description, roc_plot, result_recorder):
    if classifier_dict["status"]:
        print("=== Executing {0} ===".format(classifier_description))
        test_stats = RunStatistics()
        for i in range(const.TEST_REPEAT):

            generic_classifier = GenericClassifier(classifier_dict["classifier"], classifier_parameters, data_balancer)
            result_dictionary = generic_classifier.k_fold_train_and_evaluate(input_defaulter_set, None)
            test_stats.append_run_result(result_dictionary, generic_classifier.ml_stats.roc_list)

        avg_results = test_stats.calculate_average_run_accuracy()
        roc_plot.append((test_stats.roc_list, classifier_description))
        result_recorder.record_results(avg_results, classifier_description)
        print("=== Completed {0} ===".format(classifier_description))


def main():
    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")
            # input_defaulter_set = pd.concat([input_defaulter_set[np.isnan(input_defaulter_set["Time to Default (Days)"])], input_defaulter_set[input_defaulter_set["Time to Default (Days)"] < 200]])

            input_defaulter_set = apply_preprocessing(input_defaulter_set, data_set["numeric_columns"], data_set["categorical_columns"], data_set["classification_label"], data_set["missing_values_strategy"], create_dummy_variables=False)

            # Apply feature selection
            input_defaulter_set, numeric_columns, categorical_columns = select_features(input_defaulter_set, data_set["numeric_columns"], data_set["categorical_columns"], data_set["classification_label"], data_set["data_set_classifier_parameters"], selection_strategy=data_set["feature_selection_strategy"])

            # Preprocess data set
            input_defaulter_set = apply_preprocessing(input_defaulter_set, numeric_columns, categorical_columns, data_set["classification_label"], data_set["missing_values_strategy"], create_dummy_variables=True)
            manager = Manager()
            result_recorder = ResultRecorder(result_arr=manager.list())

            roc_plot = manager.list()

            cpu_count = get_number_of_processes_to_use()
            # Execute enabled classifiers
            Parallel(n_jobs=cpu_count)(delayed(execute_classifier_run)(input_defaulter_set, data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["classifier_parameters"], data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["data_balancer"], data_set["data_set_description"], classifier_dict, classifier_description, roc_plot, result_recorder) for classifier_description, classifier_dict in cfr.classifiers.iteritems())

            roc_plot = sorted(roc_plot, key=lambda tup: tup[1])
            result_recorder.results = sorted(result_recorder.results, key=lambda tup: tup[1])
            for (result_arr, classifier_description) in result_recorder.results:
                print("\n=== {0} ===".format(classifier_description))
                print("Matthews correlation coefficient: {0}".format(result_arr[0]))
                print("Cohen Kappa score: {0}".format(result_arr[1]))
                print("Average true rate: {0}".format(result_arr[2]))
                print("Average true positive rate: {0}".format(result_arr[3]))
                print("Average true negative rate: {0}".format(result_arr[4]))
                print("Average false positive rate: {0}".format(result_arr[5]))
                print("Average false negative rate: {0}".format(result_arr[6]))

            if const.RECORD_RESULTS:
                vis.plot_mean_roc_curve_of_classifiers(roc_plot, data_set["data_set_description"])
                result_recorder.save_results_to_file(data_set["data_set_description"])


if __name__ == "__main__":
    # Run main
    main()
