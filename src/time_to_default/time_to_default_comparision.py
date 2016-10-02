"""Primary script used to execute the defaulter prediction"""

import pandas as pd
from multiprocessing import Manager

from joblib import Parallel
from joblib import delayed
import sys
import os

from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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


def execute_classifier_run(defaulters_in_range, defaulters_out_of_range, non_defaulters, numeric_columns, categorical_columns, classification_label, missing_values_strategy, classifier_parameters, data_balancer, classifier_dict, classifier_description, roc_plot, result_recorder):
    if classifier_dict["status"]:
        print("=== Executing {0} ===".format(classifier_description))
        test_stats = RunStatistics()
        for i in range(500):
            X_train, X_test, y_train, y_test = train_test_split(non_defaulters[numeric_columns + categorical_columns], non_defaulters[classification_label], test_size=0.5, random_state=i)
            X_train = pd.concat([X_train, defaulters_in_range[numeric_columns + categorical_columns]])
            y_train = pd.concat([y_train, defaulters_in_range[classification_label]])
            X_test = pd.concat([X_test, defaulters_out_of_range[numeric_columns + categorical_columns]])
            y_test = pd.concat([y_test, defaulters_out_of_range[classification_label]])

            # Preprocess data set
            training = apply_preprocessing(pd.concat([X_train, y_train], axis=1), numeric_columns, categorical_columns, classification_label, missing_values_strategy, create_dummy_variables=True)
            testing = apply_preprocessing(pd.concat([X_test, y_test], axis=1), numeric_columns, categorical_columns, classification_label, missing_values_strategy, create_dummy_variables=True)

            X_train = training.iloc[:, :-1].as_matrix()
            y_train = training.iloc[:, -1:].as_matrix().flatten()
            X_test = testing.iloc[:, :-1].as_matrix()
            y_test = testing.iloc[:, -1:].as_matrix().flatten()
            generic_classifier = GenericClassifier(classifier_dict["classifier"], classifier_parameters, data_balancer)
            result_dictionary = generic_classifier.train_and_evaluate(X_train, y_train, X_test, y_test, i)
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

            data_processed = apply_preprocessing(input_defaulter_set, data_set["numeric_columns"], data_set["categorical_columns"], data_set["classification_label"], data_set["missing_values_strategy"], create_dummy_variables=False)

            # Apply feature selection
            data_processed, numeric_columns, categorical_columns = select_features(data_processed, data_set["numeric_columns"], data_set["categorical_columns"], data_set["classification_label"], data_set["data_set_classifier_parameters"], selection_strategy=data_set["feature_selection_strategy"])

            time_range_results = []
            time_ranges = [(0, 30), (0, 60), (0, 100), (0, 200), (50, 100), (50, 150), (100, 200), (200, 1000), (300, 1000)]

            non_defaulters = input_defaulter_set[np.isnan(input_defaulter_set["Time to Default (Days)"])]

            cpu_count = get_number_of_processes_to_use()

            for time_range in time_ranges:
                print("Time range {0} - {1}".format(time_range[0], time_range[1]))
                defaulters_in_range = input_defaulter_set[(input_defaulter_set["Time to Default (Days)"] >= time_range[0]) & (input_defaulter_set["Time to Default (Days)"] <= time_range[1])]
                defaulters_out_of_range = input_defaulter_set[(input_defaulter_set["Time to Default (Days)"] < time_range[0]) | (input_defaulter_set["Time to Default (Days)"] > time_range[1])]
                print("defaulters_in_range", len(defaulters_in_range))
                print("defaulters_out_of_range", len(defaulters_out_of_range))

                manager = Manager()
                result_recorder = ResultRecorder(result_arr=manager.list())

                roc_plot = manager.list()

                # Execute enabled classifiers
                Parallel(n_jobs=cpu_count)(delayed(execute_classifier_run)(defaulters_in_range, defaulters_out_of_range, non_defaulters, numeric_columns, categorical_columns, data_set["classification_label"], data_set["missing_values_strategy"], data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["classifier_parameters"], data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["data_balancer"], classifier_dict, classifier_description, roc_plot, result_recorder) for classifier_description, classifier_dict in cfr.classifiers.iteritems())

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

                time_range_results.append(("{0} - {1}".format(time_range[0], time_range[1]), result_recorder.results))

            vis.plot_time_to_default_results(time_range_results)


if __name__ == "__main__":
    # Run main
    main()
