"""Compares how training on different default ranges generalises to the rest of the data"""

import os
import sys
from multiprocessing import Manager
from random import Random

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn.model_selection import ShuffleSplit

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import config.classifiers as cfr
import visualisation as vis
import constants as const
from config import data_sets
from data_preprocessing import apply_preprocessing_to_train_test_dataset
from generic_classifier import GenericClassifier
from run_statistics import RunStatistics
from util import get_number_of_processes_to_use
from time_to_default.time_to_default_result_recorder import TimeToDefaultResultRecorder

const.TEST_REPEAT = 100


def execute_classifier_run(time_range, random_values, defaulters_in_range, defaulters_out_of_range, non_defaulters, numeric_columns, categorical_columns, binary_columns, classification_label,
                           missing_values_strategy, classifier_parameters, data_balancer, classifier_dict, classifier_description, roc_plot, result_recorder):
    if classifier_dict["status"]:
        print("=== Executing {0} ===".format(classifier_description))
        test_stats = RunStatistics()
        for i in range(const.TEST_REPEAT):
            split = ShuffleSplit(n_splits=1, test_size=0.5, random_state=random_values[i])
            for train_index, test_index in split.split(non_defaulters):
                train = pd.concat([non_defaulters.iloc[train_index], defaulters_in_range])
                test = pd.concat([non_defaulters.iloc[test_index], defaulters_out_of_range])

                train_index = train.index.values
                test_index = test.index.values
                combined = pd.concat([train, test])

                # Preprocess data set
                train, test = apply_preprocessing_to_train_test_dataset(combined, train_index, test_index, numeric_columns, categorical_columns, binary_columns, classification_label,
                                                                        missing_values_strategy, create_dummy_variables=True)

                X_train = train.iloc[:, :-1].as_matrix()
                y_train = train.iloc[:, -1:].as_matrix().flatten()
                X_test = test.iloc[:, :-1].as_matrix()
                y_test = test.iloc[:, -1:].as_matrix().flatten()
                generic_classifier = GenericClassifier(classifier_dict["classifier"], classifier_parameters, data_balancer, random_values[i])
                result_dictionary = generic_classifier.train_and_evaluate(X_train, y_train, X_test, y_test)
                test_stats.append_run_result(result_dictionary, generic_classifier.ml_stats.roc_list)

        avg_results = test_stats.calculate_average_run_accuracy()
        roc_plot.append((test_stats.roc_list, classifier_description))
        result_recorder.record_results(avg_results, classifier_description, time_range)
        print("=== Completed {0} ===".format(classifier_description))


def main():
    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")
            input_defaulter_set = input_defaulter_set.dropna(axis=0, subset=[
                data_set["numeric_columns"] + data_set["categorical_columns"] + [name for name, _, _ in data_set["binary_columns"]] + data_set["classification_label"]])
            input_defaulter_set = input_defaulter_set.reset_index(drop=True)

            time_range_results = []
            time_ranges = [(0, 30), (0, 60), (0, 100), (0, 200), (0, 300), (50, 150), (100, 200), (200, 1000), (300, 1000)]

            non_defaulters = input_defaulter_set[np.isnan(input_defaulter_set["Time to Default (Days)"])]
            non_defaulters = non_defaulters[data_set["numeric_columns"] + data_set["categorical_columns"] + [name for name, _, _ in data_set["binary_columns"]] + data_set["classification_label"]]

            random_values = []
            random = Random()
            for i in range(const.TEST_REPEAT):
                while True:
                    random_value = random.randint(const.RANDOM_RANGE[0], const.RANDOM_RANGE[1])
                    if random_value not in random_values:
                        random_values.append(random_value)
                        break

            cpu_count = get_number_of_processes_to_use()

            manager = Manager()
            result_recorder = TimeToDefaultResultRecorder()
            for time_range in time_ranges:
                print("Time range {0} - {1}".format(time_range[0], time_range[1]))
                defaulters_in_range = input_defaulter_set[(input_defaulter_set["Time to Default (Days)"] >= time_range[0]) & (input_defaulter_set["Time to Default (Days)"] <= time_range[1])]
                defaulters_in_range = defaulters_in_range[
                    data_set["numeric_columns"] + data_set["categorical_columns"] + [name for name, _, _ in data_set["binary_columns"]] + data_set["classification_label"]]
                defaulters_out_of_range = input_defaulter_set[(input_defaulter_set["Time to Default (Days)"] < time_range[0]) | (input_defaulter_set["Time to Default (Days)"] > time_range[1])]
                defaulters_out_of_range = defaulters_out_of_range[
                    data_set["numeric_columns"] + data_set["categorical_columns"] + [name for name, _, _ in data_set["binary_columns"]] + data_set["classification_label"]]
                print("defaulters_in_range", len(defaulters_in_range))
                print("defaulters_out_of_range", len(defaulters_out_of_range))

                roc_plot = manager.list()
                run_results = TimeToDefaultResultRecorder(result_arr=manager.list())

                # Execute enabled classifiers
                Parallel(n_jobs=cpu_count)(delayed(execute_classifier_run)(time_range, random_values, defaulters_in_range, defaulters_out_of_range, non_defaulters, data_set["numeric_columns"],
                                                                           data_set["categorical_columns"], data_set["binary_columns"], data_set["classification_label"],
                                                                           data_set["missing_values_strategy"],
                                                                           data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["classifier_parameters"],
                                                                           data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["data_balancer"], classifier_dict,
                                                                           classifier_description, roc_plot, run_results) for classifier_description, classifier_dict in cfr.classifiers.iteritems())

                result_recorder.results = sorted(result_recorder.results, key=lambda tup: tup[1])
                time_range_results.append(("{0} - {1}".format(time_range[0], time_range[1]), run_results.results))

                for (avg_results, classifier_description, time_range) in run_results.results:
                    result_recorder.record_results(avg_results, classifier_description, time_range)

            vis.plot_time_to_default_results(time_range_results)
            result_recorder.save_results_to_file(random_values, "time_to_default")


if __name__ == "__main__":
    # Run main
    main()
