"""Primary script used to execute the defaulter prediction"""

import os
import sys
from multiprocessing import Manager
from random import Random

import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from parameter_comparision_recorder import ParameterComparisionResultRecorder
from data_preprocessing import apply_preprocessing_to_train_test_dataset
from generic_classifier import GenericClassifier
from run_statistics import RunStatistics
from util import get_number_of_processes_to_use
from config import data_sets
import constants as const
import visualisation as vis
import config.classifiers as cfr

const.TEST_REPEAT = 10


def execute_classifier_run(random_values, input_defaulter_set, numeric_columns, categorical_columns, binary_columns, classification_label, classifier_parameters, data_balancer, parameter_description, classifier_dict, classifier_description, roc_plot, result_recorder, missing_value_strategy, parameter_index):
    if classifier_dict["status"]:
        print("=== Executing {0} ===".format(classifier_description))
        test_stats = RunStatistics()
        if parameter_index == 0:
            data_balancer = None
        for i in range(const.TEST_REPEAT):
            generic_classifier = GenericClassifier(classifier_dict["classifier"], classifier_parameters, data_balancer, random_values[i])
            kf = StratifiedKFold(n_splits=const.NUMBER_OF_FOLDS, shuffle=True, random_state=generic_classifier.k_fold_state)
            result_dictionary = None
            defaulter_set_copy = input_defaulter_set.copy()
            for train, test in kf.split(defaulter_set_copy.iloc[:, :-1], defaulter_set_copy.iloc[:, -1:].as_matrix().flatten()):
                train_df, test_df = apply_preprocessing_to_train_test_dataset(defaulter_set_copy, train, test, numeric_columns, categorical_columns, binary_columns, classification_label,
                                                                              missing_value_strategy, create_dummy_variables=True)
                test_df = test_df[train_df.columns]

                result_dictionary = generic_classifier.train_and_evaluate(train_df.iloc[:, :-1].as_matrix(), train_df.iloc[:, -1:].as_matrix().flatten(), test_df.iloc[:, :-1].as_matrix(), test_df.iloc[:, -1:].as_matrix().flatten())

            test_stats.append_run_result(result_dictionary, generic_classifier.ml_stats.roc_list)

        avg_results = test_stats.calculate_average_run_accuracy()
        roc_plot.append((test_stats.roc_list, classifier_description))
        result_recorder.record_results(avg_results, classifier_description, parameter_description)
        print("=== Completed {0} ===".format(classifier_description))


def main(random_values):
    result_arr = []
    data_set_arr = []
    if len(random_values) == 0:
        random = Random()
        for i in range(const.TEST_REPEAT):
            while True:
                random_value = random.randint(const.RANDOM_RANGE[0], const.RANDOM_RANGE[1])
                if random_value not in random_values:
                    random_values.append(random_value)
                    break
    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")
            input_defaulter_set = input_defaulter_set[data_set["numeric_columns"] + data_set["categorical_columns"]  + [name for name, _, _ in data_set["binary_columns"]] + data_set["classification_label"]]
            input_defaulter_set = input_defaulter_set.dropna(axis=0)
            input_defaulter_set = input_defaulter_set.reset_index(drop=True)

            parameter_comparision_results = []

            result_recorder_after = ParameterComparisionResultRecorder()
            cpu_count = get_number_of_processes_to_use()

            parameter_description = ["Default without balancer", "Default with balancer", "Tuned"]
            parameter_sets = [data_set["data_set_data_balancer_parameters"], data_set["data_set_data_balancer_parameters"], data_set["data_set_classifier_parameters"]]
            for parameter_index in range(len(parameter_sets)):
                manager = Manager()
                parameter_comparision_result_recorder = ParameterComparisionResultRecorder(result_arr=manager.list())

                roc_plot = manager.list()

                # Execute enabled classifiers
                Parallel(n_jobs=cpu_count)(delayed(execute_classifier_run)(random_values, input_defaulter_set, data_set["numeric_columns"], data_set["categorical_columns"], data_set["binary_columns"], data_set["classification_label"],
                                                                           parameter_sets[parameter_index].classifier_parameters[classifier_description]["classifier_parameters"],
                                                                           parameter_sets[parameter_index].classifier_parameters[classifier_description]["data_balancer"], parameter_description[parameter_index], classifier_dict, classifier_description, roc_plot,
                                                                           parameter_comparision_result_recorder, data_set["missing_values_strategy"], parameter_index) for classifier_description, classifier_dict in cfr.classifiers.iteritems())

                parameter_comparision_result_recorder.results = sorted(parameter_comparision_result_recorder.results, key=lambda tup: tup[1])

                for (avg_results, classifier_description, feature_selection) in parameter_comparision_result_recorder.results:
                    result_recorder_after.record_results(avg_results, classifier_description, feature_selection)

                parameter_comparision_results.append((parameter_description[parameter_index], parameter_comparision_result_recorder.results, parameter_description[parameter_index]))
            result_arr.append(parameter_comparision_results)
            data_set_arr.append(data_set["data_set_description"])
            result_recorder_after.save_results_to_file(random_values, "Parameter tuning")
    vis.plot_percentage_difference_graph(result_arr, data_set_arr, x_label="Parameter tuning approach", name_suffix="", difference_from="using default parameters", figsize=(16, 5), legend_y=-0.62, label_rotation=0, y_label_pos=-0.3)
    ParameterComparisionResultRecorder.save_results_for_multi_dataset(result_arr)

if __name__ == "__main__":
    # Run main
    random_values = []
    for p in range(1, len(sys.argv)):
        random_values.append(int(sys.argv[p]))
    main(random_values)
