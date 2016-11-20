"""Calculates optimal parameters for each classifier by searching a grid of testing parameters defined in config/classifier_tester_parameters. The search is parallelised to improve execution time."""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util import get_number_of_processes_to_use
from constants import DATA_BALANCER_STR
import config.classifier_data_balancer_only_tester_parameters as ctp
import config.classifiers as cfr
from classifier_result_recorder import ClassifierResultRecorder
import constants as const
from config import data_sets
from generic_classifier import GenericClassifier
from run_statistics import RunStatistics

# Data balancers to test
data_balancers = [None, ClusterCentroids, EditedNearestNeighbours, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule,
                  OneSidedSelection, RandomUnderSampler, TomekLinks, ADASYN, RandomOverSampler, SMOTE, SMOTEENN, SMOTETomek]


def execute_loop(classifier_description, classifier_dict, parameter_dict, sorted_keys, defaulter_set_arr, results_recorder, z, parameter_grid_len, requires_random_state, data_set_description,
                 numerical_columns, categorical_columns, binary_columns, classification_label, missing_value_strategy):
    """Executes const.TEST_REPEAT runs of stratified k-fold validation using input classifier with input parameters"""
    if z % 5 == 0:
        # Display progress to terminal
        print("==== {0} - {1} - {2}% ====".format(data_set_description, classifier_description, format((float(z) / parameter_grid_len) * 100, '.2f')))

    for data_balancer in data_balancers:
        test_stats = RunStatistics()
        success = True
        for x in range(const.TEST_REPEAT):
            try:
                # Create the generic classifier object and execute stratified k_fold cross validation

                # Workaround for CLC classifier which crashes when given the folds using these set random seeds (0-const.TEST_REPEAT) on Lima TB dataset
                if requires_random_state:
                    generic_classifier = GenericClassifier(classifier_dict['classifier'], parameter_dict, data_balancer, None)
                else:
                    print(classifier_dict['classifier'], parameter_dict, data_balancer)
                    generic_classifier = GenericClassifier(classifier_dict['classifier'], parameter_dict, data_balancer, x)

                result_dictionary = generic_classifier.k_fold_train_and_evaluate(defaulter_set_arr.copy(), numerical_columns=numerical_columns, categorical_columns=categorical_columns,
                                                                                 binary_columns=binary_columns, classification_label=classification_label,
                                                                                 missing_value_strategy=missing_value_strategy, apply_preprocessing=True)
                test_stats.append_run_result(result_dictionary, generic_classifier.ml_stats.roc_list)
            except Exception as e:
                success = False
                print("INFO: parameter caused classifier to raise exception - {1} - {0}".format(e, data_balancer))
                break

        if success:
            # Calculate average results across the runs and record it
            avg_results = test_stats.calculate_average_run_accuracy()
            avg_results = [avg_results[0], avg_results[1], avg_results[2], avg_results[15], avg_results[28], avg_results[3], avg_results[4], avg_results[5], avg_results[6],
                           avg_results[20], avg_results[21], avg_results[13], avg_results[22], avg_results[29], avg_results[24], avg_results[25]]
            values = [parameter_dict.get(k) if k in parameter_dict else "value_left_unset" for k in sorted_keys] + [data_balancer.__name__ if data_balancer is not None else "None"]
            results_recorder.record_results(values + avg_results)


def execute_parameter_test():
    """Runs the classifier parameter testing process"""
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
            input_defaulter_set = input_defaulter_set.dropna(axis=0)
            # Reset index to prevent issues further in the pipeline
            input_defaulter_set = input_defaulter_set.reset_index(drop=True)

            # Sets number of processes to use for parallelised grid search
            try:
                cpu_count = int(sys.argv[1])
            except IndexError:
                cpu_count = get_number_of_processes_to_use()


            for classifier_description, classifier_dict in cfr.classifiers.iteritems():
                parameter_dict = ctp.generic_classifier_parameter_dict[classifier_description]
                if classifier_dict['status'] and parameter_dict is not None:
                    # Generated a list that supports concurrent appends and use
                    manager = Manager()
                    result_recorder = ClassifierResultRecorder(result_arr=manager.list())

                    # Execute enabled classifiers
                    parameter_grid = list(ParameterGrid(parameter_dict["parameters"]))

                    # Adds default parameters to grid as well
                    if {} not in parameter_grid and "SVM" not in classifier_description and "Clustering-Launched Classification" not in classifier_description:
                        parameter_grid.append({})
                    elif "SVM" in classifier_description and {"kernel": parameter_grid[0]["kernel"]} not in parameter_grid:
                        parameter_grid.append({"kernel": parameter_grid[0]["kernel"]})

                    Parallel(n_jobs=cpu_count)(
                        delayed(execute_loop)(classifier_description, classifier_dict, parameter_grid[z], sorted(parameter_grid[0]), input_defaulter_set, result_recorder, z, len(parameter_grid),
                                              parameter_dict["requires_random_state"], data_set["data_set_description"], data_set["numeric_columns"], data_set["categorical_columns"],
                                              data_set["binary_columns"], data_set["classification_label"],
                                              data_set["missing_values_strategy"]) for z in
                        range(len(parameter_grid)))

                    # Record the results to file
                    if const.RECORD_RESULTS is True:
                        result_recorder.save_results_to_file(sorted(parameter_grid[0]) + [DATA_BALANCER_STR],
                                                             prepend_name_description=data_set["data_set_description"] + "_" + classifier_description)


if __name__ == "__main__":
    execute_parameter_test()
