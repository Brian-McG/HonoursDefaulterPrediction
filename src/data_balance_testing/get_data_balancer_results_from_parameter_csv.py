"""Uses the optimal paramters for each classifier to test data balancers for each dataset."""
import os
import sys
from multiprocessing import Manager
from random import Random

from joblib import Parallel
from joblib import delayed
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import data_sets
from data_balance_testing.data_balancer_result_recorder import DataBalancerResultRecorder
from generic_classifier import GenericClassifier
from run_statistics import RunStatistics
from util import get_number_of_processes_to_use
import pandas as pd
import classifier_parameter_testing.classifier_parameter_tester as classifier_parameter_tester
from constants import DATA_BALANCER_STR
from visualisation import plot_balancer_results_per_classifier

import numpy as np
import constants as const
import config.classifiers as cfr
import visualisation as vis
from imblearn.under_sampling.cluster_centroids import ClusterCentroids
from imblearn.under_sampling.edited_nearest_neighbours import EditedNearestNeighbours
from imblearn.under_sampling.condensed_nearest_neighbour import CondensedNearestNeighbour
from imblearn.under_sampling.instance_hardness_threshold import InstanceHardnessThreshold
from imblearn.under_sampling.nearmiss import NearMiss
from imblearn.under_sampling.neighbourhood_cleaning_rule import NeighbourhoodCleaningRule
from imblearn.under_sampling.one_sided_selection import OneSidedSelection
from imblearn.under_sampling.tomek_links import TomekLinks
from imblearn.under_sampling.random_under_sampler import RandomUnderSampler
from imblearn.over_sampling.smote import SMOTE
from imblearn.over_sampling.adasyn import ADASYN
from imblearn.over_sampling.random_over_sampler import RandomOverSampler
from imblearn.combine.smote_enn import SMOTEENN
from imblearn.combine.smote_tomek import SMOTETomek
import config.balancer_comparision_input as bci

# Number of runs
const.TEST_REPEAT = 10


def override_parameters(parameter_results):
    """Overrides default classifier parameters with """
    data_balancer_arr = {}
    for (classifier_name, classifier_path) in parameter_results:
        data_balancer_arr[classifier_name] = []
        parameter_results = pd.DataFrame.from_csv(classifier_path, index_col=False, encoding="UTF-8")
        data_balancers = classifier_parameter_tester.data_balancers
        for data_balancer in data_balancers:
            data_balance_df = parameter_results.loc[parameter_results[DATA_BALANCER_STR] == (data_balancer.__name__ if data_balancer is not None else "None")]
            parameter_headers = []
            for parameter_header in data_balance_df.columns.values.tolist():
                parameter_headers.append(str(parameter_header))
                if parameter_header == DATA_BALANCER_STR:
                    break
            # Gets best parameter set for the data balancer
            if "Average true rate" in data_balance_df:
                best_parameters = data_balance_df.loc[data_balance_df["Average true rate"].argmax()]
            elif "Balanced Accuracy" in data_balance_df:
                best_parameters = data_balance_df.loc[data_balance_df["Balanced Accuracy"].argmax()]
            parameter_dict = {}
            data_balancer = None

            # Converts the textual parameters to the correct type
            for parameter_header in parameter_headers:
                if parameter_header != DATA_BALANCER_STR:
                    if type(best_parameters[parameter_header]) == unicode:
                        if str(best_parameters[parameter_header]) != "value_left_unset":
                            try:
                                parameter_dict[parameter_header] = eval(best_parameters[parameter_header])
                            except NameError:
                                parameter_dict[parameter_header] = str(best_parameters[parameter_header])
                    else:
                        if np.isnan(best_parameters[parameter_header]):
                            parameter_dict[parameter_header] = None
                        else:
                            parameter_dict[parameter_header] = best_parameters[parameter_header]
                        if type(best_parameters[parameter_header]) == np.float64 and best_parameters[parameter_header].is_integer():
                            parameter_dict[parameter_header] = np.int64(best_parameters[parameter_header])
                else:
                    data_balancer = eval(best_parameters[parameter_header])
            data_balancer_arr[classifier_name].append((data_balancer, parameter_dict))
    return data_balancer_arr


def execute_classifier_run(data_balancer_results, random_values, input_defaulter_set, numerical_columns, categorical_columns, binary_columns, classification_label, missing_value_strategy,
                           classifier_arr, classifier_description, roc_plot):
    """Executes each classifier with each data balancer and their respective parameter_dict with const.TEST_REPEAT runs of stratified k-fold validation"""
    if cfr.classifiers[classifier_description]["status"]:
        result_arr = []
        for (data_balancer, parameter_dict) in classifier_arr:
            if "SVM" in classifier_description:
                parameter_dict["probability"] = True
            print("=== Executing {0} - {1} ===".format(classifier_description, data_balancer.__name__ if data_balancer is not None else "None"))
            test_stats = RunStatistics()
            for i in range(const.TEST_REPEAT):
                generic_classifier = GenericClassifier(cfr.classifiers[classifier_description]["classifier"], parameter_dict, data_balancer, random_values[i])
                result_dictionary = generic_classifier.k_fold_train_and_evaluate(input_defaulter_set.copy(), numerical_columns=numerical_columns, categorical_columns=categorical_columns,
                                                                                 binary_columns=binary_columns, classification_label=classification_label,
                                                                                 missing_value_strategy=missing_value_strategy, apply_preprocessing=True)
                test_stats.append_run_result(result_dictionary, generic_classifier.ml_stats.roc_list)

            avg_results = test_stats.calculate_average_run_accuracy()
            roc_plot.append((test_stats.roc_list, classifier_description))
            result_arr.append((data_balancer.__name__ if data_balancer is not None else "None", avg_results))
            print("=== Completed {0} - {1} ===".format(classifier_description, data_balancer.__name__ if data_balancer is not None else "None"))
        data_balancer_results.append((classifier_description, result_arr))


def main(classifier_dict):
    data_set_results = []
    for data_set in data_sets.data_set_arr:
        if data_set["status"] and data_set["data_set_description"] in classifier_dict:
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

            result_recorder = DataBalancerResultRecorder()
            cpu_count = get_number_of_processes_to_use()

            # Generate the random seeds to use
            random_values = []
            random = Random()
            for i in range(const.TEST_REPEAT):
                while True:
                    random_value = random.randint(const.RANDOM_RANGE[0], const.RANDOM_RANGE[1])
                    if random_value not in random_values:
                        random_values.append(random_value)
                        break
            classifier_parameters = override_parameters(classifier_dict[data_set["data_set_description"]])
            manager = Manager()

            roc_plot = manager.list()
            data_balancer_results = manager.list()

            # Execute enabled classifiers
            Parallel(n_jobs=cpu_count)(
                delayed(execute_classifier_run)(data_balancer_results, random_values, input_defaulter_set, data_set["numeric_columns"], data_set["categorical_columns"], data_set["binary_columns"],
                                                data_set["classification_label"], data_set["missing_values_strategy"], classifier_dict, classifier_description, roc_plot) for
                classifier_description, classifier_dict in classifier_parameters.iteritems())

            data_balancer_results = sorted(data_balancer_results, key=lambda tup: tup[0])

            for (classifier_name, classifier_arr) in data_balancer_results:
                for (data_balancer_name, result_arr) in classifier_arr:
                    result_recorder.record_results(result_arr, classifier_name, data_balancer_name)

            result_recorder.save_results_to_file(random_values, "data_balancer")
            plot_balancer_results_per_classifier(data_balancer_results, (2, "Balanced Accuracy"))
            plot_balancer_results_per_classifier(data_balancer_results, (3, "Average true positive rate"))
            plot_balancer_results_per_classifier(data_balancer_results, (4, "Average true negative rate"))
            data_set_results.append((data_set["data_set_description"], data_balancer_results))
            vis.visualise_dataset_balancer_results([(data_set["data_set_description"], data_balancer_results)])
    vis.visualise_dataset_balancer_results_multi_dataset(data_set_results)
    DataBalancerResultRecorder.save_results_for_multi_dataset(data_set_results)


if __name__ == "__main__":
    classifiers = []
    classifier_arr = []

    data_set_arr = [bci.parameters[0]]
    for i in range(1, len(bci.parameters), 2):
        if bci.parameters[i] == "next_dataset":
            data_set_arr.append(bci.parameters[i + 1])
            classifier_arr.append(classifiers)
            classifiers = []
            continue
        classifiers.append((bci.parameters[i], bci.parameters[i + 1]))
    classifier_arr.append(classifiers)

    classifier_dict = dict()
    for i in range(len(data_set_arr)):
        classifier_dict[data_set_arr[i]] = classifier_arr[i]
    main(classifier_dict)
