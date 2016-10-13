import os
import sys

from multiprocessing import Manager
from random import Random

from joblib import Parallel
from joblib import delayed

from config import data_sets
from data_balance_testing.data_balancer_result_recorder import DataBalancerResultRecorder
from data_preprocessing import apply_preprocessing
from generic_classifier import GenericClassifier
from result_recorder import ResultRecorder
from run_statistics import RunStatistics
from util import get_number_of_processes_to_use

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import classifier_parameter_testing.classifier_parameter_tester as classifier_parameter_tester
from config.constants import DATA_BALANCER_STR
from visualisation import plot_balancer_results_per_classifier
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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier


import numpy as np
import config.constants as const
import config.classifiers as cfr
import visualisation as vis

const.TEST_REPEAT = 1

def override_parameters(parameter_results):
    data_balancer_arr = {}
    for (classifier_name, classifier_path) in parameter_results:
        data_balancer_arr[classifier_name] = []
        parameter_results = pd.DataFrame.from_csv(classifier_path, index_col=None, encoding="UTF-8")
        data_balancers = classifier_parameter_tester.data_balancers
        for data_balancer in data_balancers:
            data_balance_df = parameter_results.loc[parameter_results[DATA_BALANCER_STR] == (data_balancer.__name__ if data_balancer is not None else "None")]
            parameter_headers = []
            for parameter_header in data_balance_df.columns.values.tolist():
                parameter_headers.append(str(parameter_header))
                if parameter_header == DATA_BALANCER_STR:
                    break
            best_parameters = data_balance_df.loc[data_balance_df["Average true rate"].argmax()]
            parameter_dict = {}
            data_balancer = None
            for parameter_header in parameter_headers:
                if parameter_header != DATA_BALANCER_STR:
                    if type(best_parameters[parameter_header]) == unicode:
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


    # plot_balancer_results_per_classifier(classifiers, "Average true rate")
    # plot_balancer_results_per_classifier(classifiers, "Average true positive rate")
    # plot_balancer_results_per_classifier(classifiers, "Average true negative rate")

def execute_classifier_run(data_balancer_results, random_values, input_defaulter_set, classifier_arr, classifier_description, roc_plot):
    result_arr = []
    for(data_balancer, parameter_dict) in classifier_arr:
        print("=== Executing {0} - {1} ===".format(classifier_description, data_balancer.__name__ if data_balancer is not None else "None"))
        test_stats = RunStatistics()
        for i in range(const.TEST_REPEAT):
            generic_classifier = GenericClassifier(cfr.classifiers[classifier_description]["classifier"], parameter_dict, data_balancer, random_values[i])
            result_dictionary = generic_classifier.k_fold_train_and_evaluate(input_defaulter_set)
            test_stats.append_run_result(result_dictionary, generic_classifier.ml_stats.roc_list)

        avg_results = test_stats.calculate_average_run_accuracy()
        roc_plot.append((test_stats.roc_list, classifier_description))
        result_arr.append((data_balancer.__name__ if data_balancer is not None else "None", avg_results))
        print("=== Completed {0} - {1} ===".format(classifier_description, data_balancer.__name__ if data_balancer is not None else "None"))
    data_balancer_results.append((classifier_description, result_arr))


def main(dataset, parameter_results):
    data_set_results = []
    for data_set in data_sets.data_set_arr:
        if data_set["data_set_description"] == dataset:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")

            input_defaulter_set = apply_preprocessing(input_defaulter_set, data_set["numeric_columns"], data_set["categorical_columns"], data_set["classification_label"], data_set["missing_values_strategy"], create_dummy_variables=True)

            manager = Manager()
            result_recorder = DataBalancerResultRecorder()
            cpu_count = get_number_of_processes_to_use()

            random_values = []
            random = Random()
            for i in range(const.TEST_REPEAT):
                while True:
                    random_value = random.randint(const.RANDOM_RANGE[0], const.RANDOM_RANGE[1])
                    if random_value not in random_values:
                        random_values.append(random_value)
                        break
            classifier_parameters = override_parameters(parameter_results)
            manager = Manager()

            roc_plot = manager.list()
            data_balancer_results = manager.list()

            # Execute enabled classifiers
            Parallel(n_jobs=cpu_count)(delayed(execute_classifier_run)(data_balancer_results, random_values, input_defaulter_set, classifier_dict, classifier_description, roc_plot) for classifier_description, classifier_dict in classifier_parameters.iteritems())

            data_balancer_results = sorted(data_balancer_results, key=lambda tup: tup[0])

            for (classifier_name, classifier_arr) in data_balancer_results:
                for(data_balancer_name, result_arr) in classifier_arr:
                    result_recorder.record_results(result_arr, classifier_name, data_balancer_name)

            result_recorder.save_results_to_file(random_values, "data_balancer")
            plot_balancer_results_per_classifier(data_balancer_results, (0, "Average Matthews correlation coefficient"))
            plot_balancer_results_per_classifier(data_balancer_results, (2, "Average Balanced classification rate"))
            plot_balancer_results_per_classifier(data_balancer_results, (3, "Average true positive rate"))
            plot_balancer_results_per_classifier(data_balancer_results, (4, "Average true negative rate"))
            data_set_results.append((data_set["data_set_description"], data_balancer_results))
    vis.visualise_dataset_balancer_results(data_set_results)
if __name__ == "__main__":
    if len(sys.argv) < 3 and len(sys.argv) % 2 == 0:
        print('Expected "cdn_perf.py <parameter_result_label> <parameter_results_path>"')
    else:
        classifier_arr = []
        data_set = sys.argv[1]
        for i in range(2, len(sys.argv), 2):
            classifier_arr.append((sys.argv[i], sys.argv[i + 1]))
        main(data_set, classifier_arr)
