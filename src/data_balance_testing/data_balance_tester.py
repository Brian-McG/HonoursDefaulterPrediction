"""Primary script used to execute the defaulter prediction"""
import os
import sys
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

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util import get_number_of_processes_to_use
import config.classifiers as cfr
import visualisation as vis
from config import constants as const
from config import data_sets
from data_balancer_result_recorder import DataBalancerResultRecorder
from data_preprocessing import apply_preprocessing
from generic_classifier import GenericClassifier
from run_statistics import RunStatistics


def main():
    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")

            # Preprocess data set
            input_defaulter_set = apply_preprocessing(input_defaulter_set)

            data_balancers = [None, ClusterCentroids, EditedNearestNeighbours, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule,
                              OneSidedSelection, RandomUnderSampler, TomekLinks, ADASYN, RandomOverSampler, SMOTE, SMOTEENN, SMOTETomek]

            cpu_count = get_number_of_processes_to_use()
            manager = Manager()
            result_recorder = DataBalancerResultRecorder(result_arr=manager.list())
            data_balance_roc_results = manager.list()

            # Execute enabled classifiers
            for classifier_description, classifier_dict in cfr.classifiers.iteritems():
                if classifier_dict["status"]:
                    print("== {0} ==".format(classifier_description))
                    Parallel(n_jobs=cpu_count)(delayed(run_test)(classifier_description, classifier_dict,
                                                                 data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["classifier_parameters"],
                                                                 input_defaulter_set, data_balancers[z], data_balance_roc_results, result_recorder) for z in
                                               range(len(data_balancers)))
                    # Plot ROC results of each balancer
                    vis.plot_mean_roc_curve_of_balancers(data_balance_roc_results, data_set["data_set_description"], classifier_description)

            print(result_recorder.results)
            if const.RECORD_RESULTS is True:
                result_recorder.save_results_to_file(data_set["data_set_description"])


def run_test(classifier_description, classifier_dict, classifier_parameters, input_defaulter_set, data_balancer, data_balance_roc_results, result_recorder):
    classifier_roc_results = []
    test_stats = RunStatistics()

    # Execute classifier TEST_REPEAT number of times
    for i in range(const.TEST_REPEAT):
        print("==== {0} - Run {1} ====".format(data_balancer.__name__ if data_balancer is not None else "None", i + 1))
        classifier = GenericClassifier(classifier_dict["classifier"], classifier_parameters, data_balancer)
        result_dictionary = classifier.k_fold_train_and_evaluate(input_defaulter_set, i)

        # Add ROC results
        classifier_roc_results.append(classifier.ml_stats.roc_list)
        test_stats.append_run_result(result_dictionary, classifier.ml_stats.roc_list)

    data_balance_roc_results.append((classifier_roc_results, data_balancer.__name__ if data_balancer is not None else "None"))

    individual_data_balancer_results = test_stats.calculate_average_run_accuracy()
    classifier_tuple = (data_balancer.__name__ if data_balancer is not None else "None", individual_data_balancer_results)
    result_recorder.record_results((classifier_description, classifier_tuple))


if __name__ == "__main__":
    # Run main
    main()
