"""Primary script used to execute the defaulter prediction"""
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
from config.constants import DATA_BALANCER_STR
import config.classifier_tester_parameters as ctp
import config.classifiers as cfr
from classifier_result_recorder import ClassifierResultRecorder
from config import constants as const
from config import data_sets
from data_preprocessing import apply_preprocessing
from generic_classifier import GenericClassifier
from run_statistics import RunStatistics

data_balancers = [None, ClusterCentroids, EditedNearestNeighbours, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule,
                  OneSidedSelection, RandomUnderSampler, TomekLinks, ADASYN, RandomOverSampler, SMOTE, SMOTEENN, SMOTETomek]


def execute_loop(classifier_description, classifier_dict, parameter_dict, defaulter_set_arr, results_recorder, z, parameter_grid_len, requires_random_state, data_set_description):
    if z % 5 == 0:
        print("==== {0} - {1} - {2}% ====".format(data_set_description, classifier_description, format((float(z) / parameter_grid_len) * 100, '.2f')))

    for data_balancer in data_balancers:
        test_stats = RunStatistics()
        success = True
        for x in range(const.TEST_REPEAT):
            try:
                generic_classifier = GenericClassifier(classifier_dict['classifier'], parameter_dict, data_balancer)
                if requires_random_state:
                    result_dictionary = generic_classifier.train_and_evaluate(defaulter_set_arr, None)
                else:
                    result_dictionary = generic_classifier.train_and_evaluate(defaulter_set_arr, x)
                test_stats.append_run_result(result_dictionary, generic_classifier.ml_stats.roc_list)
            except Exception:
                success = False
                print("INFO: parameter caused classifier to raise exception")

        if success:
            avg_results = test_stats.calculate_average_run_accuracy()
            sorted_keys = sorted(parameter_dict)
            values = [parameter_dict.get(k) for k in sorted_keys if k in parameter_dict] + [data_balancer.__name__ if data_balancer is not None else "None"]
            results_recorder.record_results(values + avg_results)


if __name__ == "__main__":
    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")

            # Preprocess data set
            input_defaulter_set = apply_preprocessing(input_defaulter_set, data_set["numeric_columns"], data_set["categorical_columns"], data_set["classification_label"],
                                                      data_set["missing_values_strategy"])

            cpu_count = get_number_of_processes_to_use()
            for classifier_description, classifier_dict in cfr.classifiers.iteritems():
                parameter_dict = ctp.generic_classifier_parameter_dict[classifier_description]
                if classifier_dict['status'] and parameter_dict is not None:
                    manager = Manager()
                    result_recorder = ClassifierResultRecorder(result_arr=manager.list())

                    # Execute enabled classifiers
                    parameter_grid = ParameterGrid(parameter_dict["parameters"])
                    Parallel(n_jobs=cpu_count)(
                        delayed(execute_loop)(classifier_description, classifier_dict, parameter_grid[z], input_defaulter_set, result_recorder, z, len(parameter_grid),
                                              parameter_dict["requires_random_state"], data_set["data_set_description"]) for z in
                        range(len(parameter_grid)))

                    if const.RECORD_RESULTS is True:
                        result_recorder.save_results_to_file(sorted(parameter_grid[0]) + [DATA_BALANCER_STR],
                                                             prepend_name_description=data_set["data_set_description"] + "_" + classifier_description)
