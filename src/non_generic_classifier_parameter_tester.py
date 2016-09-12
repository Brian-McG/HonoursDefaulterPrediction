"""Primary script used to execute the defaulter prediction"""
import multiprocessing
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
from multiprocessing import Manager

from joblib import Parallel
from joblib import delayed
from sklearn.grid_search import ParameterGrid

import classifier_tester_parameters as ctp

import classifiers as cfr
import constants as const
from artificial_neural_network import ArtificialNeuralNetwork
from classifier_result_recorder import ClassifierResultRecorder
from data_preprocessing import apply_preprocessing
from result_recorder import ResultRecorder


def main():
    # Load in data set
    input_defaulter_set = pd.DataFrame.from_csv("../data/lima_tb/Lima-TB-Treatment-base.csv", index_col=None,
                                                encoding="UTF-8")
    # input_defaulter_set = pd.DataFrame.from_csv("../data/german_finance/german_dataset_numberised.csv", index_col=None, encoding="UTF-8")
    # input_defaulter_set = pd.DataFrame.from_csv("../data/australian_finance/australian.csv", index_col=None, encoding="UTF-8")
    # input_defaulter_set = pd.DataFrame.from_csv("../data/credit_screening/credit_screening.csv", index_col=None, encoding="UTF-8")

    # Preprocess data set
    input_defaulter_set = apply_preprocessing(input_defaulter_set)

    assert len(ctp.non_generic_classifier_parameter_arr) == len(cfr.non_generic_classifiers)

    logical_cpu_count = multiprocessing.cpu_count()

    for i in range(len(ctp.non_generic_classifier_parameter_arr)):
        if cfr.non_generic_classifiers[i]['status'] is True:
            manager = Manager()
            result_recorder = ClassifierResultRecorder(result_arr=manager.list())

            # Execute enabled classifiers
            parameter_grid = ParameterGrid(ctp.non_generic_classifier_parameter_arr)
            Parallel(n_jobs=logical_cpu_count)(delayed(run_test)(cfr.non_generic_classifiers[i]['classifier'], input_defaulter_set, result_recorder, parameter_grid[z], z, len(parameter_grid)) for z in range(len(parameter_grid)))

            print(result_recorder.results)
            if const.RECORD_RESULTS is True:
                result_recorder.save_results_to_file(sorted(parameter_grid[0]) + ["Data balancer"], prepend_name_description=cfr.non_generic_classifiers[i]['classifier'].__class__.__name__)


def run_test(classifier, input_defaulter_set, result_recorder, parameter_dict, z, paramater_grid_len):
    data_balancers = [None, ClusterCentroids(), EditedNearestNeighbours(), InstanceHardnessThreshold(), NearMiss(), NeighbourhoodCleaningRule(),
                      OneSidedSelection(), RandomUnderSampler(), TomekLinks(), ADASYN(), RandomOverSampler(), SMOTE(), SMOTEENN(), SMOTETomek()]

    if z % 5 == 0:
        print("==== {0}% ====".format(format((z/paramater_grid_len) * 100, '.2f')))

    for data_balancer in data_balancers:
        # Execute classifier TEST_REPEAT number of times
        overall_true_rate, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, true_positive_rate_cutoff, true_negative_rate_cutoff, \
            false_positive_rate_cutoff, false_negative_rate_cutoff, unclassified_cutoff = [0] * 10
        for i in range(const.TEST_REPEAT):
            result_dictionary = classifier.train_and_evaluate(input_defaulter_set, number_of_threads=1, **parameter_dict)
            overall_true_rate += (result_dictionary["avg_true_positive_rate"] + result_dictionary["avg_true_negative_rate"]) / 2.0
            true_positive_rate += result_dictionary["avg_true_positive_rate"]
            true_negative_rate += result_dictionary["avg_true_negative_rate"]
            false_positive_rate += result_dictionary["avg_false_positive_rate"]
            false_negative_rate += result_dictionary["avg_false_negative_rate"]
            true_positive_rate_cutoff += result_dictionary["avg_true_positive_rate_with_prob_cutoff"]
            true_negative_rate_cutoff += result_dictionary["avg_true_negative_rate_with_prob_cutoff"]
            false_positive_rate_cutoff += result_dictionary["avg_false_positive_rate_with_prob_cutoff"]
            false_negative_rate_cutoff += result_dictionary["avg_false_negative_rate_with_prob_cutoff"]
            unclassified_cutoff += result_dictionary["avg_false_negative_rate_with_prob_cutoff"]

        individual_results = [None, None, None, None, None, None, None, None, None, None]
        individual_results[0] = overall_true_rate / const.TEST_REPEAT
        individual_results[1] = true_positive_rate / const.TEST_REPEAT
        individual_results[2] = true_negative_rate / const.TEST_REPEAT
        individual_results[3] = false_positive_rate / const.TEST_REPEAT
        individual_results[4] = false_negative_rate / const.TEST_REPEAT
        individual_results[5] = true_positive_rate_cutoff / const.TEST_REPEAT
        individual_results[6] = true_negative_rate_cutoff / const.TEST_REPEAT
        individual_results[7] = false_positive_rate_cutoff / const.TEST_REPEAT
        individual_results[8] = false_negative_rate_cutoff / const.TEST_REPEAT
        individual_results[9] = unclassified_cutoff / const.TEST_REPEAT
        sorted_keys = sorted(parameter_dict)
        values = [parameter_dict.get(k) for k in sorted_keys if k in parameter_dict] + [data_balancer.__class__.__name__]
        result_recorder.record_results(values + [data_balancer.__class__.__name__] + individual_results)


if __name__ == "__main__":
    # Add ANN to classifier list - this needs to be here due to the use of Processes in ArtificialNeuralNetwork
    ann = ArtificialNeuralNetwork(cfr.ann_data_balancer)
    cfr.append_classifier_details(None, ann, cfr.ann_enabled, "Artificial neural network", cfr.non_generic_classifiers)

    # Run main
    main()
