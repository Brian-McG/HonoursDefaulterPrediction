"""Primary script used to execute the defaulter prediction"""
import multiprocessing
import pandas as pd
from joblib import Parallel
from joblib import delayed
from multiprocessing import Manager
from sklearn.grid_search import ParameterGrid

import classifier_tester_parameters as ctp

import classifiers as cfr
import constants as const
from artificial_neural_network import ArtificialNeuralNetwork
from classifier_result_recorder import ClassifierResultRecorder
from data_preprocessing import apply_preprocessing
from generic_classifier import GenericClassifier


def execute_loop(classifier_dict, parameter_dict, input_defaulter_set, result_recorder, z, paramater_grid_len):
    classifier = classifier_dict['classifier'].__class__(**parameter_dict)
    generic_classifier = GenericClassifier(classifier, classifier_dict['data_balancer'])
    overall_true_rate, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, true_positive_rate_cutoff, true_negative_rate_cutoff, \
        false_positive_rate_cutoff, false_negative_rate_cutoff, unclassified_cutoff = [0] * 10
    if z % 5 == 0:
        print("==== {0}% ====".format(format((z/paramater_grid_len) * 100, '.2f')))
    for i in range(const.TEST_REPEAT):
        try:
            result_dictionary = generic_classifier.train_and_evaluate(input_defaulter_set)
        except Exception:
            const.verbose_print("WARNING: incompatible input parameters")
            return
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

    individual_data_balancer_results = [None, None, None, None, None, None, None, None, None, None]
    individual_data_balancer_results[0] = overall_true_rate / const.TEST_REPEAT
    individual_data_balancer_results[1] = true_positive_rate / const.TEST_REPEAT
    individual_data_balancer_results[2] = true_negative_rate / const.TEST_REPEAT
    individual_data_balancer_results[3] = false_positive_rate / const.TEST_REPEAT
    individual_data_balancer_results[4] = false_negative_rate / const.TEST_REPEAT
    individual_data_balancer_results[5] = true_positive_rate_cutoff / const.TEST_REPEAT
    individual_data_balancer_results[6] = true_negative_rate_cutoff / const.TEST_REPEAT
    individual_data_balancer_results[7] = false_positive_rate_cutoff / const.TEST_REPEAT
    individual_data_balancer_results[8] = false_negative_rate_cutoff / const.TEST_REPEAT
    individual_data_balancer_results[9] = unclassified_cutoff / const.TEST_REPEAT
    sorted_keys = sorted(parameter_dict)
    values = [parameter_dict.get(k) for k in sorted_keys if k in parameter_dict]
    result_recorder.record_results(values + individual_data_balancer_results)


if __name__ == "__main__":
    # Add ANN to classifier list - this needs to be here due to the use of Processes in ArtificialNeuralNetwork
    ann = ArtificialNeuralNetwork(cfr.ann_data_balancer)
    cfr.append_classifier_details(None, ann, cfr.ann_enabled, "Artificial neural network", cfr.non_generic_classifiers)

    input_defaulter_set = pd.DataFrame.from_csv("../data/lima_tb/Lima-TB-Treatment-base.csv", index_col=None,
                                                encoding="UTF-8")
    # input_defaulter_set = pd.DataFrame.from_csv("../data/german_finance/german_dataset_numberised.csv", index_col=None, encoding="UTF-8")
    # input_defaulter_set = pd.DataFrame.from_csv("../data/australian_finance/australian.csv", index_col=None, encoding="UTF-8")
    # input_defaulter_set = pd.DataFrame.from_csv("../data/credit_screening/credit_screening.csv", index_col=None, encoding="UTF-8")

    # Preprocess data set
    input_defaulter_set = apply_preprocessing(input_defaulter_set)

    #assert(len(ctp.generic_classifier_parameter_arr) == len(cfr.generic_classifiers))
    logical_cpu_count = multiprocessing.cpu_count()

    for i in range(len(ctp.generic_classifier_parameter_arr)):
        if cfr.generic_classifiers[i]['status'] is True and cfr.generic_classifiers[i] is not None:
            manager = Manager()
            result_recorder = ClassifierResultRecorder(result_arr=manager.list())

            # Execute enabled classifiers
            parameter_grid = ParameterGrid(ctp.generic_classifier_parameter_arr[i])
            Parallel(n_jobs=logical_cpu_count)(delayed(execute_loop)(cfr.generic_classifiers[i], parameter_grid[z], input_defaulter_set, result_recorder, z, len(parameter_grid)) for z in range(len(parameter_grid)))

            print(result_recorder.results)
            if const.RECORD_RESULTS is True:
                result_recorder.save_results_to_file(sorted(parameter_grid[0]), prepend_name_description=cfr.generic_classifiers[i]['classifier_description'])
