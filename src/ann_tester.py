"""Primary script used to execute the defaulter prediction"""
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalanceCascade
from imblearn.ensemble import EasyEnsemble
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import TomekLinks

import classifiers as cfr
import constants as const
from artificial_neural_network import ArtificialNeuralNetwork
from classifier_result_recorder import ClassifierResultRecorder
from data_balancer_result_recorder import DataBalancerResultRecorder
from data_preprocessing import apply_preprocessing
from generic_classifier import GenericClassifier
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
    result_recorder = ResultRecorder()

    result_recorder = ClassifierResultRecorder()
    middle_layer_types = ['Rectifier', 'Sigmoid', 'Tanh', 'ExpLin']
    output_layer_types = ['Softmax']
    number_of_hidden_nodes = [1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 50, 75, 100]

    # Execute enabled classifiers
    run_test(cfr.non_generic_classifiers[0]['classifier'], input_defaulter_set, result_recorder, middle_layer_types, output_layer_types, number_of_hidden_nodes)

    print(result_recorder.results)
    if const.RECORD_RESULTS is True:
        result_recorder.save_results_to_file(["Hidden layer", "Number of hidden nodes", "Output layer"])


def run_test(classifier, input_defaulter_set, result_recorder, middle_layer_types, output_layer_types, number_of_hidden_nodes):

    for middle_layer_type in middle_layer_types:
        for number_of_hidden_node in number_of_hidden_nodes:
            for output_layer_type in output_layer_types:
                # Execute classifier TEST_REPEAT number of times
                overall_true_rate, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, true_positive_rate_cutoff, true_negative_rate_cutoff, false_positive_rate_cutoff, false_negative_rate_cutoff, unclassified_cutoff = [0] * 10
                for i in range(const.TEST_REPEAT):
                    print("==== Run {0} - {1} - {2} - {3} ====".format(i+1, middle_layer_type, number_of_hidden_node, output_layer_type))

                    result_dictionary = classifier.train_and_evaluate(input_defaulter_set, output_layer=output_layer_type, hidden_layer=middle_layer_type, number_of_hidden_nodes=number_of_hidden_node)
                    overall_true_rate += result_dictionary["avg_true_positive_rate"] + result_dictionary["avg_true_negative_rate"]
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
                result_recorder.record_results([middle_layer_type, number_of_hidden_node, output_layer_type] + individual_data_balancer_results)


if __name__ == "__main__":
    # Add ANN to classifier list - this needs to be here due to the use of Processes in ArtificialNeuralNetwork
    ann = ArtificialNeuralNetwork(cfr.ann_data_balancer)
    cfr.append_classifier_details(None, ann, cfr.ann_enabled, "Artificial neural network", cfr.non_generic_classifiers)

    # Run main
    main()
