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

    # Works
    #

    # does not work
    # RepeatedEditedNearestNeighbours()

    # Requires additional setup
    # BalanceCascade(), EasyEnsemble()

    data_balancers = [None, ClusterCentroids(), CondensedNearestNeighbour(), EditedNearestNeighbours(), InstanceHardnessThreshold(), NearMiss(), NeighbourhoodCleaningRule(),
                      OneSidedSelection(), RandomUnderSampler(), TomekLinks(), ADASYN(), RandomOverSampler(), SMOTE(), SMOTEENN(), SMOTETomek()]
    result_recorder = DataBalancerResultRecorder()

    # Execute enabled classifiers
    for classifier_dict in cfr.generic_classifiers:

        for data_balancer in data_balancers:
            individual_data_balancer_results = [None, None, None, None, None, None, None, None, None, None]
            print(data_balancer.__class__.__name__)
            if classifier_dict['status']:
                generic_classifier = GenericClassifier(classifier_dict['classifier'], data_balancer)
                result_dictionary = generic_classifier.train_and_evaluate(input_defaulter_set)

                individual_data_balancer_results[0] = ("overall_true_rate", result_dictionary["avg_true_positive_rate"] + result_dictionary["avg_true_negative_rate"])
                individual_data_balancer_results[1] = ("true_positive_rate", result_dictionary["avg_true_positive_rate"])
                individual_data_balancer_results[2] = ("true_negative_rate", result_dictionary["avg_true_negative_rate"])
                individual_data_balancer_results[3] = ("false_positive_rate", result_dictionary["avg_false_positive_rate"])
                individual_data_balancer_results[4] = ("false_negative_rate", result_dictionary["avg_false_negative_rate"])
                individual_data_balancer_results[5] = ("true_positive_rate_cutoff", result_dictionary["avg_true_positive_rate_with_prob_cutoff"])
                individual_data_balancer_results[6] = ("true_negative_rate_cutoff", result_dictionary["avg_true_negative_rate_with_prob_cutoff"])
                individual_data_balancer_results[7] = ("false_positive_rate_cutoff", result_dictionary["avg_false_positive_rate_with_prob_cutoff"])
                individual_data_balancer_results[8] = ("false_negative_rate_cutoff", result_dictionary["avg_false_negative_rate_with_prob_cutoff"])
                individual_data_balancer_results[9] = ("unclassified_cutoff", result_dictionary["avg_false_negative_rate_with_prob_cutoff"])
                classifier_tuple = (data_balancer.__class__.__name__, individual_data_balancer_results)
                result_recorder.record_results((classifier_dict['classifier_description'], classifier_tuple))

    for classifier_dict in cfr.non_generic_classifiers:
        for data_balancer in data_balancers:
            if classifier_dict['status']:
                classifier_dict['classifier'].data_balancer = data_balancer
                result_dictionary = (classifier_dict['classifier']).train_and_evaluate(input_defaulter_set)
                individual_data_balancer_results[0] = ("overall_true_rate", result_dictionary["avg_true_positive_rate"] + result_dictionary["avg_true_negative_rate"])
                individual_data_balancer_results[1] = ("true_positive_rate", result_dictionary["avg_true_positive_rate"])
                individual_data_balancer_results[2] = ("true_negative_rate", result_dictionary["avg_true_negative_rate"])
                individual_data_balancer_results[3] = ("false_positive_rate", result_dictionary["avg_false_positive_rate"])
                individual_data_balancer_results[4] = ("false_negative_rate", result_dictionary["avg_false_negative_rate"])
                individual_data_balancer_results[5] = ("true_positive_rate_cutoff", result_dictionary["avg_true_positive_rate_with_prob_cutoff"])
                individual_data_balancer_results[6] = ("true_negative_rate_cutoff", result_dictionary["avg_true_negative_rate_with_prob_cutoff"])
                individual_data_balancer_results[7] = ("false_positive_rate_cutoff", result_dictionary["avg_false_positive_rate_with_prob_cutoff"])
                individual_data_balancer_results[8] = ("false_negative_rate_cutoff", result_dictionary["avg_false_negative_rate_with_prob_cutoff"])
                individual_data_balancer_results[9] = ("unclassified_cutoff", result_dictionary["avg_false_negative_rate_with_prob_cutoff"])
                classifier_tuple = (data_balancer.__class__.__name__, individual_data_balancer_results)
                result_recorder.record_results((classifier_dict['classifier_description'], classifier_tuple))

    print(result_recorder.results)
    if const.RECORD_RESULTS is True:
        result_recorder.save_results_to_file()


if __name__ == "__main__":
    # Add ANN to classifier list - this needs to be here due to the use of Processes in ArtificialNeuralNetwork
    ann = ArtificialNeuralNetwork()
    cfr.append_classifier_details(None, ann, cfr.ann_enabled, "Artificial neural network", cfr.non_generic_classifiers)

    # Run main
    main()
