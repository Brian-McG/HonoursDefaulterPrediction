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
from sklearn.metrics import auc

import classifiers as cfr
import constants as const
from artificial_neural_network import ArtificialNeuralNetwork
from data_balancer_result_recorder import DataBalancerResultRecorder
from data_preprocessing import apply_preprocessing
from generic_classifier import GenericClassifier
from result_recorder import ResultRecorder
import visualisation as vis


def main():
    # Load in data set
    input_defaulter_set = pd.DataFrame.from_csv("../data/lima_tb/Lima-TB-Treatment-base.csv", index_col=None,
                                                encoding="UTF-8")
    # input_defaulter_set = pd.DataFrame.from_csv("../data/german_finance/german_dataset_numberised.csv", index_col=None, encoding="UTF-8")
    # input_defaulter_set = pd.DataFrame.from_csv("../data/australian_finance/australian.csv", index_col=None, encoding="UTF-8")
    # input_defaulter_set = pd.DataFrame.from_csv("../data/credit_screening/credit_screening.csv", index_col=None, encoding="UTF-8")

    # Preprocess data set
    input_defaulter_set = apply_preprocessing(input_defaulter_set)

    data_balancers = [None, ClusterCentroids(), EditedNearestNeighbours(), InstanceHardnessThreshold(), NearMiss(), NeighbourhoodCleaningRule(),
                      OneSidedSelection(), RandomUnderSampler(), TomekLinks(), ADASYN(), RandomOverSampler(), SMOTE(), SMOTEENN(), SMOTETomek()]
    result_recorder = DataBalancerResultRecorder()

    # Execute enabled classifiers
    run_test(cfr.non_generic_classifiers, input_defaulter_set, data_balancers, result_recorder, is_generic=False)
    run_test(cfr.generic_classifiers, input_defaulter_set, data_balancers, result_recorder, is_generic=True)

    print(result_recorder.results)
    if const.RECORD_RESULTS is True:
        result_recorder.save_results_to_file()


def run_test(classifiers, input_defaulter_set, data_balancers, result_recorder, is_generic=True):
    for classifier_dict in classifiers:
        if classifier_dict['status'] is True:
            data_balance_roc_results = []
            print("== {0} ==".format(classifier_dict['classifier_description']))
            for data_balancer in data_balancers:
                classifier_roc_results = []
                print("=== {0} ===".format(data_balancer.__class__.__name__))
                overall_true_rate, true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate, true_positive_rate_cutoff, true_negative_rate_cutoff, \
                    false_positive_rate_cutoff, false_negative_rate_cutoff, unclassified_cutoff = [0] * 10

                # Execute classifier TEST_REPEAT number of times
                for i in range(const.TEST_REPEAT):
                    print("==== Run {0} ====".format(i+1))
                    if is_generic is True:
                        classifier = GenericClassifier(classifier_dict['classifier'], data_balancer)
                    else:
                        classifier = classifier_dict['classifier']
                    result_dictionary = classifier.train_and_evaluate(input_defaulter_set)

                    # Add ROC results
                    classifier_roc_results.append(classifier.ml_stats.roc_list)

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

                data_balance_roc_results.append((classifier_roc_results, data_balancer.__class__.__name__))

                individual_data_balancer_results = [None, None, None, None, None, None, None, None, None, None]
                individual_data_balancer_results[0] = ("overall_true_rate", overall_true_rate / const.TEST_REPEAT)
                individual_data_balancer_results[1] = ("true_positive_rate", true_positive_rate / const.TEST_REPEAT)
                individual_data_balancer_results[2] = ("true_negative_rate", true_negative_rate / const.TEST_REPEAT)
                individual_data_balancer_results[3] = ("false_positive_rate", false_positive_rate / const.TEST_REPEAT)
                individual_data_balancer_results[4] = ("false_negative_rate", false_negative_rate / const.TEST_REPEAT)
                individual_data_balancer_results[5] = ("true_positive_rate_cutoff", true_positive_rate_cutoff / const.TEST_REPEAT)
                individual_data_balancer_results[6] = ("true_negative_rate_cutoff", true_negative_rate_cutoff / const.TEST_REPEAT)
                individual_data_balancer_results[7] = ("false_positive_rate_cutoff", false_positive_rate_cutoff / const.TEST_REPEAT)
                individual_data_balancer_results[8] = ("false_negative_rate_cutoff", false_negative_rate_cutoff / const.TEST_REPEAT)
                individual_data_balancer_results[9] = ("unclassified_cutoff", unclassified_cutoff / const.TEST_REPEAT)
                classifier_tuple = (data_balancer.__class__.__name__, individual_data_balancer_results)
                result_recorder.record_results((classifier_dict['classifier_description'], classifier_tuple))

            # Plot ROC results of each balancer
            vis.plot_mean_roc_curve_of_balancers(data_balance_roc_results, classifier_dict['classifier_description'])


if __name__ == "__main__":
    # Add ANN to classifier list - this needs to be here due to the use of Processes in ArtificialNeuralNetwork
    ann = ArtificialNeuralNetwork(cfr.ann_data_balancer)
    cfr.append_classifier_details(None, ann, cfr.ann_enabled, "Artificial neural network", cfr.non_generic_classifiers)

    # Run main
    main()
