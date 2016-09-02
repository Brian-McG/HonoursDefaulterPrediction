"""Primary script used to execute the defaulter prediction"""
import pandas as pd

# User imports
from multiprocessing import freeze_support

from artificial_neural_network import ArtificialNeuralNetwork
import constants as const
from data_preprocessing import apply_preprocessing
import classifiers as cfr
from generic_classifier import GenericClassifier


def main():
    # Load in data set
    input_defaulter_set = pd.DataFrame.from_csv("../data/lima_tb/Lima-TB-Treatment-base.csv", index_col=None, encoding="UTF-8")
    #input_defaulter_set = pd.DataFrame.from_csv("../data/german_finance/german_dataset_numberised.csv", index_col=None, encoding="UTF-8")
    #input_defaulter_set = pd.DataFrame.from_csv("../data/australian_finance/australian.csv", index_col=None, encoding="UTF-8")
    #input_defaulter_set = pd.DataFrame.from_csv("../data/credit_screening/credit_screening.csv", index_col=None, encoding="UTF-8")

    # Preprocess data set
    input_defaulter_set = apply_preprocessing(input_defaulter_set)

    # Execute enabled classifiers
    for classifier_dict in cfr.generic_classifiers:
        if classifier_dict['status']:
            generic_classifier = GenericClassifier(classifier_dict['classifier'], classifier_dict['data_balancer'])
            result_dictionary = generic_classifier.train_and_evaluate(input_defaulter_set)

    for classifier_dict in cfr.non_generic_classifiers:
        if classifier_dict['status']:
            result_dictionary = (classifier_dict['classifier']).train_and_evaluate(input_defaulter_set)

    # TODO: record results


if __name__ == "__main__":
    # Add ANN to classifier list - this needs to be here due to the use of Processes in ArtificialNeuralNetwork
    ann = ArtificialNeuralNetwork()
    cfr.append_classifier_details(None, ann, cfr.ann_enabled, "Artificial neural network", cfr.non_generic_classifiers)

    # Run main
    main()
