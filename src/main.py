"""Primary script used to execute the defaulter prediction"""
import inspect

import pandas as pd

import config.classifiers as cfr
import visualisation as vis
from config import constants as const
from data_preprocessing import apply_preprocessing
from generic_classifier import GenericClassifier
from result_recorder import ResultRecorder
from config import data_sets


def main():
    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")

            # Preprocess data set
            input_defaulter_set = apply_preprocessing(input_defaulter_set)
            result_recorder = ResultRecorder()
            roc_plot = []

            # Execute enabled classifiers
            for classifier_dict in cfr.classifiers:
                if classifier_dict["status"]:
                    generic_classifier = GenericClassifier(classifier_dict["classifier"], classifier_dict["classifier_parameters"], classifier_dict["data_balancer"])
                    result_dictionary = generic_classifier.train_and_evaluate(input_defaulter_set, None)
                    result_recorder.record_results(result_dictionary, classifier_dict)
                    vis.plot_roc_curve_of_classifier(generic_classifier.ml_stats.roc_list, data_set["data_set_description"], classifier_dict["classifier_description"])
                    roc_plot.append((generic_classifier.ml_stats.roc_list, classifier_dict["classifier_description"]))

            if const.RECORD_RESULTS:
                vis.plot_mean_roc_curve_of_classifiers(roc_plot, data_set["data_set_description"])
                result_recorder.save_results_to_file(data_set["data_set_description"])


if __name__ == "__main__":
    # Run main
    main()
