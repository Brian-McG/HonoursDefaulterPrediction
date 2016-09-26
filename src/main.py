"""Primary script used to execute the defaulter prediction"""

import pandas as pd

import config.classifiers as cfr
import visualisation as vis
from config import constants as const
from config import data_sets
from data_preprocessing import apply_preprocessing
from generic_classifier import GenericClassifier
from result_recorder import ResultRecorder
from run_statistics import RunStatistics


def main():
    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")

            # Preprocess data set
            input_defaulter_set = apply_preprocessing(input_defaulter_set, data_set["numeric_columns"], data_set["categorical_columns"], data_set["classification_label"], data_set["missing_values_strategy"])
            result_recorder = ResultRecorder()
            roc_plot = []

            # Execute enabled classifiers
            for classifier_description, classifier_dict in cfr.classifiers.iteritems():
                if classifier_dict["status"]:
                    print("\n=== {0} ===".format(classifier_description))
                    test_stats = RunStatistics()
                    for i in range(const.TEST_REPEAT):
                        classifier_parameters = data_set["data_set_classifier_parameters"]
                        generic_classifier = GenericClassifier(classifier_dict["classifier"],
                                                               classifier_parameters.classifier_parameters[classifier_description]["classifier_parameters"],
                                                               classifier_parameters.classifier_parameters[classifier_description]["data_balancer"])
                        result_dictionary = generic_classifier.train_and_evaluate(input_defaulter_set, None)
                        vis.plot_roc_curve_of_classifier(generic_classifier.ml_stats.roc_list, data_set["data_set_description"] + "_run{0}_".format(i + 1), classifier_description)
                        test_stats.append_run_result(result_dictionary, generic_classifier.ml_stats.roc_list)

                    avg_results = test_stats.calculate_average_run_accuracy()
                    roc_plot.append((test_stats.roc_list, classifier_description))
                    result_recorder.record_results(avg_results, classifier_description)
                    print("Matthews correlation coefficient: {0}".format(avg_results[0]))
                    print("Cohen Kappa score: {0}".format(avg_results[1]))
                    print("Average true rate: {0}".format(avg_results[2]))
                    print("Average true positive rate: {0}".format(avg_results[3]))
                    print("Average true negative rate: {0}".format(avg_results[4]))
                    print("Average false positive rate: {0}".format(avg_results[5]))
                    print("Average false negative rate: {0}".format(avg_results[6]))

            if const.RECORD_RESULTS:
                vis.plot_mean_roc_curve_of_classifiers(roc_plot, data_set["data_set_description"])
                result_recorder.save_results_to_file(data_set["data_set_description"])


if __name__ == "__main__":
    # Run main
    main()
