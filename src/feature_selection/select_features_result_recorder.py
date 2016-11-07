import csv
import os
from datetime import datetime

import constants as const


class FeatureSelectionResultRecorder:
    def __init__(self, result_arr=None):
        if result_arr is None:
            self.results = []
        else:
            self.results = result_arr

    def record_results(self, result_dict, classifier_dict, feature_selection_strategy, features_selected, feature_summary=None):
        self.results.append((result_dict, classifier_dict, feature_selection_strategy, features_selected, feature_summary))

    def save_results_to_file(self, random_values, data_set_description):
        """Records results to file. If file_name is None, then a default filename of data_<number of folds>_<timestamp>.csv"""
        if len(self.results) > 0:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = "{0}_data_{1}-folds_{2}.csv".format(data_set_description, const.NUMBER_OF_FOLDS, current_time)
            output_file = open(os.path.dirname(os.path.realpath(__file__)) + "/../../results/" + file_name, "wb")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            title_row = ["Feature selection strategy", "Classifier description", "Matthews correlation coefficient", "Cohen Kappa score", "Average true rate", "Average true positive rate",
                         "Average true negative rate", "Average false positive rate", "Average false negative rate", "initialisation_values", "features_selected", "feature_summary"]
            csv_writer.writerow(title_row)
            x = 0
            feature_selection_approach = None
            feature_summary = None
            for result_tuple in self.results:
                if x == 0:
                    random_vals = random_values
                else:
                    random_vals = None
                if result_tuple[3] == feature_selection_approach:
                    features = None
                else:
                    features = result_tuple[3]
                if result_tuple[4] == feature_summary:
                    feature_smmry = None
                else:
                    feature_smmry = result_tuple[4]
                feature_selection_approach = result_tuple[3]
                csv_writer.writerow((result_tuple[2], result_tuple[1], result_tuple[0][0], result_tuple[0][1],
                                     result_tuple[0][2], result_tuple[0][15], result_tuple[0][28], result_tuple[0][3], result_tuple[0][4], result_tuple[0][5], result_tuple[0][6],
                                     result_tuple[0][20], result_tuple[0][21], result_tuple[0][13], result_tuple[0][22], result_tuple[0][29], result_tuple[0][24], result_tuple[0][25], random_vals, features, feature_smmry))
                x += 1
            output_file.close()
