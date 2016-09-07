import csv
from datetime import datetime

import constants as const
import sys

class ResultRecorder:
    def __init__(self):
        self.results = []

    def record_results(self, result_dict, classifier_dict):
        self.results.append((result_dict, classifier_dict))

    def save_results_to_file(self, file_name=None):
        """Records results to file. If file_name is None, then a default filename of data_<number of folds>_<timestamp>.csv"""
        if len(self.results) > 0:
            if file_name is None:
                current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                file_name = "data_{0}-folds_{1}.csv".format(const.NUMBER_OF_FOLDS, current_time)
            output_file = open(sys.path[0] + "/../results/" + file_name, "w", newline="")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            title_row = ["Classifier description", "Average true positive rate",
                         "Average true negative rate", "Average false positive rate", "Average false negative rate"]
            csv_writer.writerow(title_row)
            for result_tuple in self.results:
                csv_writer.writerow((result_tuple[1]['classifier_description'], result_tuple[0]["avg_true_positive_rate"],
                                     result_tuple[0]["avg_true_negative_rate"],
                                     result_tuple[0]["avg_false_positive_rate"],
                                     result_tuple[0]["avg_false_negative_rate"]))
            output_file.close()
