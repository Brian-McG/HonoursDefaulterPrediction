import csv
import os
from datetime import datetime

from config import constants as const


class ResultRecorder:
    def __init__(self):
        self.results = []

    def record_results(self, result_dict, classifier_dict):
        self.results.append((result_dict, classifier_dict))

    def save_results_to_file(self, data_set_description):
        """Records results to file. If file_name is None, then a default filename of data_<number of folds>_<timestamp>.csv"""
        if len(self.results) > 0:
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            file_name = "{0}_data_{1}-folds_{2}.csv".format(data_set_description, const.NUMBER_OF_FOLDS, current_time)
            output_file = open(os.path.dirname(os.path.realpath(__file__)) + "/../results/" + file_name, "wb")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            title_row = ["Classifier description", "Average true rate", "Average true positive rate",
                         "Average true negative rate", "Average false positive rate", "Average false negative rate"]
            csv_writer.writerow(title_row)
            for result_tuple in self.results:
                csv_writer.writerow((result_tuple[1], result_tuple[0]['avg_true_rate'], result_tuple[0]["avg_true_positive_rate"],
                                     result_tuple[0]["avg_true_negative_rate"], result_tuple[0]["avg_false_positive_rate"], result_tuple[0]["avg_false_negative_rate"]))
            output_file.close()
