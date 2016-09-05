import csv
from datetime import datetime

import constants as const


class ClassifierResultRecorder:
    def __init__(self):
        self.results = []

    def record_results(self, result):
        self.results.append(result)

    def save_results_to_file(self, classifier_details, file_name=None):
        """Records results to file. If file_name is None, then a default filename of data_<number of folds>_<timestamp>.csv"""
        if len(self.results) > 0:
            if file_name is None:
                current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                file_name = "classifier_result_recorder_{0}-folds_{1}.csv".format(const.NUMBER_OF_FOLDS, current_time)
            output_file = open(file_name, "w", newline="")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            title_row = classifier_details + ["Average true rate", "Average true positive rate",
                         "Average true negative rate", "Average false positive rate", "Average false negative rate", "Average true positive with cutoff",
                         "Average true negative rate with cutoff", "Average false positive rate with cutoff", "Average false negative rate with cutoff",
                         "Average unclassified from cutoff"]
            csv_writer.writerow(title_row)
            for result_arr in self.results:
                csv_writer.writerow(result_arr)
            output_file.close()
