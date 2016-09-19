import csv
import os
from datetime import datetime

from config import constants as const


class ClassifierResultRecorder:
    def __init__(self, result_arr=None):
        if result_arr is None:
            result_arr = []
        self.results = result_arr

    def record_results(self, result):
        self.results.append(result)

    def save_results_to_file(self, classifier_details, file_name=None, prepend_name_description=""):
        """Records results to file. If file_name is None, then a default filename of data_<number of folds>_<timestamp>.csv"""
        if len(self.results) > 0:
            if file_name is None:
                current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                file_name = "{0}_classifier_result_recorder_{1}-folds_{2}.csv".format(prepend_name_description, const.NUMBER_OF_FOLDS, current_time)
            output_file = open(os.path.dirname(os.path.realpath(__file__)) + "/../../results/" + file_name, "wb")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            title_row = classifier_details + ["Average true rate", "Average true positive rate",
                                              "Average true negative rate", "Average false positive rate", "Average false negative rate", "Average true positive with cutoff",
                                              "Average true negative rate with cutoff", "Average false positive rate with cutoff", "Average false negative rate with cutoff",
                                              "Average unclassified from cutoff"]
            csv_writer.writerow(title_row)
            for result_arr in self.results:
                csv_writer.writerow(result_arr)
            output_file.close()
