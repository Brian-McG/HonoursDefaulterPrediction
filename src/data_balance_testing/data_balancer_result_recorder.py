import csv
import os
from datetime import datetime

from config import constants as const


class DataBalancerResultRecorder:
    def __init__(self, result_arr=None):
        if result_arr is None:
            result_arr = []
        self.results = result_arr

    def record_results(self, result):
        self.results.append(result)

    def save_results_to_file(self, prepend_file_name):
        """Records results to file. If file_name is None, then a default filename of data_<number of folds>_<timestamp>.csv"""
        if len(self.results) > 0:
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            file_name = "{0}_data_balancer_{1}-folds_{2}.csv".format(prepend_file_name, const.NUMBER_OF_FOLDS, current_time)
            output_file = open(os.path.dirname(os.path.realpath(__file__)) + "/../../results/" + file_name, "wb")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            title_row = ["Classifier description", "Data balancer", "Average true rate", "Average true positive rate",
                         "Average true negative rate", "Average false positive rate", "Average false negative rate", "Average true positive with cutoff",
                         "Average true negative rate with cutoff", "Average false positive rate with cutoff", "Average false negative rate with cutoff",
                         "Average unclassified from cutoff"]
            csv_writer.writerow(title_row)
            for result_tuple in self.results:
                csv_writer.writerow((result_tuple[0], result_tuple[1][0], result_tuple[1][1][0], result_tuple[1][1][1], result_tuple[1][1][2], result_tuple[1][1][3],
                                     result_tuple[1][1][4], result_tuple[1][1][5], result_tuple[1][1][6], result_tuple[1][1][7], result_tuple[1][1][8],
                                     result_tuple[1][1][9]))
            output_file.close()