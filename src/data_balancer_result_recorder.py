import csv
from datetime import datetime

import constants as const
import sys


class DataBalancerResultRecorder:
    def __init__(self):
        self.results = []

    def record_results(self, result):
        self.results.append(result)

    def save_results_to_file(self, file_name=None):
        """Records results to file. If file_name is None, then a default filename of data_<number of folds>_<timestamp>.csv"""
        if len(self.results) > 0:
            if file_name is None:
                current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                file_name = "data_balancer_{0}-folds_{1}.csv".format(const.NUMBER_OF_FOLDS, current_time)
            output_file = open(sys.path[0] + "/../results/" + file_name, "w", newline="")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            title_row = ["Classifier description", "Data balancer", "Average true rate", "Average true positive rate",
                         "Average true negative rate", "Average false positive rate", "Average false negative rate", "Average true positive with cutoff",
                         "Average true negative rate with cutoff", "Average false positive rate with cutoff", "Average false negative rate with cutoff",
                         "Average unclassified from cutoff"]
            csv_writer.writerow(title_row)
            for result_tuple in self.results:
                csv_writer.writerow((result_tuple[0], result_tuple[1][0], result_tuple[1][1][0][1], result_tuple[1][1][1][1], result_tuple[1][1][2][1], result_tuple[1][1][3][1],
                                     result_tuple[1][1][4][1], result_tuple[1][1][5][1], result_tuple[1][1][6][1], result_tuple[1][1][7][1], result_tuple[1][1][8][1],
                                     result_tuple[1][1][9][1]))
            output_file.close()
