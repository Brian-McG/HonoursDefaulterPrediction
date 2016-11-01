import csv
import os
from datetime import datetime

from config import constants as const
from config.constants import TITLE_ROW_WITH_TIME_TO_FIT, TITLE_ROW


class ResultRecorder:
    def __init__(self, result_arr=None):
        if result_arr is None:
            self.results = []
        else:
            self.results = result_arr

    def record_results(self, result_dict, classifier_dict):
        self.results.append((result_dict, classifier_dict))

    def save_results_to_file(self, random_values, data_set_description, display_time_to_fit_results=True):
        """Records results to file. If file_name is None, then a default filename of data_<number of folds>_<timestamp>.csv"""
        if len(self.results) > 0:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = "{0}_data_{1}-folds_{2}.csv".format(data_set_description, const.NUMBER_OF_FOLDS, current_time)
            output_file = open(os.path.dirname(os.path.realpath(__file__)) + "/../results/" + file_name, "wb")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)

            if display_time_to_fit_results:
                csv_writer.writerow(TITLE_ROW_WITH_TIME_TO_FIT)
                for result_tuple in self.results:
                    csv_writer.writerow((result_tuple[1], result_tuple[0][0], result_tuple[0][1],
                                         result_tuple[0][2], result_tuple[0][3], result_tuple[0][4], result_tuple[0][5], result_tuple[0][6], result_tuple[0][13], result_tuple[0][12], random_values))
            else:
                csv_writer.writerow(TITLE_ROW)
                for result_tuple in self.results:
                    csv_writer.writerow((result_tuple[1], result_tuple[0][0], result_tuple[0][1],
                                         result_tuple[0][2], result_tuple[0][3], result_tuple[0][4], result_tuple[0][5], result_tuple[0][6], result_tuple[0][13], random_values))
            output_file.close()
