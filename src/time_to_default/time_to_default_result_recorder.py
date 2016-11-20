import csv
import os
from datetime import datetime

import constants as const
from constants import TITLE_ROW_WITH_DEFAULT_TIME_RANGE


class TimeToDefaultResultRecorder:
    """Records time to default results to file"""
    def __init__(self, result_arr=None):
        if result_arr is None:
            self.results = []
        else:
            self.results = result_arr

    def record_results(self, result_dict, classifier_dict, default_time_range):
        """Records an individual result"""
        self.results.append((result_dict, classifier_dict, default_time_range))

    def save_results_to_file(self, random_values, data_set_description):
        """Saves all results to file"""
        if len(self.results) > 0:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = "{0}_data_{1}-folds_{2}.csv".format(data_set_description, const.NUMBER_OF_FOLDS, current_time)
            output_file = open(os.path.dirname(os.path.realpath(__file__)) + "/../../results/" + file_name, "wb")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(TITLE_ROW_WITH_DEFAULT_TIME_RANGE)
            x = 0
            for result_tuple in self.results:
                if x == 0:
                    random_vals = random_values
                else:
                    random_vals = None
                csv_writer.writerow((result_tuple[2], result_tuple[1], result_tuple[0][0], result_tuple[0][1],
                                     result_tuple[0][2], result_tuple[0][15], result_tuple[0][28], result_tuple[0][3], result_tuple[0][4], result_tuple[0][5], result_tuple[0][6],
                                     result_tuple[0][20], result_tuple[0][21], result_tuple[0][13], result_tuple[0][22], result_tuple[0][29], result_tuple[0][24], result_tuple[0][25], random_vals))
                x += 1
            output_file.close()
