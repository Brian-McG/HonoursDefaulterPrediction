import csv
import os
from datetime import datetime

from scipy.stats import friedmanchisquare, f_oneway

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
                                         result_tuple[0][2], result_tuple[0][15], result_tuple[0][3], result_tuple[0][4], result_tuple[0][5], result_tuple[0][6], result_tuple[0][13], result_tuple[0][12], random_values))
            else:
                csv_writer.writerow(TITLE_ROW)
                for result_tuple in self.results:
                    csv_writer.writerow((result_tuple[1], result_tuple[0][0], result_tuple[0][1],
                                         result_tuple[0][2], result_tuple[0][15], result_tuple[0][3], result_tuple[0][4], result_tuple[0][5], result_tuple[0][6], result_tuple[0][13], random_values))
            output_file.close()

    @staticmethod
    def save_results_for_multi_dataset(dataset_results):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        metric = ["BARR", "AUC", "BS", "fit_time", "MCC"]
        index = [14, 16, 17, 18, 19]
        file_paths = []
        assert len(metric) == len(index)

        for z in range(len(metric)):
            file_name = "classifier_{1}_results_full_{0}.csv".format(current_time, metric[z])
            file_path = os.path.dirname(os.path.realpath(__file__)) + "/../results/" + file_name
            output_file = open(file_path, "wb")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)

            header = []
            data = []
            for i in range(len(dataset_results[0][1])):
                data.append([])
                header.append(dataset_results[0][1][i][1])


            for (data_set, dataset_result) in dataset_results:
                i = 0
                for (result_arr, classifier_description) in dataset_result:
                    data[i] += result_arr[index[z]]
                    i += 1

            data = zip(*data)
            csv_writer.writerow(header)
            for data_slice in data:
                csv_writer.writerow(data_slice)

            file_paths.append(file_path)

        return metric, file_paths

