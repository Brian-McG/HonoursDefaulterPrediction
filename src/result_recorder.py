import csv
import os
from datetime import datetime

import constants as const
from constants import TITLE_ROW_WITH_TIME_TO_FIT, TITLE_ROW


class ResultRecorder:
    """Records results for the parameter comparision to file"""
    def __init__(self, result_arr=None):
        if result_arr is None:
            self.results = []
        else:
            self.results = result_arr

    def record_results(self, result_dict, classifier_dict):
        """Records an individual result"""
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
                                         result_tuple[0][2], result_tuple[0][15], result_tuple[0][28], result_tuple[0][3], result_tuple[0][4], result_tuple[0][5], result_tuple[0][6],
                                         result_tuple[0][20], result_tuple[0][21], result_tuple[0][13], result_tuple[0][22], result_tuple[0][29], result_tuple[0][24], result_tuple[0][25],
                                         result_tuple[0][12], result_tuple[0][23], random_values))
            else:
                csv_writer.writerow(TITLE_ROW)
                for result_tuple in self.results:
                    csv_writer.writerow((result_tuple[1], result_tuple[0][0], result_tuple[0][1],
                                         result_tuple[0][2], result_tuple[0][15], result_tuple[0][28], result_tuple[0][3], result_tuple[0][4], result_tuple[0][5], result_tuple[0][6],
                                         result_tuple[0][20], result_tuple[0][21], result_tuple[0][13], result_tuple[0][22], result_tuple[0][29], result_tuple[0][24], result_tuple[0][25],
                                         random_values))
            output_file.close()

    @staticmethod
    def save_results_for_multi_dataset(dataset_results, dataset="all_dataset"):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        metric = ["BACC", "AUC", "BS", "fit_time", "MCC", "hmeasure"]
        index = [14, 16, 17, 18, 19, 30]
        file_paths = []
        assert len(metric) == len(index)

        for z in range(len(metric)):
            file_name = "classifier_{2}_{1}_results_full_{0}.csv".format(current_time, metric[z], dataset)
            file_path = os.path.dirname(os.path.realpath(__file__)) + "/../results/" + file_name
            output_file = open(file_path, "wb")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)

            header = []
            data = []
            for i in range(len(dataset_results[0][1])):
                if -9999 not in dataset_results[0][1][i][0][index[z]]:
                    data.append([])
                    header.append(dataset_results[0][1][i][1])

            for (data_set, dataset_result) in dataset_results:
                i = 0
                for (result_arr, classifier_description) in dataset_result:
                    if -9999 not in result_arr[index[z]]:
                        data[i] += result_arr[index[z]]
                        i += 1

            data = zip(*data)
            csv_writer.writerow(header)
            for data_slice in data:
                csv_writer.writerow(data_slice)

            file_paths.append(file_path)

        return metric, file_paths
