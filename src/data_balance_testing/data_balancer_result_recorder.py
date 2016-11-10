import csv
import os
from datetime import datetime

import constants as const
from constants import TITLE_ROW_BALANCER_RESULT


class DataBalancerResultRecorder:
    def __init__(self, result_arr=None):
        if result_arr is None:
            self.results = []
        else:
            self.results = result_arr

    def record_results(self, result_dict, classifier_dict, feature_selection_strategy):
        self.results.append((result_dict, classifier_dict, feature_selection_strategy))

    def save_results_to_file(self, random_values, data_set_description):
        """Records results to file. If file_name is None, then a default filename of data_<number of folds>_<timestamp>.csv"""
        if len(self.results) > 0:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = "{0}_data_{1}-folds_{2}.csv".format(data_set_description, const.NUMBER_OF_FOLDS, current_time)
            output_file = open(os.path.dirname(os.path.realpath(__file__)) + "/../../results/" + file_name, "wb")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
            title_row = TITLE_ROW_BALANCER_RESULT
            csv_writer.writerow(title_row)
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

    @staticmethod
    def save_results_for_multi_dataset(dataset_results, dataset="all_dataset"):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        metric = ["BACC"]
        index = [14]
        file_paths = []

        for z in range(len(metric)):
            file_name = "select_features_{2}_{1}_results_full_{0}.csv".format(current_time, metric[z], dataset)
            file_path = os.path.dirname(os.path.realpath(__file__)) + "/../../results/" + file_name
            output_file = open(file_path, "wb")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)

            header = []
            data = []
            print(dataset_results)
            for i in range(len(dataset_results[0][1][0][1])):
                data.append([])
                header.append(dataset_results[0][1][0][1][i][0] if dataset_results[0][1][0][1][i][0] is not None else "None")

            print(header)
            for dataset_result in dataset_results:
                i = 0
                for y in range(len(dataset_result[1])):
                    for x in range(len(dataset_result[1][y][1])):
                        print(dataset_result[1][y][1][x])
                        data[i] += dataset_result[1][y][1][x][index[z]]
                    i += 1

            data = zip(*data)
            csv_writer.writerow(header)
            for data_slice in data:
                csv_writer.writerow(data_slice)

            file_paths.append(file_path)

        return metric, file_paths
