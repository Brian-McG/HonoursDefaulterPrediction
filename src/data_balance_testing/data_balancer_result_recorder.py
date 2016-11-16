import csv
import os
from collections import OrderedDict
from datetime import datetime

import constants as const
from config import visualisation_input
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
        metric = ["BACC", "hmeasure"]
        index = [14, 30]
        file_paths = []
        classifier_skip = ["AdaBoost", "Decision Tree", "Logistic regression", "Random forest", "SVM (RBF)", "SVM (linear)", "SVM (polynomial)"]
        data_set_skip = ["Lima_TB", "German_credit", "Australian_credit"]

        for z in range(len(metric)):
            file_name = "select_features_{2}_{1}_results_full_{0}.csv".format(current_time, metric[z], dataset)
            file_path = os.path.dirname(os.path.realpath(__file__)) + "/../../results/" + file_name
            output_file = open(file_path, "wb")
            csv_writer = csv.writer(output_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)

            header = []
            data = []
            for i in range(len(dataset_results[0][1][0][1])):
                data.append([])
                header.append(dataset_results[0][1][0][1][i][0] if dataset_results[0][1][0][1][i][0] is not None else "None")

            print(header)
            balancer_result = OrderedDict()
            for (data_set, dataset_result) in dataset_results:
                if data_set not in data_set_skip:
                    print(data_set)
                    for (classifier_description, result_arr) in dataset_result:
                        if classifier_description not in classifier_skip:
                            for (balancer_description, results) in result_arr:
                                if balancer_description in balancer_result:
                                    balancer_result[balancer_description] = balancer_result[balancer_description] + results[index[z]]
                                else:
                                    balancer_result[balancer_description] = results[index[z]]

            balancer_index = 0
            for key, value in balancer_result.iteritems():
                assert key == header[balancer_index]
                data[balancer_index] = value
                balancer_index += 1

            data = zip(*data)
            csv_writer.writerow(header)
            for data_slice in data:
                csv_writer.writerow(data_slice)

            file_paths.append(file_path)

        return metric, file_paths

if __name__ == "__main__":
    DataBalancerResultRecorder.save_results_for_multi_dataset(visualisation_input.results)