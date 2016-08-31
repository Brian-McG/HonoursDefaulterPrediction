from time import sleep

import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
from multiprocessing import Manager

from imblearn.combine import SMOTEENN
from sklearn import preprocessing
from sknn.platform import cpu32, threading
from sknn.mlp import Classifier, Layer
from multiprocessing import Process
from sklearn.preprocessing import Imputer

import constants as const
from data_preprocessing import apply_preprocessing
from ml_technique import MLTechnique, train_and_evaluate_fold
from ml_statistics import MLStatistics


class ArtificialNeuralNetwork(MLTechnique):
    """Contains functionality to train and evaluate an artificial neural network (ANN)"""
    def __init__(self):
        manager = Manager()
        self.errors = manager.list(range(const.NUMBER_OF_FOLDS))
        for error in self.errors:
            error = {}

        self.current_i = None
        error_list = manager.list()
        self.ml_stats = MLStatistics(error_list)
        self.logical_cpu_count = multiprocessing.cpu_count()

    def store_stats(self, avg_train_error, **_):
        """Stores average training error. Called at the end of each training iteration."""
        if const.TRAINING_ERROR not in self.errors[self.current_i]:
            self.errors[self.current_i][const.TRAINING_ERROR] = []
            self.errors[self.current_i]["training_error_count"] = 1
        self.errors[self.current_i][const.TRAINING_ERROR].append(avg_train_error)
        self.errors[self.current_i]["training_error_count"] += 1

    def train_and_evaluate(self, defaulter_set):
        """Applies k-fold cross validation to train and evaluate the ANN"""

        number_of_concurrent_processes = min(const.NUMBER_OF_FOLDS, self.logical_cpu_count)
        remaining_runs = const.NUMBER_OF_FOLDS
        while remaining_runs > 0:
            process_pool = []
            process_count = min(number_of_concurrent_processes, remaining_runs)
            for i in range(process_count):
                data_balancer = None
                nn = Classifier(layers=[Layer("Rectifier", units=10), Layer("Softmax")], learning_rate=0.001, n_iter=1000, n_stable=100)
                p = Process(target=train_and_evaluate_fold, args=(self, defaulter_set, (const.NUMBER_OF_FOLDS - remaining_runs) + i, nn, data_balancer))
                process_pool.append(p)
                p.start()
                sleep(3)

            for process in process_pool:
                process.join()

            remaining_runs -= process_count

        # Error rates
        avg_accuracy_dict = self.ml_stats.calculate_average_predictive_accuracy()

        print("\Average true positive rate:", avg_accuracy_dict["avg_true_positive_rate"])
        print("Average true negative rate:", avg_accuracy_dict["avg_true_negative_rate"])
        print("Average false positive rate:", avg_accuracy_dict["avg_false_positive_rate"])
        print("Average false negative rate:", avg_accuracy_dict["avg_false_negative_rate"])

        # Plot training error
        # error_map = {}
        # for i in range(const.NUMBER_OF_FOLDS):
        #     error_map[i + 1] = self.errors[i][const.TRAINING_ERROR]
        #
        # error_df = pd.DataFrame({k: pd.Series(v) for k, v in error_map.items()})
        # error_df.plot()
        # plt.show()

if __name__ == "__main__":
    input_defaulter_set = pd.DataFrame.from_csv("../data/lima_tb/Lima-TB-Treatment-base.csv", index_col=None, encoding="UTF-8")

    #input_defaulter_set = pd.DataFrame.from_csv("../data/german_finance/german_dataset_numberised.csv", index_col=None, encoding="UTF-8")
    #input_defaulter_set = pd.DataFrame.from_csv("../data/australian_finance/australian_dataset.csv", index_col=None, encoding="UTF-8")
    #input_defaulter_set = pd.DataFrame.from_csv("../data/credit_screening/credit_screening_data_fill_missing_data.csv", index_col=None, encoding="UTF-8")

    input_defaulter_set = apply_preprocessing(input_defaulter_set)
    ann = ArtificialNeuralNetwork()
    ann.train_and_evaluate(input_defaulter_set)
