import multiprocessing
import os
from multiprocessing import Manager
from multiprocessing import Process
from time import sleep

import random

import sys
from imblearn.combine import SMOTEENN
from sklearn.cross_validation import StratifiedKFold
from sknn.mlp import Classifier, Layer

import constants as const
from constants import verbose_print
from ml_statistics import MLStatistics
from ml_technique import MLTechnique, train_and_evaluate_fold


class ArtificialNeuralNetwork(MLTechnique):
    """Contains functionality to train and evaluate an artificial neural network (ANN)"""

    def __init__(self, data_balancer):
        manager = Manager()
        self.errors = manager.list(range(const.NUMBER_OF_FOLDS))
        for error in self.errors:
            error = {}

        self.current_i = None
        error_list = manager.list()
        roc_list = manager.list()
        self.ml_stats = MLStatistics(error_list, roc_list)
        self.logical_cpu_count = multiprocessing.cpu_count()
        self.data_balancer = data_balancer

    def train_and_evaluate_fold_with_failover(self, defaulter_set, training_indices, testing_indices, classifier, index, data_balancer=None):
        for x in range(const.RETRY_COUNT):
            try:
                old_stdout = sys.stdout
                f = open(os.devnull, 'w')
                sys.stdout = f
                train_and_evaluate_fold(self, defaulter_set, training_indices, testing_indices, classifier, index, data_balancer)
                sys.stdout = old_stdout
                return
            except Exception:
                if x + 1 >= const.RETRY_COUNT:
                    raise
                sleep_time = random.uniform(1, 3)
                sleep(sleep_time)
                continue

    def store_stats(self, avg_train_error, **_):
        """Stores average training error. Called at the end of each training iteration."""
        if const.TRAINING_ERROR not in self.errors[self.current_i]:
            self.errors[self.current_i][const.TRAINING_ERROR] = []
            self.errors[self.current_i]["training_error_count"] = 1
        self.errors[self.current_i][const.TRAINING_ERROR].append(avg_train_error)
        self.errors[self.current_i]["training_error_count"] += 1

    def train_and_evaluate(self, defaulter_set, hidden_layer='Rectifier', number_of_hidden_nodes=75, output_layer='Softmax', number_of_threads=-1, state=0):
        """Applies k-fold cross validation to train and evaluate the ANN"""

        if number_of_threads == -1:
            number_of_threads = self.logical_cpu_count

        if number_of_threads > 1:
            manager = Manager()
            self.ml_stats.errors = manager.list()
            self.ml_stats.roc_list = manager.list()

            number_of_concurrent_processes = min(const.NUMBER_OF_FOLDS, number_of_threads)
            remaining_runs = const.NUMBER_OF_FOLDS
            kf = StratifiedKFold(defaulter_set.iloc[:, -1:].as_matrix().flatten(), n_folds=const.NUMBER_OF_FOLDS, shuffle=state)
            kf = list(kf)

            while remaining_runs > 0:
                process_pool = []
                process_count = min(number_of_concurrent_processes, remaining_runs)
                for i in range(process_count):
                    nn = Classifier(layers=[Layer(hidden_layer, units=number_of_hidden_nodes), Layer(output_layer)], learning_rate=0.001, n_iter=1000)
                    index = (const.NUMBER_OF_FOLDS - remaining_runs) + i
                    training_indices, testing_indices = kf[index]
                    p = Process(target=self.train_and_evaluate_fold_with_failover, args=(defaulter_set, training_indices, testing_indices, nn, index, self.data_balancer))
                    p.start()
                    process_pool.append(p)

                for process in process_pool:
                    process.join()

                remaining_runs -= process_count
        else:
            self.ml_stats.errors = []
            self.ml_stats.roc_list = []
            kf = StratifiedKFold(defaulter_set.iloc[:, -1:].as_matrix().flatten(), n_folds=const.NUMBER_OF_FOLDS, shuffle=True)
            index = 0
            for train, test in kf:
                nn = Classifier(layers=[Layer(hidden_layer, units=number_of_hidden_nodes), Layer(output_layer)], learning_rate=0.001, n_iter=1000)
                self.train_and_evaluate_fold_with_failover(defaulter_set, train, test, nn, index, self.data_balancer)
                index += 1

        # Error rates
        avg_accuracy_dict = self.ml_stats.calculate_average_predictive_accuracy()

        verbose_print("\nAverage true rate: {0}".format((avg_accuracy_dict["avg_true_positive_rate"] + avg_accuracy_dict["avg_true_negative_rate"]) / 2.0))
        verbose_print("Average true positive rate: {0}".format(avg_accuracy_dict["avg_true_positive_rate"]))
        verbose_print("Average true negative rate: {0}".format(avg_accuracy_dict["avg_true_negative_rate"]))
        verbose_print("Average false positive rate: {0}".format(avg_accuracy_dict["avg_false_positive_rate"]))
        verbose_print("Average false negative rate: {0}".format(avg_accuracy_dict["avg_false_negative_rate"]))

        verbose_print("\nAverage true positive rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_true_positive_rate_with_prob_cutoff"]))
        verbose_print("Average true negative rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_true_negative_rate_with_prob_cutoff"]))
        verbose_print("Average false positive rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_false_positive_rate_with_prob_cutoff"]))
        verbose_print("Average false negative rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_false_negative_rate_with_prob_cutoff"]))
        verbose_print("Average unclassified (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_unclassified_with_prob_cutoff"]))

        return avg_accuracy_dict
