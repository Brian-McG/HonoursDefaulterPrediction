import multiprocessing
from multiprocessing import Manager
from multiprocessing import Process
from time import sleep

from imblearn.combine import SMOTEENN
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
        self.ml_stats = MLStatistics(error_list)
        self.logical_cpu_count = multiprocessing.cpu_count()
        self.data_balancer = data_balancer

    def store_stats(self, avg_train_error, **_):
        """Stores average training error. Called at the end of each training iteration."""
        if const.TRAINING_ERROR not in self.errors[self.current_i]:
            self.errors[self.current_i][const.TRAINING_ERROR] = []
            self.errors[self.current_i]["training_error_count"] = 1
        self.errors[self.current_i][const.TRAINING_ERROR].append(avg_train_error)
        self.errors[self.current_i]["training_error_count"] += 1

    def train_and_evaluate(self, defaulter_set, hidden_layer='Rectifier', number_of_hidden_nodes=10, output_layer='Softmax'):
        """Applies k-fold cross validation to train and evaluate the ANN"""
        manager = Manager()
        self.ml_stats.errors = manager.list()

        number_of_concurrent_processes = min(const.NUMBER_OF_FOLDS, self.logical_cpu_count)
        remaining_runs = const.NUMBER_OF_FOLDS
        while remaining_runs > 0:
            process_pool = []
            process_count = min(number_of_concurrent_processes, remaining_runs)
            for i in range(process_count):
                for x in range(const.RETRY_COUNT):
                    try:
                        nn = Classifier(layers=[Layer(hidden_layer, units=number_of_hidden_nodes), Layer(output_layer)], learning_rate=0.001, n_iter=1000)
                        p = Process(target=train_and_evaluate_fold, args=(self, defaulter_set, (const.NUMBER_OF_FOLDS - remaining_runs) + i, nn, self.data_balancer))
                        p.start()
                        process_pool.append(p)
                        sleep(3)
                        break
                    except Exception:
                        if x + 1 >= const.RETRY_COUNT:
                            raise
                        sleep(1)

            for process in process_pool:
                process.join()

            remaining_runs -= process_count

        # Error rates
        avg_accuracy_dict = self.ml_stats.calculate_average_predictive_accuracy()

        verbose_print("\nAverage true positive rate: {0}".format(avg_accuracy_dict["avg_true_positive_rate"]))
        verbose_print("Average true negative rate: {0}".format(avg_accuracy_dict["avg_true_negative_rate"]))
        verbose_print("Average false positive rate: {0}".format(avg_accuracy_dict["avg_false_positive_rate"]))
        verbose_print("Average false negative rate: {0}".format(avg_accuracy_dict["avg_false_negative_rate"]))

        verbose_print("\nAverage true positive rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_true_positive_rate_with_prob_cutoff"]))
        verbose_print("Average true negative rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_true_negative_rate_with_prob_cutoff"]))
        verbose_print("Average false positive rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_false_positive_rate_with_prob_cutoff"]))
        verbose_print("Average false negative rate (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_false_negative_rate_with_prob_cutoff"]))
        verbose_print("Average unclassified (with cutoff {0}): {1}".format(const.CUTOFF_RATE, avg_accuracy_dict["avg_unclassified_with_prob_cutoff"]))

        return avg_accuracy_dict
