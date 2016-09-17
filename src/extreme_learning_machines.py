import multiprocessing
import os
import random
import sys
from multiprocessing import Manager
from multiprocessing import Process
from time import sleep

from hpelm import ELM
from sklearn.cross_validation import StratifiedKFold

import constants as const
from constants import verbose_print
from ml_statistics import MLStatistics
from ml_technique import MLTechnique, train_and_evaluate_fold


class ExtremeLearningMachine(MLTechnique):
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

    def store_stats(self, avg_train_error, **_):
        """Stores average training error. Called at the end of each training iteration."""
        if const.TRAINING_ERROR not in self.errors[self.current_i]:
            self.errors[self.current_i][const.TRAINING_ERROR] = []
            self.errors[self.current_i]["training_error_count"] = 1
        self.errors[self.current_i][const.TRAINING_ERROR].append(avg_train_error)
        self.errors[self.current_i]["training_error_count"] += 1

    def train_and_evaluate(self, defaulter_set, state=0):
        """Applies k-fold cross validation to train and evaluate the ANN"""

        if self.data_balancer is not None:
            self.data_balancer = self.data_balancer(random_state=state)

        self.ml_stats.errors = []
        self.ml_stats.roc_list = []
        kf = StratifiedKFold(defaulter_set.iloc[:, -1:].as_matrix().flatten(), n_folds=const.NUMBER_OF_FOLDS, shuffle=True, random_state=state)
        index = 0
        for train, test in kf:
            elm = ELM(defaulter_set.shape[1] - 1, 2, "c")
            elm.add_neurons(20, "sigm")
            elm.add_neurons(3, "rbf_l2")
            train_and_evaluate_fold(self, defaulter_set, train, test, elm, index, data_balancer=self.data_balancer)
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
