from time import sleep

import matplotlib.pyplot as plt
import multiprocessing
import pandas as pd
from multiprocessing import Manager
from sknn.platform import cpu32, threading
from sknn.mlp import Classifier, Layer
from multiprocessing import Process

from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import ADASYN

import constants as const
from ml_technique import MLTechnique
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
                p = Process(target=self.train_and_evaluate_fold, args=(defaulter_set, (const.NUMBER_OF_FOLDS - remaining_runs) + i))
                process_pool.append(p)
                p.start()
                sleep(3)

            for process in process_pool:
                process.join()

            remaining_runs -= process_count

        # Error rates
        avg_accuracy_dict = self.ml_stats.calculate_average_predictive_accuracy()

        print("Average true positive rate:", avg_accuracy_dict["avg_true_positive_rate"])
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

    def train_and_evaluate_fold(self, defaulter_set, index):
        defaulter_set_len = defaulter_set.shape[0]

        # Prepare data set
        input_set = defaulter_set[const.CLASSIFICATION_FEATURES]
        output_set = defaulter_set[const.TREATMENT_OUTCOME]

        fold_len = defaulter_set_len / const.NUMBER_OF_FOLDS
        min_range = int(fold_len * index)
        max_range = int(fold_len * (index + 1))

        # Training data
        x_train_dataframe = pd.concat([input_set.iloc[0:min_range], input_set.iloc[max_range:defaulter_set_len]])
        y_train_dataframe = pd.concat([output_set.iloc[0:min_range], output_set.iloc[max_range:defaulter_set_len]])

        X_resampled, y_resampled = x_train_dataframe.as_matrix(), y_train_dataframe.as_matrix()

        # Apply SMOTE + ENN
        # sm = ADASYN()
        # X_resampled, y_resampled = sm.fit_sample(x_train_dataframe, y_train_dataframe)
        # X_res_vis = pca.transform(X_resampled)

        # Testing data
        test_dataframe = defaulter_set.iloc[min_range:max_range]

        # Assert that data is as expected
        assert (X_resampled.shape[0] == y_resampled.shape[0])
        # assert (test_dataframe.shape[0] == defaulter_set_len - X_resampled.shape[0])

        # Training fold specific statistics
        print("\n== Training Stats Fold {0} ==".format(index + 1))
        print("Number of rows for training fold {0}: ".format(index + 1), X_resampled.shape[0])
        print("Number of defaulters for training fold {0}: ".format(index + 1),
              y_resampled[y_resampled == 1].shape[0])

        # Train classifier
        nn = Classifier(layers=[Layer("Rectifier", units=10), Layer("Softmax")], learning_rate=0.001, n_iter=1000, n_stable=100)
        # callback={'on_epoch_finish': self.store_stats})
        nn.fit(X_resampled, y_resampled)

        # Testing fold specific statistics
        print("== Testing Stats Fold {0} ==".format(index + 1))
        print("Number of rows for training fold {0}: ".format(index + 1), test_dataframe.shape[0])
        print("Number of defaulters for training fold {0}: ".format(index + 1),
              test_dataframe[test_dataframe[const.TREATMENT_OUTCOME] == 1].shape[0])

        # Test accuracy
        test_classification = nn.predict(test_dataframe[const.CLASSIFICATION_FEATURES].as_matrix())
        actual_outcome = test_dataframe[const.TREATMENT_OUTCOME].as_matrix()

        self.ml_stats.calculate_and_append_fold_accuracy(test_classification, actual_outcome)


if __name__ == "__main__":
    input_defaulter_set = pd.DataFrame.from_csv("../data/Lima-TB-Treatment-v3.csv", index_col=None, encoding="UTF-8")
    ann = ArtificialNeuralNetwork()
    ann.train_and_evaluate(input_defaulter_set)
