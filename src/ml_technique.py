from abc import ABC, abstractmethod

import constants as const
import pandas as pd

from visualisation import visualise_two_data_sets


def train_and_evaluate_fold(self, defaulter_set, index, classifier, data_balancer=None):
    defaulter_set_len = defaulter_set.shape[0]

    # Prepare data set
    input_set = defaulter_set.iloc[:, :-1]
    output_set = defaulter_set.iloc[:, -1:]

    fold_len = defaulter_set_len / const.NUMBER_OF_FOLDS
    min_range = int(fold_len * index)
    max_range = int(fold_len * (index + 1))

    # Training data
    x_train_dataframe = pd.concat([input_set.iloc[0:min_range], input_set.iloc[max_range:defaulter_set_len]])
    y_train_dataframe = pd.concat([output_set.iloc[0:min_range], output_set.iloc[max_range:defaulter_set_len]])

    # Apply Data balancing
    x_resampled, y_resampled = x_train_dataframe.as_matrix(), y_train_dataframe.as_matrix().ravel()

    if data_balancer is not None:
        x_resampled, y_resampled = data_balancer.fit_sample(x_resampled, y_resampled)

    # Visualise the two data-sets
    visualise_two_data_sets(x_train_dataframe.as_matrix(), y_train_dataframe.as_matrix().ravel(), x_resampled, y_resampled)

    # Testing data
    test_dataframe = defaulter_set.iloc[min_range:max_range]

    # Training fold specific statistics
    print("\n== Training Stats Fold {0} ==".format(index + 1))
    print("Number of rows for training fold {0}: ".format(index + 1), x_resampled.shape[0])
    print("Number of defaulters for training fold {0}: ".format(index + 1),
          y_resampled[y_resampled == 1].shape[0])

    classifier.fit(x_resampled, y_resampled)

    # Testing fold specific statistics
    print("== Testing Stats Fold {0} ==".format(index + 1))
    print("Number of rows for training fold {0}: ".format(index + 1), test_dataframe.shape[0])
    print("Number of defaulters for training fold {0}: ".format(index + 1),
          test_dataframe[test_dataframe[test_dataframe.columns[-1]] == 1].shape[0])

    # Test accuracy
    test_classification = classifier.predict(test_dataframe[test_dataframe.columns[:-1]].as_matrix())
    actual_outcome = test_dataframe[test_dataframe.columns[-1]].as_matrix()

    self.ml_stats.calculate_and_append_fold_accuracy(test_classification, actual_outcome)


class MLTechnique(ABC):
    """An abstract class that encapsulates the concept of a machine learning technique"""

    @abstractmethod
    def train_and_evaluate(self, defaulter_set):
        pass
