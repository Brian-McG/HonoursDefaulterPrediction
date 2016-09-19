from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.metrics import roc_curve

from constants import verbose_print


def train_and_evaluate_fold(self, defaulter_set, training_set_indices, testing_set_indices, classifier, index, data_balancer=None):
    # Prepare data set
    input_set = defaulter_set.iloc[:, :-1].as_matrix()
    output_set = defaulter_set.iloc[:, -1:].as_matrix().flatten()

    # Apply Data balancing
    x_resampled, y_resampled = input_set[training_set_indices], output_set[training_set_indices]

    if data_balancer is not None:
        x_resampled, y_resampled = data_balancer.fit_sample(x_resampled, y_resampled)

    # Visualise the two data-sets
    # visualise_two_data_sets(x_train_dataframe.as_matrix(), y_train_dataframe.as_matrix().ravel(), x_resampled, y_resampled)

    # Training fold specific statistics
    verbose_print("\n== Training Stats Fold {0} ==".format(index + 1))
    verbose_print("Number of rows for training fold {0}: {1}".format(index + 1, x_resampled.shape[0]))
    verbose_print("Number of defaulters for training fold {0}: {1}".format(index + 1, y_resampled[y_resampled == 1].shape[0]))

    classifier.fit(x_resampled, y_resampled)

    # Testing set
    x_testing = input_set[testing_set_indices]
    y_testing = output_set[testing_set_indices]

    # Testing fold specific statistics
    verbose_print("== Testing Stats Fold {0} ==".format(index + 1))
    verbose_print("Number of rows for training fold {0}: {1}".format(index + 1, len(y_testing)))
    verbose_print("Number of defaulters for training fold {0}: {1}".format(index + 1, np.count_nonzero(y_testing == 1)))

    # Test accuracy
    test_classification = classifier.predict(x_testing)
    test_classification = np.array(test_classification)
    test_classification = test_classification.flatten()

    try:
        test_probabilities = classifier.predict_proba(x_testing)
    except AttributeError:
        test_probabilities = [[-1, -1]] * len(test_classification)

    outcome_decision_values = None
    if "SVC" in classifier.__class__.__name__:
        try:
            outcome_decision_values = classifier.decision_function(x_testing)
        except AttributeError:
            pass

    if outcome_decision_values is None:
        try:
            predictions = classifier.predict_proba(x_testing)
            outcome_decision_values = predictions[:, 1]
            print(np.array([[predictions[i][0], predictions[i][1], y_testing[i], test_classification[i]] for i in range(len(predictions))]))
        except AttributeError:
            outcome_decision_values = None
            verbose_print("WARNING: unable to calculate classification accuracy")

    fpr, tpr = None, None
    if outcome_decision_values is not None:
        fpr, tpr, _ = roc_curve(y_testing, outcome_decision_values)
        fpr = fpr.tolist()
        tpr = tpr.tolist()

    self.ml_stats.calculate_and_append_fold_accuracy(test_classification, y_testing, tpr, fpr, test_probabilities=test_probabilities)


class MLTechnique:
    """An abstract class that encapsulates the concept of a machine learning technique"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def train_and_evaluate(self, defaulter_set, state):
        pass
