import numpy as np
from sklearn.metrics import roc_curve

from util import verbose_print
from timeit import default_timer as timer


def train_and_evaluate_fold(generic_classifier, x_train, y_train, x_test, y_test, classifier, index, data_balancer=None):
    """Trains on input data and evaluates on test data for a specific classifier, can also apply data balancing using an input data_balancer."""
    if data_balancer is not None:
        x_train, y_train = data_balancer.fit_sample(x_train, y_train)

    # Training fold specific statistics
    verbose_print("\n== Training Stats Fold {0} ==".format(index + 1))
    verbose_print("Number of rows for training fold {0}: {1}".format(index + 1, x_train.shape[0]))
    verbose_print("Number of defaulters for training fold {0}: {1}".format(index + 1, y_train[y_train == 1].shape[0]))

    start_time = timer()
    classifier.fit(x_train, y_train)
    end_time = timer()
    fit_time = end_time - start_time

    # Testing fold specific statistics
    verbose_print("== Testing Stats Fold {0} ==".format(index + 1))
    verbose_print("Number of rows for training fold {0}: {1}".format(index + 1, len(y_test)))
    verbose_print("Number of defaulters for training fold {0}: {1}".format(index + 1, np.count_nonzero(y_test == 1)))

    # Test accuracy
    test_classification = classifier.predict(x_test)
    test_classification = np.array(test_classification)
    test_classification = test_classification.flatten()

    try:
        test_probabilities = classifier.predict_proba(x_test)
        if len(test_probabilities[0]) < 2:
            raise RuntimeError("test probabilities is not correct length")
    except Exception:
        test_probabilities = [[-1, -1]] * len(test_classification)

    outcome_decision_values = None
    try:
        predictions = classifier.predict_proba(x_test)
        outcome_decision_values = predictions[:, 1]
    except Exception as e:
        outcome_decision_values = None
        verbose_print("WARNING: unable to calculate classification accuracy - {0} - {1}".format(classifier.__class__.__name__, e))

    fpr, tpr = None, None
    if outcome_decision_values is not None:
        try:
            fpr, tpr, _ = roc_curve(y_test, outcome_decision_values)
            fpr = fpr.tolist()
            tpr = tpr.tolist()
        except Exception as e:
            print(e)

    generic_classifier.ml_stats.calculate_and_append_fold_accuracy(test_classification, y_test, tpr, fpr, fit_time, test_probabilities=test_probabilities)
