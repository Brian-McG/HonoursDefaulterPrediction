from sklearn.model_selection import StratifiedKFold
import numpy as np
import constants as const
from fold_statistics import ClassifierStatistics
from data_preprocessing import apply_preprocessing_to_train_test_dataset
from sklearn.metrics import roc_curve
from util import verbose_print
from timeit import default_timer as timer


class GenericClassifier:
    """Contains functionality to train and evaluate a classifier."""

    def __init__(self, classifier_class, classifier_parameters, data_balancer_class, state):
        self.ml_stats = ClassifierStatistics()
        self.classifier_class = classifier_class
        self.classifier_parameters = classifier_parameters
        self.data_balancer_class = data_balancer_class
        self.data_balancer = None
        if state is not None:
            self.k_fold_state = state
            self.data_balancer_state = ((state * 3) % const.RANDOM_RANGE[1]) + const.RANDOM_RANGE[0]
            self.classifier_state = ((state + 5) % const.RANDOM_RANGE[1]) + const.RANDOM_RANGE[0]
        else:
            self.data_balancer_state = None
            self.k_fold_state = None
            self.classifier_state = None

    def k_fold_train_and_evaluate(self, defaulter_set, numerical_columns=None, categorical_columns=None, binary_columns=None, classification_label=None, missing_value_strategy=None,
                                  apply_preprocessing=False):
        """Applies k-fold cross validation to train and evaluate a classifier"""
        try:
            classifier = self.classifier_class(random_state=self.classifier_state, **self.classifier_parameters)
        except TypeError:
            classifier = self.classifier_class(**self.classifier_parameters)

        if self.data_balancer_class is not None:
            self.data_balancer = self.data_balancer_class(random_state=self.data_balancer_state)

        self.ml_stats.results = []
        self.ml_stats.roc_list = []

        kf = StratifiedKFold(n_splits=const.NUMBER_OF_FOLDS, shuffle=True, random_state=self.k_fold_state)
        index = 0

        for train, test in kf.split(defaulter_set.iloc[:, :-1].as_matrix(), defaulter_set.iloc[:, -1:].as_matrix().flatten()):
            if apply_preprocessing:
                train, test = apply_preprocessing_to_train_test_dataset(defaulter_set, train, test, numerical_columns, categorical_columns, binary_columns, classification_label,
                                                                        missing_value_strategy, create_dummy_variables=True)

            x_train, y_train = train.iloc[:, :-1].as_matrix(), train.iloc[:, -1:].as_matrix().flatten()
            x_test, y_test = test.iloc[:, :-1].as_matrix(), test.iloc[:, -1:].as_matrix().flatten()

            self.train_and_evaluate_fold(x_train, y_train, x_test, y_test, classifier, index, data_balancer=self.data_balancer)
            index += 1

        # Error rates
        avg_metric_dict = self.ml_stats.calculate_average_results()

        return avg_metric_dict

    def train_and_evaluate(self, x_train, y_train, x_test, y_test):
        """Trains on the input training set and evaluates on the test set.
           Returns a dictionary of metrics"""
        try:
            classifier = self.classifier_class(random_state=self.classifier_state, **self.classifier_parameters)
        except TypeError:
            classifier = self.classifier_class(**self.classifier_parameters)
        if self.data_balancer_class is not None:
            self.data_balancer = self.data_balancer_class(random_state=self.data_balancer_state)
        self.train_and_evaluate_fold(x_train, y_train, x_test, y_test, classifier, 0, data_balancer=self.data_balancer)

        # Error rates
        avg_metric_dict = self.ml_stats.calculate_average_results()

        return avg_metric_dict

    def train_and_evaluate_fold(self, x_train, y_train, x_test, y_test, classifier, index, data_balancer=None):
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

        self.ml_stats.calculate_and_append_fold_accuracy(test_classification, y_test, tpr, fpr, fit_time, test_probabilities=test_probabilities)
