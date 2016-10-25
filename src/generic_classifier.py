import subprocess

from sklearn.model_selection import StratifiedKFold

from classifier_statistics import ClassifierStatistics
from config import constants as const
from data_preprocessing import apply_preprocessing_to_train_test_dataset
from ml_technique import MLTechnique, train_and_evaluate_fold


class GenericClassifier(MLTechnique):
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

    def k_fold_train_and_evaluate(self, defaulter_set, numerical_columns=None, categorical_columns=None, classification_label=None, missing_value_strategy=None, apply_preprocessing=False):
        """Applies k-fold cross validation to train and evaluate a classifier"""
        try:
            classifier = self.classifier_class(random_state=self.classifier_state, **self.classifier_parameters)
        except TypeError:
            classifier = self.classifier_class(**self.classifier_parameters)

        if self.data_balancer_class is not None:
            self.data_balancer = self.data_balancer_class(random_state=self.data_balancer_state)

        for i in range(const.RETRY_COUNT + 1):
            try:
                self.ml_stats.results = []
                self.ml_stats.roc_list = []

                kf = StratifiedKFold(n_splits=const.NUMBER_OF_FOLDS, shuffle=True, random_state=self.k_fold_state)
                index = 0

                for train, test in kf.split(defaulter_set.iloc[:, :-1].as_matrix(), defaulter_set.iloc[:, -1:].as_matrix().flatten()):
                    if apply_preprocessing:
                        train, test = apply_preprocessing_to_train_test_dataset(defaulter_set, train, test, numerical_columns, categorical_columns, classification_label, missing_value_strategy, create_dummy_variables=True)

                    x_train, y_train = train.iloc[:, :-1].as_matrix(), train.iloc[:, -1:].as_matrix().flatten()
                    x_test, y_test = test.iloc[:, :-1].as_matrix(), test.iloc[:, -1:].as_matrix().flatten()

                    train_and_evaluate_fold(self, x_train, y_train, x_test, y_test, classifier, index, data_balancer=self.data_balancer)
                    index += 1

                break
            except subprocess.CalledProcessError as e:
                if i + 1 > const.RETRY_COUNT:
                    raise e
                else:
                    print("INFO: Repeating classification step - attempt {0} of {1}".format(i + 1, const.RETRY_COUNT))

        # Error rates
        avg_accuracy_dict = self.ml_stats.calculate_average_results()

        return avg_accuracy_dict

    def train_and_evaluate(self, x_train, y_train, x_test, y_test):
        try:
            classifier = self.classifier_class(random_state=self.classifier_state, **self.classifier_parameters)
        except TypeError:
            classifier = self.classifier_class(**self.classifier_parameters)
        if self.data_balancer_class is not None:
            self.data_balancer = self.data_balancer_class(random_state=self.data_balancer_state)
        train_and_evaluate_fold(self, x_train, y_train, x_test, y_test, classifier, 0, data_balancer=self.data_balancer)

        # Error rates
        avg_accuracy_dict = self.ml_stats.calculate_average_results()

        return avg_accuracy_dict


