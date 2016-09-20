import subprocess

from sklearn.model_selection import StratifiedKFold

from config import constants as const
from classifier_statistics import ClassifierStatistics
from ml_technique import train_and_evaluate_fold, MLTechnique
from util import verbose_print


class GenericClassifier(MLTechnique):
    """Contains functionality to train and evaluate a classifier."""

    def __init__(self, classifier_class, classifier_parameters, data_balancer_class):
        self.ml_stats = ClassifierStatistics()
        self.classifier_class = classifier_class
        self.classifier_parameters = classifier_parameters
        self.data_balancer_class = data_balancer_class
        self.data_balancer = None

    def train_and_evaluate(self, defaulter_set, state):
        """Applies k-fold cross validation to train and evaluate a classifier"""

        if "defaulter_set" in self.classifier_parameters and self.classifier_parameters["defaulter_set"] is None:
            self.classifier_parameters["defaulter_set"] = defaulter_set

        classifier = self.classifier_class(**self.classifier_parameters)
        if self.data_balancer_class is not None:
            self.data_balancer = self.data_balancer_class(random_state=state)

        for i in range(const.RETRY_COUNT + 1):
            try:
                self.ml_stats.errors = []
                self.ml_stats.roc_list = []

                kf = StratifiedKFold(n_splits=const.NUMBER_OF_FOLDS, shuffle=True, random_state=state)
                index = 0

                for train, test in kf.split(defaulter_set.iloc[:, :-1].as_matrix(), defaulter_set.iloc[:, -1:].as_matrix().flatten()):
                    train_and_evaluate_fold(self, defaulter_set, train, test, classifier, index, data_balancer=self.data_balancer)
                    index += 1

                break
            except subprocess.CalledProcessError as e:
                if i + 1 > const.RETRY_COUNT:
                    raise e
                else:
                    print("INFO: Repeating classification step - attempt {0} of {1}".format(i + 1, const.RETRY_COUNT))

        # Error rates
        avg_accuracy_dict = self.ml_stats.calculate_average_predictive_accuracy()

        return avg_accuracy_dict
