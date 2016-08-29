import pandas as pd
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn import preprocessing

import constants as const
from ml_statistics import MLStatistics
from ml_technique import MLTechnique


class SupportVectorMachine(MLTechnique):
    """Contains functionality to train and evaluate a support vector machine (SVM)."""

    def __init__(self):
        self.current_i = None
        self.ml_stats = MLStatistics()

    @staticmethod
    def apply_standardization(series):
        if series.name == const.TREATMENT_OUTCOME:
            return series
        else:
            min_max_scaler = preprocessing.MinMaxScaler()
            return min_max_scaler.fit_transform(preprocessing.scale(series))

    def train_and_evaluate(self, defaulter_set):
        """Applies k-fold cross validation to train and evaluate the SVM"""
        defaulter_set_len = defaulter_set.shape[0]
        defaulter_set = defaulter_set[const.CLASSIFICATION_FEATURES + [const.TREATMENT_OUTCOME]]
        # defaulter_set = defaulter_set.apply(self.apply_standardization)

        # Prepare data set
        input_set = defaulter_set[const.CLASSIFICATION_FEATURES]
        output_set = defaulter_set[const.TREATMENT_OUTCOME]

        # Apply k-fold cross validation
        fold_len = defaulter_set_len / const.NUMBER_OF_FOLDS
        for i in range(const.NUMBER_OF_FOLDS):
            self.current_i = i
            min_range = int(fold_len * i)
            max_range = int(fold_len * (i + 1))

            # Training data
            x_train_dataframe = pd.concat([input_set.iloc[0:min_range], input_set.iloc[max_range:defaulter_set_len]])
            y_train_dataframe = pd.concat([output_set.iloc[0:min_range], output_set.iloc[max_range:defaulter_set_len]])

            # Testing data
            test_dataframe = defaulter_set.iloc[min_range:max_range]

            # Assert that data is as expected
            # assert (x_train_dataframe.shape[0] == y_train_dataframe.shape[0])
            # assert (test_dataframe.shape[0] == defaulter_set_len - x_train_dataframe.shape[0])

            svm = NuSVC(kernel="rbf", gamma=0.5, nu=0.01)
            svm.fit(x_train_dataframe.as_matrix(), y_train_dataframe.as_matrix())

            # Test accuracy
            test_classification = svm.predict(test_dataframe[const.CLASSIFICATION_FEATURES].as_matrix())

            actual_outcome = test_dataframe[const.TREATMENT_OUTCOME].as_matrix()

            self.ml_stats.calculate_and_append_fold_accuracy(test_classification, actual_outcome)

        # Error rates
        avg_accuracy_dict = self.ml_stats.calculate_average_predictive_accuracy()

        print("Average true positive rate:", avg_accuracy_dict["avg_true_positive_rate"])
        print("Average true negative rate:", avg_accuracy_dict["avg_true_negative_rate"])
        print("Average false positive rate:", avg_accuracy_dict["avg_false_positive_rate"])
        print("Average false negative rate:", avg_accuracy_dict["avg_false_negative_rate"])


if __name__ == "__main__":
    input_defaulter_set = pd.DataFrame.from_csv("../data/Lima-TB-Treatment-v2.csv", index_col=None, encoding="UTF-8")
    svm_imp = SupportVectorMachine()
    svm_imp.train_and_evaluate(input_defaulter_set)