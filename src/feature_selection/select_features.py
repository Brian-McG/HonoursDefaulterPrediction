"""Primary script used to execute the defaulter prediction"""

import os
import sys
from multiprocessing import Manager

import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_preprocessing import apply_preprocessing
from generic_classifier import GenericClassifier
from result_recorder import ResultRecorder
from run_statistics import RunStatistics
from util import verbose_print, get_number_of_processes_to_use
from config import data_sets
from config import constants as const
import visualisation as vis
import config.classifiers as cfr

ANOVA = "Anova"
CHI2 = "chi2"
LOGISTIC_REGRESSION = "Logistic regression"
BERNOULLI_NAIVE_BAYES = "Bernoulli Naive Bayes"
SVM_LINEAR = "SVM (linear)"
DECISION_TREE = "Decision Tree"
RANDOM_FOREST = "Random forest"

feature_selection_strategies = [None, ANOVA, CHI2, LOGISTIC_REGRESSION, BERNOULLI_NAIVE_BAYES, SVM_LINEAR, DECISION_TREE, RANDOM_FOREST]


def select_features(input_defaulter_set, numeric_columns, categorical_columns, classification_label, data_set_classifier_parameters, selection_strategy="ANOVA"):
    X = pd.concat([input_defaulter_set[numeric_columns], input_defaulter_set[categorical_columns]], axis=1)
    y = input_defaulter_set[classification_label]
    if selection_strategy is None or selection_strategy == "None":
        return input_defaulter_set, numeric_columns, categorical_columns
    split_arr = selection_strategy.split("-")
    for selection_strategy_split in split_arr:
        if ANOVA == selection_strategy_split:
            p_vals = f_classif(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = []
            indices_dropped = []
            for i in range(len(p_vals[0])):
                if p_vals[1][i] < 0.05:
                    indices_usable.append(i)
                else:
                    indices_dropped.append(i)
            verbose_print("Dropped indices: {0}".format(indices_dropped))
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        elif CHI2 == selection_strategy_split:
            p_vals = chi2(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = []
            indices_dropped = []
            for i in range(len(p_vals[0])):
                if p_vals[1][i] < 0.05:
                    indices_usable.append(i)
                else:
                    indices_dropped.append(i)
            verbose_print("Dropped indices: {0}".format(indices_dropped))
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        elif LOGISTIC_REGRESSION == selection_strategy_split:
            estimator = LogisticRegression(**data_set_classifier_parameters.classifier_parameters["Logistic regression"]["classifier_parameters"])
            selector = SelectFromModel(estimator, threshold=0.25)
            selector = selector.fit(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = selector.get_support(indices=True)
            y = X.columns.values
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        elif BERNOULLI_NAIVE_BAYES == selection_strategy_split:
            estimator = BernoulliNB(**data_set_classifier_parameters.classifier_parameters[BERNOULLI_NAIVE_BAYES]["classifier_parameters"])
            selector = SelectFromModel(estimator, threshold=0.25)
            selector = selector.fit(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = selector.get_support(indices=True)
            y = X.columns.values
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        elif SVM_LINEAR == selection_strategy_split:
            estimator = svm.SVC(**data_set_classifier_parameters.classifier_parameters[SVM_LINEAR]["classifier_parameters"])
            selector = SelectFromModel(estimator, threshold=0.00002)
            selector = selector.fit(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = selector.get_support(indices=True)
            y = X.columns.values
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        elif DECISION_TREE == selection_strategy_split:
            estimator = DecisionTreeClassifier(**data_set_classifier_parameters.classifier_parameters[DECISION_TREE]["classifier_parameters"])
            selector = SelectFromModel(estimator, threshold=0.00002)
            selector = selector.fit(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = selector.get_support(indices=True)
            y = X.columns.values
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        elif RANDOM_FOREST == selection_strategy_split:
            forest = RandomForestClassifier(**data_set_classifier_parameters.classifier_parameters[RANDOM_FOREST]["classifier_parameters"])
            selector = SelectFromModel(forest, threshold=0.05)
            selector.fit(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = selector.get_support(indices=True)
            y = X.columns.values
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        else:
            raise RuntimeError("Unexpected selection_strategy - {0}".format(selection_strategy_split))

    new_numeric_columns = [numeric_column for numeric_column in numeric_columns if numeric_column in X.columns.values]
    new_categorical_columns = [categorical_column for categorical_column in categorical_columns if categorical_column in X.columns.values]
    return pd.concat([input_defaulter_set[X.columns.values], input_defaulter_set[classification_label]], axis=1), new_numeric_columns, new_categorical_columns


def execute_classifier_run(input_defaulter_set, classifier_parameters, data_balancer, data_set_description, classifier_dict, classifier_description, roc_plot, result_recorder):
    if classifier_dict["status"]:
        print("=== Executing {0} ===".format(classifier_description))
        test_stats = RunStatistics()
        for i in range(500):
            generic_classifier = GenericClassifier(classifier_dict["classifier"], classifier_parameters, data_balancer)
            result_dictionary = generic_classifier.k_fold_train_and_evaluate(input_defaulter_set, i)
            test_stats.append_run_result(result_dictionary, generic_classifier.ml_stats.roc_list)

        avg_results = test_stats.calculate_average_run_accuracy()
        roc_plot.append((test_stats.roc_list, classifier_description))
        result_recorder.record_results(avg_results, classifier_description)
        print("=== Completed {0} ===".format(classifier_description))


def main():
    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")

            input_defaulter_set = apply_preprocessing(input_defaulter_set, data_set["numeric_columns"], data_set["categorical_columns"], data_set["classification_label"], data_set["missing_values_strategy"], create_dummy_variables=False)

            feature_selection_results = []

            for feature_selection_strategy in feature_selection_strategies:
                # Apply feature selection
                new_defaulter_set, numeric_columns, categorical_columns = select_features(input_defaulter_set, data_set["numeric_columns"], data_set["categorical_columns"], data_set["classification_label"], data_set["data_set_classifier_parameters"], selection_strategy=feature_selection_strategy)

                # Preprocess data set
                new_defaulter_set = apply_preprocessing(new_defaulter_set, numeric_columns, categorical_columns, data_set["classification_label"], data_set["missing_values_strategy"], create_dummy_variables=True)
                manager = Manager()
                result_recorder = ResultRecorder(result_arr=manager.list())

                roc_plot = manager.list()

                cpu_count = get_number_of_processes_to_use()
                # Execute enabled classifiers
                Parallel(n_jobs=cpu_count)(delayed(execute_classifier_run)(new_defaulter_set, data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["classifier_parameters"], data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["data_balancer"], data_set["data_set_description"], classifier_dict, classifier_description, roc_plot, result_recorder) for classifier_description, classifier_dict in cfr.classifiers.iteritems())

                result_recorder.results = sorted(result_recorder.results, key=lambda tup: tup[1])
                for (result_arr, classifier_description) in result_recorder.results:
                    print("\n=== {0} ===".format(classifier_description))
                    print("Matthews correlation coefficient: {0}".format(result_arr[0]))
                    print("Cohen Kappa score: {0}".format(result_arr[1]))
                    print("Average true rate: {0}".format(result_arr[2]))
                    print("Average true positive rate: {0}".format(result_arr[3]))
                    print("Average true negative rate: {0}".format(result_arr[4]))
                    print("Average false positive rate: {0}".format(result_arr[5]))
                    print("Average false negative rate: {0}".format(result_arr[6]))

                feature_selection_results.append((feature_selection_strategy, result_recorder.results))
            vis.plot_percentage_difference_on_feature_selection(feature_selection_results)




if __name__ == "__main__":
    # Run main
    main()

