"""Primary script used to execute the defaulter prediction"""

import os
import sys
from multiprocessing import Manager
from random import Random

import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

from config.constants import ANOVA_CHI2, BERNOULLI_NAIVE_BAYES, SVM_LINEAR, DECISION_TREE, RANDOM_FOREST
from config.constants import LOGISTIC_REGRESSION
from feature_selection.select_features_result_recorder import FeatureSelectionResultRecorder

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_preprocessing import apply_preprocessing, apply_preprocessing_to_train_test_dataset
from generic_classifier import GenericClassifier
from result_recorder import ResultRecorder
from run_statistics import RunStatistics
from util import verbose_print, get_number_of_processes_to_use, bcr_scorer
from config import data_sets
from config import constants as const
import visualisation as vis
import config.classifiers as cfr
import matplotlib.pyplot as plt

feature_selection_strategies = [None, LOGISTIC_REGRESSION, BERNOULLI_NAIVE_BAYES, SVM_LINEAR, RANDOM_FOREST]

const.TEST_REPEAT = 10


def select_features(input_defaulter_set, numeric_columns, categorical_columns, classification_label, classifier_parameters, random_state=None, selection_strategy=ANOVA_CHI2):
    if random_state is not None:
        random_state = (random_state + 885) % const.RANDOM_RANGE[1]

    X = pd.concat([input_defaulter_set[numeric_columns], input_defaulter_set[categorical_columns]], axis=1)
    y = input_defaulter_set[classification_label]
    if selection_strategy is None or selection_strategy == "None":
        return input_defaulter_set, numeric_columns, categorical_columns
    split_arr = selection_strategy.split("-")
    for selection_strategy_split in split_arr:
        if ANOVA_CHI2 == selection_strategy_split:
            X_numeric = pd.concat([input_defaulter_set[numeric_columns]], axis=1)
            X_categorical = pd.concat([input_defaulter_set[categorical_columns]], axis=1)
            y = input_defaulter_set[classification_label]

            if len(X_numeric.columns) > 0:
                p_vals_numeric = f_classif(X_numeric.as_matrix(), y.as_matrix().flatten())
            else:
                p_vals_numeric = [[], []]
            if len(X_categorical.columns) > 0:
                p_vals_categorical = chi2(X_categorical.as_matrix(), y.as_matrix().flatten())
            else:
                p_vals_categorical = [[], []]

            indices_usable = []
            indices_dropped = []
            p_threshold = 0.05
            for i in range(len(p_vals_numeric[0])):
                if p_vals_numeric[1][i] < p_threshold:
                    indices_usable.append(i)
                else:
                    indices_dropped.append(i)
            for i in range(len(p_vals_categorical[0])):
                if p_vals_categorical[1][i] < p_threshold:
                    indices_usable.append(i)
                else:
                    indices_dropped.append(i)
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        elif LOGISTIC_REGRESSION == selection_strategy_split:
            estimator = LogisticRegression(random_state=random_state, **classifier_parameters["Logistic regression"]["classifier_parameters"])
            selector = SelectFromModel(estimator, threshold="median")
            selector = selector.fit(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = selector.get_support(indices=True)
            y = X.columns.values
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        elif BERNOULLI_NAIVE_BAYES == selection_strategy_split:
            estimator = BernoulliNB(**classifier_parameters[BERNOULLI_NAIVE_BAYES]["classifier_parameters"])
            selector = SelectFromModel(estimator, threshold="median")
            selector = selector.fit(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = selector.get_support(indices=True)
            y = X.columns.values
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        elif SVM_LINEAR == selection_strategy_split:
            estimator = svm.SVC(random_state=random_state, C=7.371053, decision_function_shape="ovr", class_weight="balanced", kernel="linear")
            selector = SelectFromModel(estimator, threshold="median")
            selector = selector.fit(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = selector.get_support(indices=True)
            y = X.columns.values
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        elif DECISION_TREE == selection_strategy_split:
            estimator = DecisionTreeClassifier(random_state=random_state, class_weight="balanced")
            selector = SelectFromModel(estimator, threshold="median")
            selector = selector.fit(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = selector.get_support(indices=True)
            y = X.columns.values
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        elif RANDOM_FOREST == selection_strategy_split:
            forest = RandomForestClassifier(random_state=random_state, **classifier_parameters[RANDOM_FOREST]["classifier_parameters"])
            selector = SelectFromModel(forest, threshold="median")

            selector.fit(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = selector.get_support(indices=True)
            y = X.columns.values
            X = X[[X.columns.values[i] for i in range(len(X.columns.values)) if i in indices_usable]]

        else:
            raise RuntimeError("Unexpected selection_strategy - {0}".format(selection_strategy_split))

    new_numeric_columns = [numeric_column for numeric_column in numeric_columns if numeric_column in X.columns.values]
    new_categorical_columns = [categorical_column for categorical_column in categorical_columns if categorical_column in X.columns.values]
    return pd.concat([input_defaulter_set[X.columns.values], input_defaulter_set[classification_label]], axis=1), new_numeric_columns, new_categorical_columns


def execute_classifier_run(random_values, input_defaulter_set, numeric_columns, categorical_columns, classification_label, all_classifier_parameters, classifier_parameters, data_balancer, feature_selection_strategy, classifier_dict, classifier_description, roc_plot, result_recorder, missing_value_strategy):
    if classifier_dict["status"]:
        print("=== Executing {0} ===".format(classifier_description))
        test_stats = RunStatistics()
        for i in range(const.TEST_REPEAT):
            # Apply feature selection

            generic_classifier = GenericClassifier(classifier_dict["classifier"], classifier_parameters, data_balancer, random_values[i])
            kf = StratifiedKFold(n_splits=const.NUMBER_OF_FOLDS, shuffle=True, random_state=generic_classifier.k_fold_state)
            result_dictionary = None
            avg_features_selected = 0
            min_features_selected = sys.maxint
            max_features_selected = -sys.maxint - 1
            loop_count = 0
            for train, test in kf.split(input_defaulter_set.iloc[:, :-1], input_defaulter_set.iloc[:, -1:].as_matrix().flatten()):
                numeric_columns_with_dummy = apply_preprocessing(input_defaulter_set, numeric_columns, [], classification_label, missing_value_strategy,
                                                                 create_dummy_variables=True).columns[:-1]
                categorical_columns_with_dummy = apply_preprocessing(input_defaulter_set, [], categorical_columns, classification_label, missing_value_strategy,
                                                                     create_dummy_variables=True).columns[:-1]
                train_df, test_df = apply_preprocessing_to_train_test_dataset(input_defaulter_set, train, test, numeric_columns, categorical_columns, classification_label,
                                                                              missing_value_strategy, create_dummy_variables=True)

                train_df, _, _ = select_features(train_df, numeric_columns_with_dummy, categorical_columns_with_dummy, classification_label,
                                                                          all_classifier_parameters, random_state=random_values[i], selection_strategy=feature_selection_strategy)
                test_df = test_df[train_df.columns]

                number_of_features = len(train_df.columns) - 1
                avg_features_selected += number_of_features

                result_dictionary = generic_classifier.train_and_evaluate(train_df.iloc[:, :-1].as_matrix(), train_df.iloc[:, -1:].as_matrix().flatten(), test_df.iloc[:, :-1].as_matrix(), test_df.iloc[:, -1:].as_matrix().flatten())

                loop_count += 1

                if number_of_features < min_features_selected:
                    min_features_selected = number_of_features
                if number_of_features > max_features_selected:
                    max_features_selected = number_of_features

            test_stats.append_run_result(result_dictionary, generic_classifier.ml_stats.roc_list)

            avg_features_selected /= float(loop_count)
            print("{4} - Average features selected from folds - {0}, Min features selected - {1}, Max features selected - {2}, run - {3}".format(avg_features_selected, min_features_selected,
                                                                                                                                                 max_features_selected, i, feature_selection_strategy))
        avg_results = test_stats.calculate_average_run_accuracy()
        roc_plot.append((test_stats.roc_list, classifier_description))
        result_recorder.record_results(avg_results, classifier_description, feature_selection_strategy)
        print("=== Completed {0} ===".format(classifier_description))


def main():
    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")
            input_defaulter_set = input_defaulter_set[data_set["numeric_columns"] + data_set["categorical_columns"] + data_set["classification_label"]]
            input_defaulter_set = input_defaulter_set.dropna(axis=0)
            input_defaulter_set = input_defaulter_set.reset_index(drop=True)

            feature_selection_results_after = []

            result_recorder_after = FeatureSelectionResultRecorder()
            cpu_count = get_number_of_processes_to_use()

            random_values = []
            random = Random()
            for i in range(const.TEST_REPEAT):
                while True:
                    random_value = random.randint(const.RANDOM_RANGE[0], const.RANDOM_RANGE[1])
                    if random_value not in random_values:
                        random_values.append(random_value)
                        break

            for feature_selection_strategy in feature_selection_strategies:

                manager = Manager()
                feature_selection_result_recorder_after = FeatureSelectionResultRecorder(result_arr=manager.list())

                roc_plot = manager.list()

                # Execute enabled classifiers
                Parallel(n_jobs=cpu_count)(delayed(execute_classifier_run)(random_values, input_defaulter_set, data_set["numeric_columns"], data_set["categorical_columns"], data_set["classification_label"],
                                                                           data_set["data_set_classifier_parameters"].classifier_parameters,
                                                                           data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["classifier_parameters"],
                                                                           data_set["data_set_classifier_parameters"].classifier_parameters[classifier_description]["data_balancer"],
                                                                           feature_selection_strategy, classifier_dict, classifier_description, roc_plot,
                                                                           feature_selection_result_recorder_after, data_set["missing_values_strategy"]) for classifier_description, classifier_dict in cfr.classifiers.iteritems())

                feature_selection_result_recorder_after.results = sorted(feature_selection_result_recorder_after.results, key=lambda tup: tup[1])

                for (avg_results, classifier_description, feature_selection) in feature_selection_result_recorder_after.results:
                    result_recorder_after.record_results(avg_results, classifier_description, feature_selection)

                feature_selection_results_after.append((feature_selection_strategy, feature_selection_result_recorder_after.results, feature_selection_strategy))
            vis.plot_percentage_difference_graph(feature_selection_results_after, data_set["data_set_description"], name_suffix="_after")
            result_recorder_after.save_results_to_file(random_values, "select_features_after")


if __name__ == "__main__":
    # Run main
    main()
