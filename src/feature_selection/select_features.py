"""Primary script used to execute the defaulter prediction"""

import os
import sys

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression

from config.default_classifier_parameters import logistic_regression_parameters
from data_preprocessing import apply_preprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import data_sets
from config import constants as const


def main():
    for data_set in data_sets.data_set_arr:
        if data_set["status"]:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")
            input_defaulter_set = apply_preprocessing(input_defaulter_set, data_set["numeric_columns"], data_set["categorical_columns"], data_set["classification_label"],
                                                      data_set["missing_values_strategy"], create_dummy_variables=False)
            X = pd.concat([input_defaulter_set[data_set["numeric_columns"]], input_defaulter_set[data_set["categorical_columns"]]], axis=1)
            y = input_defaulter_set[data_set["classification_label"]]

            X_new = f_classif(X.as_matrix(), y.as_matrix().flatten())
            indices_usable = []
            indices_dropped = []
            for i in range(len(X_new[0])):
                print("{0:f}, {1:f}".format(X_new[0][i], X_new[1][i]))
                if X_new[1][i] < 0.05:
                    indices_usable.append(i)
                else:
                    indices_dropped.append(i)

            X_2 = input_defaulter_set[input_defaulter_set.columns[indices_usable]]
            feature_selection = SelectFromModel(LogisticRegression(**logistic_regression_parameters))
            feature_selection.fit(X_2.as_matrix(), y.as_matrix().flatten())
            print(indices_usable)
            print(indices_dropped)
            usable_indices = feature_selection.get_support()
            final_indices = []
            for i in range(len(indices_usable)):
                if usable_indices[i]:
                    final_indices.append(indices_usable[i])
            print(input_defaulter_set[input_defaulter_set.columns[final_indices]].head())

            if const.RECORD_RESULTS:
                pass


if __name__ == "__main__":
    # Run main
    main()
