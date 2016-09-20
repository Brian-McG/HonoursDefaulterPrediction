import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np


def apply_preprocessing(input_defaulter_set, numerical_columns, categorical_columns, classification_label, missing_value_strategy):
    if missing_value_strategy == "remove":
        input_defaulter_set = input_defaulter_set[numerical_columns + categorical_columns + classification_label]
        input_defaulter_set = input_defaulter_set.dropna(axis=0)

    elif missing_value_strategy == "mean/most_frequent" or missing_value_strategy == "median/most_frequent":
        numerical_values_filled = []
        if len(numerical_columns) > 1:
            strategy = ""
            if missing_value_strategy == "mean/most_frequent":
                strategy = "mean"
            elif missing_value_strategy == "median/most_frequent":
                strategy = "median"
            imp_numerical = Imputer(missing_values='NaN', strategy=strategy, axis=0)
            imp_numerical.fit(input_defaulter_set[numerical_columns])
            numerical_values_filled = imp_numerical.transform(input_defaulter_set[numerical_columns])

        numerical_categorises_filled = []
        if len(categorical_columns) > 1:
            for categorical_column in categorical_columns:
                input_defaulter_set[categorical_column] = input_defaulter_set[categorical_column].astype('category')
            categorical_df = input_defaulter_set[categorical_columns]
            numerical_categorises = categorical_df.apply(lambda x: x.cat.codes)
            imp_categorical = Imputer(missing_values=-1, strategy='most_frequent', axis=0)
            imp_categorical.fit(numerical_categorises)
            numerical_categorises_filled = imp_categorical.transform(numerical_categorises)

        final_df = input_defaulter_set[classification_label]
        for i in range(len(numerical_columns)):
            final_df.insert(len(final_df.columns) - 1, numerical_columns[i], numerical_values_filled.T[i])

        for i in range(len(categorical_columns)):
            final_df.insert(len(final_df.columns) - 1, categorical_columns[i], pd.Series(data=numerical_categorises_filled.T[i], dtype="category"))

        input_defaulter_set = final_df

    else:
        raise RuntimeError("{0} is an invalid missing value strategy, only delete, mean/most_frequent and median/most_frequent are supported.".format(missing_value_strategy))

    categorical_df_with_dummies = pd.DataFrame()
    if len(categorical_columns) > 0:
        categorical_df_with_dummies = pd.get_dummies(input_defaulter_set[categorical_columns])

    numerical_df = pd.DataFrame()
    if len(numerical_columns) > 0:
        numerical_input_df = input_defaulter_set[numerical_columns]
        scaled_numerical_arr = preprocessing.scale(numerical_input_df)
        for i in range(len(numerical_columns)):
            numerical_df[numerical_columns[i]] = scaled_numerical_arr.T[i]

    final_df = pd.concat([numerical_df, categorical_df_with_dummies, input_defaulter_set[classification_label]], axis=1)
    return final_df
