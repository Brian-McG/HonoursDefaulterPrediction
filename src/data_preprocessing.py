"""Applies data pre-processing to the datasets which applies one-hot encoding to categorical fields and scales numeric data to a normal distribution"""

import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer


def apply_preprocessing_to_train_test_dataset(input_defaulter_set, train_indices, test_indices, numerical_columns, categorical_columns, binary_columns, classification_label, missing_value_strategy,
                                              duplicate_removal_column=None, create_dummy_variables=True):
    """Applies pre-processing to a dataset split into training and testing indices"""
    for categorical_column in categorical_columns:
        input_defaulter_set[categorical_column] = input_defaulter_set[categorical_column].astype('category')
    for binary_column, true_val, false_val in binary_columns:
        try:
            mapping = {true_val: 1, false_val: 0}
            input_defaulter_set.replace({binary_column: mapping}, inplace=True)
        except TypeError as e:
            pass

    categorical_df_with_dummies = input_defaulter_set[categorical_columns].apply(lambda x: x.cat.codes)
    if len(categorical_columns) > 0 and create_dummy_variables:
        categorical_df_with_dummies = pd.get_dummies(input_defaulter_set[categorical_columns])

    numerical_train_df = pd.DataFrame()
    numerical_test_df = pd.DataFrame()
    if len(numerical_columns) > 0:
        numerical_train_df = input_defaulter_set.loc[train_indices][numerical_columns]
        numerical_test_df = input_defaulter_set.loc[test_indices][numerical_columns]
        scaler = preprocessing.StandardScaler().fit(numerical_train_df)
        scaled_numerical_train_arr = scaler.transform(numerical_train_df)
        scaled_numerical_test_arr = scaler.transform(numerical_test_df)

        for i in range(len(numerical_columns)):
            numerical_train_df[numerical_columns[i]] = scaled_numerical_train_arr.T[i]
            numerical_test_df[numerical_columns[i]] = scaled_numerical_test_arr.T[i]

    final_train_df = pd.concat([numerical_train_df, categorical_df_with_dummies.loc[train_indices], input_defaulter_set.loc[train_indices][[name for name, _, _ in binary_columns]],
                                input_defaulter_set.loc[train_indices][classification_label]], axis=1)
    final_test_df = pd.concat([numerical_test_df, categorical_df_with_dummies.loc[test_indices], input_defaulter_set.loc[test_indices][[name for name, _, _ in binary_columns]],
                               input_defaulter_set.loc[test_indices][classification_label]], axis=1)
    return final_train_df, final_test_df


def apply_preprocessing(input_defaulter_set, numerical_columns, categorical_columns, binary_columns, classification_label, missing_value_strategy, create_dummy_variables=True):
    """Applies pre-processing to an input data frame"""
    if missing_value_strategy == "remove":
        input_defaulter_set = input_defaulter_set[numerical_columns + categorical_columns + [name for name, _, _ in binary_columns] + classification_label]
        input_defaulter_set = input_defaulter_set.dropna(axis=0)
        input_defaulter_set = input_defaulter_set.reset_index(drop=True)

    categorical_df_with_dummies = pd.DataFrame()
    for categorical_column in categorical_columns:
        input_defaulter_set[categorical_column] = input_defaulter_set[categorical_column].astype('category')
    for binary_column, true_val, false_val in binary_columns:
        try:
            mapping = {true_val: 1, false_val: 0}
            input_defaulter_set.replace({binary_column: mapping}, inplace=True)
        except TypeError as e:
            pass
    categorical_df_with_dummies = input_defaulter_set[categorical_columns].apply(lambda x: x.cat.codes)
    if len(categorical_columns) > 0 and create_dummy_variables:
        categorical_df_with_dummies = pd.get_dummies(input_defaulter_set[categorical_columns])

    numerical_df = pd.DataFrame()
    if len(numerical_columns) > 0:
        numerical_input_df = input_defaulter_set[numerical_columns]
        scaled_numerical_arr = preprocessing.scale(numerical_input_df)
        for i in range(len(numerical_columns)):
            numerical_df[numerical_columns[i]] = scaled_numerical_arr.T[i]

    final_df = pd.concat([numerical_df, categorical_df_with_dummies, input_defaulter_set[[name for name, _, _ in binary_columns]], input_defaulter_set[classification_label]], axis=1)
    return final_df
