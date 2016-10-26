import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Imputer


def apply_preprocessing_to_train_test_dataset(input_defaulter_set, train_indices, test_indices, numerical_columns, categorical_columns, classification_label, missing_value_strategy, duplicate_removal_column=None, create_dummy_variables=True):

    for categorical_column in categorical_columns:
        input_defaulter_set[categorical_column] = input_defaulter_set[categorical_column].astype('category')

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

    final_train_df = pd.concat([numerical_train_df, categorical_df_with_dummies.loc[train_indices], input_defaulter_set.loc[train_indices][classification_label]], axis=1)
    final_test_df = pd.concat([numerical_test_df, categorical_df_with_dummies.loc[test_indices], input_defaulter_set.loc[test_indices][classification_label]], axis=1)
    return final_train_df, final_test_df


def apply_preprocessing(input_defaulter_set, numerical_columns, categorical_columns, classification_label, missing_value_strategy, create_dummy_variables=True):
    if missing_value_strategy == "remove":
        input_defaulter_set = input_defaulter_set[numerical_columns + categorical_columns + classification_label]
        input_defaulter_set = input_defaulter_set.dropna(axis=0)
        input_defaulter_set = input_defaulter_set.reset_index(drop=True)

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
    for categorical_column in categorical_columns:
        input_defaulter_set[categorical_column] = input_defaulter_set[categorical_column].astype('category')
    categorical_df_with_dummies = input_defaulter_set[categorical_columns].apply(lambda x: x.cat.codes)
    if len(categorical_columns) > 0 and create_dummy_variables:
        categorical_df_with_dummies = pd.get_dummies(input_defaulter_set[categorical_columns])

    numerical_df = pd.DataFrame()
    if len(numerical_columns) > 0:
        numerical_input_df = input_defaulter_set[numerical_columns]
        scaled_numerical_arr = preprocessing.scale(numerical_input_df)
        for i in range(len(numerical_columns)):
            numerical_df[numerical_columns[i]] = scaled_numerical_arr.T[i]

    final_df = pd.concat([numerical_df, categorical_df_with_dummies, input_defaulter_set[classification_label]], axis=1)
    return final_df
