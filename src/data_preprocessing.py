import pandas as pd
from sklearn import preprocessing


def apply_preprocessing(input_defaulter_set):
    dummy_set = pd.get_dummies(input_defaulter_set.iloc[:, :-1])
    min_max_scaler = preprocessing.MinMaxScaler()
    dummy_set = pd.DataFrame(min_max_scaler.fit_transform(dummy_set))
    return dummy_set.assign(label=input_defaulter_set.iloc[:, -1:])