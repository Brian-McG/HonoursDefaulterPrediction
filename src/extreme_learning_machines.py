import numpy as np
from hpelm import ELM


class ExtremeLearningMachine:
    """Wraps the hpelm ELM class to have the same interface as sci-kit learn"""

    def __init__(self, defaulter_set, layers=None):
        if layers is None:
            layers = [(20, "sigm"), (3, "rbf_l2")]

        self.elm = ELM(defaulter_set.shape[1] - 1, 2, "c")
        for layer_tuple in layers:
            self.elm.add_neurons(layer_tuple[0], layer_tuple[1])

    def fit(self, x_train, y_train):
        y_resampled2 = np.array([0 if item == 1 else 1 for item in y_train])
        class_arr = [None] * len(y_train)
        for i in range(len(y_train)):
            class_arr[i] = [y_train[i], y_resampled2[i]]
        class_arr = np.array(class_arr)
        return self.elm.train(x_train, class_arr, "CV" "OP", 'wc', k=10)

    def predict(self, x_test):
        prediction_weights = self.elm.predict(x_test)
        return np.array([1 if item[0] > item[1] else 0 for item in prediction_weights])
