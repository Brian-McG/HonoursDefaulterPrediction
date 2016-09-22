import numpy as np
from hpelm import ELM


class ExtremeLearningMachine:
    """Wraps the hpelm ELM class to have the same interface as sci-kit learn"""

    def __init__(self, number_of_features, layers=None):
        if layers is None:
            layers = [(20, "sigm"), (3, "rbf_l2")]

        self.elm = ELM(number_of_features, 2, "c")
        for layer_tuple in layers:
            self.elm.add_neurons(layer_tuple[0], layer_tuple[1])

    def fit(self, x_train, class_1):
        # TODO generalise this to multiple classes
        class_0 = np.array([0 if item == 1 else 1 for item in class_1])
        class_arr = [None] * len(class_1)
        for i in range(len(class_1)):
            class_arr[i] = [class_0[i], class_1[i]]
        class_arr = np.array(class_arr)
        return self.elm.train(x_train, class_arr, "CV" "OP", 'c', k=10)

    def predict(self, x_test):
        prediction_weights = self.elm.predict(x_test)
        return np.array([1 if item[1] > item[0] else 0 for item in prediction_weights])

    def predict_proba(self, x_test):
        return self.elm.predict(x_test)
