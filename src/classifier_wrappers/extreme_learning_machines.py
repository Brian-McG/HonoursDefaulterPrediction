import numpy as np
from hpelm import ELM


class ExtremeLearningMachine:
    """Wraps the hpelm ELM class to have the same interface as sci-kit learn"""

    def __init__(self, layers=None, random_state=None):
        self.layers = layers
        if layers is None:
            self.layers = [(100, 'sigm')]

        if random_state is None:
            self.random_state = np.random.RandomState()
        else:
            self.random_state = np.random.RandomState(random_state)


        self.elm = None

    def fit(self, x_train, class_1):
        """Fits x_resampled and y_resampled to create a classification model."""
        input_len = len(x_train[0])
        self.elm = ELM(len(x_train[0]), 2, "c")

        # Apply the same process as in the hpelm codebase but do it via the RandomState instance so results are repeatable
        for layer_tuple in self.layers:
            if layer_tuple[1] != "lin":
                w = self.random_state.randn(input_len, layer_tuple[0])
                if layer_tuple[1] not in ("rbf_l1", "rbf_l2", "rbf_linf"):
                    w *= 3.0 / input_len**0.5  # high dimensionality fix
                b = self.random_state.randn(layer_tuple[0])
                if layer_tuple[1] in ("rbf_l2", "rbf_l1", "rbf_linf"):
                    b = np.abs(b)
                    b *= input_len
            else:
                w = None
                b = None
            self.elm.add_neurons(layer_tuple[0], layer_tuple[1], W=w, B=b)

        class_0 = np.array([0 if item == 1 else 1 for item in class_1])
        class_arr = [None] * len(class_1)
        for i in range(len(class_1)):
            class_arr[i] = [class_0[i], class_1[i]]
        class_arr = np.array(class_arr)
        return self.elm.train(x_train, class_arr, "CV" "OP", 'c', k=10)

    def predict(self, x_test):
        """Predicts y for x_test"""
        prediction_weights = self.elm.predict(x_test)
        return np.array([1 if item[1] > item[0] else 0 for item in prediction_weights])

    def predict_proba(self, x_test):
        """Predicts probability of positive class for x_test"""
        return self.elm.predict(x_test)
