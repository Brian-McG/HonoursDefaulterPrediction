import numpy as np

# List of classifier information
generic_classifier_parameter_arr = []
non_generic_classifier_parameter_arr = []

# Non generic classifiers
# Artificial neural network
ann_parameter_dict = {"hidden_layer": ['Rectifier', 'Sigmoid', 'Tanh', 'ExpLin'], "number_of_hidden_nodes": [1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 50, 75, 100],
                      "output_layer": ['Softmax']}
non_generic_classifier_parameter_arr.append(ann_parameter_dict)

# Generic classifiers
# Support Vector Machines (RBF)
rbf_svm_parameter_dict = {"cache_size": [2048], "decision_function_shape": ['ovo', "ovr"], "kernel": ["rbf"], "C": np.linspace(0.01, 10, 100).tolist(),
                          "gamma": ["auto"] + np.linspace(0.001, 2, 100).tolist(), 'class_weight': [None, 'balanced']}
generic_classifier_parameter_arr.append(rbf_svm_parameter_dict)

# Support Vector Machines (linear)
linear_svm_parameter_dict = {"cache_size": [2048], "decision_function_shape": ['ovo', "ovr"], "kernel": ["linear"], "C": np.linspace(0.01, 10, 100).tolist(),
                             "gamma": ["auto"] + np.linspace(0.001, 2, 100).tolist(), 'class_weight': [None, 'balanced']}
generic_classifier_parameter_arr.append(linear_svm_parameter_dict)

# Support Vector Machines (poly)
poly_svm_parameter_dict = {"cache_size": [2048], "decision_function_shape": ['ovo', "ovr"], "kernel": ["poly"], "C": np.linspace(0.01, 10, 100).tolist(),
                           "gamma": ["auto"] + np.linspace(0.001, 2, 100).tolist(), 'class_weight': [None, 'balanced'], 'degree': [1, 2, 3, 4, 5],
                           'coef0': np.linspace(0, 5, 20).tolist()}
generic_classifier_parameter_arr.append(poly_svm_parameter_dict)

# Logistic regression
logistic_regression_dict = {"C": np.linspace(0.01, 10, 100).tolist(), "fit_intercept": [True, False], "intercept_scaling": np.linspace(0.01, 10, 100).tolist(), "class_weight": [None, "balanced"], "solver": ["newton-cg", "lbfgs", "liblinear", "sag"]}
generic_classifier_parameter_arr.append(logistic_regression_dict)

# Decision tree
decision_tree_dict = {"criterion": ["gini", "entropy"], "splitter": ["best", "random"], "max_features": ["auto", "sqrt", "log2"], "class_weight": [None, "balanced"]}
generic_classifier_parameter_arr.append(decision_tree_dict)
