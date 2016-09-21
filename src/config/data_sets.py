"""Data sets to test against"""
import os

from config import default_classifier_parameters

data_set_arr = []


def append_data_set_details(data_set_path, numeric_columns, categorical_columns, classification_label, missing_values_strategy, status, classifier_parameters, data_set_description,
                            data_sets):
    """Adds data set path, data set status and data set description to data_set_arr as a dictionary"""
    data_sets.append({"data_set_path": data_set_path, "numeric_columns": numeric_columns, "categorical_columns": categorical_columns, "classification_label": classification_label,
                      "missing_values_strategy": missing_values_strategy, "status": status, "data_set_classifier_parameters": classifier_parameters,
                      "data_set_description": data_set_description})


# Lima TB data set
lima_tb_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/lima_tb/Lima TB Treatment Default Data.csv")
lima_tb_numeric_columns = []
lima_tb_categorical_columns = ["Age", "Sex", "Marital Status", "Prison Hx", "Completed Secondary Education", "Hx of Tobacco Use", "Alcohol Use at Least Once Per Week",
                               "History of Drug Use", "Hx of Rehab", "MDR-TB", "Body Mass Index", "Hx Chronic Disease", "HIV Status", "Hx Diabetes Melitus"]
lima_tb_classification_label = ["Treatment Outcome"]
lima_tb_missing_values_strategy = "remove"
lima_tb_enabled = True
lima_tb_classifier_parameters = default_classifier_parameters
append_data_set_details(lima_tb_path, lima_tb_numeric_columns, lima_tb_categorical_columns, lima_tb_classification_label, lima_tb_missing_values_strategy, lima_tb_enabled,
                        lima_tb_classifier_parameters, "Lima_TB", data_set_arr)

# # Baobab data set
# baobab_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "<TODO>")
# baobab_enabled = False
# baobab_classifier_parameters = default_classifier_parameters
# append_data_set_details(baobab_path, baobab_enabled, baobab_classifier_parameters, "Baobab", data_set_arr)
#
# German credit data set
german_credit_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/german_finance/german_dataset.csv")
german_credit_numeric_columns = ["Attribute 2", "Attribute 5", "Attribute 8", "Attribute 11", "Attribute 13", "Attribute 16", "Attribute 18"]
german_credit_categorical_columns = ["Attribute 1", "Attribute 3", "Attribute 4", "Attribute 6", "Attribute 7", "Attribute 9", "Attribute 10", "Attribute 12", "Attribute 14",
                                     "Attribute 15", "Attribute 17", "Attribute 19", "Attribute 20"]
german_credit_classification_label = ["Classification Label"]
german_credit_missing_values_strategy = "remove"
german_credit_enabled = False
german_credit_parameters = default_classifier_parameters
append_data_set_details(german_credit_path, german_credit_numeric_columns, german_credit_categorical_columns, german_credit_classification_label,
                        german_credit_missing_values_strategy, german_credit_enabled, german_credit_parameters, "German_credit", data_set_arr)

# Australian credit data set
australian_credit_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/australian_finance/australian.csv")
australian_credit_numeric_columns = ["A2", "A3", "A7", "A10", "A13", "A14"]
australian_credit_categorical_columns = ["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12"]
australian_credit_classification_label = ["Classification Label"]
australian_credit_missing_values_strategy = "remove"
australian_credit_enabled = False
australian_credit_parameters = default_classifier_parameters
append_data_set_details(australian_credit_path, australian_credit_numeric_columns, australian_credit_categorical_columns, australian_credit_classification_label,
                        australian_credit_missing_values_strategy, australian_credit_enabled, australian_credit_parameters, "Australian_credit", data_set_arr)
