"""Contains all paramters required for each dataset"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import india_attrition_classifier_parameters
from config import india_attrition_data_balancer_only_parameters
from config import australian_data_balancer_only_parameters
from config import german_data_balancer_only_parameters
from config import lima_tb_classifier_parameters, default_classifier_parameters, german_classifier_parameters, australian_classifier_parameters
from config import lima_tb_data_balancer_only_parameters

data_set_arr = []


def append_data_set_details(data_set_path, numeric_columns, categorical_columns, binary_columns, classification_label, time_to_default, missing_values_strategy, status,
                            classifier_parameters, data_set_description, data_set_data_balancer_parameters, duplicate_removal_column, data_sets):
    """Adds data set path, data set status and data set description to data_set_arr as a dictionary"""
    data_sets.append({"data_set_path": data_set_path, "numeric_columns": numeric_columns, "categorical_columns": categorical_columns, "binary_columns": binary_columns, "classification_label": classification_label,
         "time_to_default": time_to_default, "missing_values_strategy": missing_values_strategy, "status": status, "data_set_classifier_parameters": classifier_parameters,
         "data_set_description": data_set_description, "data_set_data_balancer_parameters": data_set_data_balancer_parameters, "duplicate_removal_column": duplicate_removal_column})


# Lima TB data set
lima_tb_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/lima_tb/Lima TB Treatment Default Data.csv")
lima_tb_numeric_columns = []
lima_tb_categorical_columns = ["Age", "Sex", "Marital Status", "Hx of Tobacco Use", "Body Mass Index", "HIV Status"]
lima_tb_binary_columns = [("Prison Hx", "Yes", "No"), ("Completed Secondary Education", "Yes", "No"), ("Alcohol Use at Least Once Per Week", "Yes", "No"), ("History of Drug Use", "Yes", "No"),
                          ("Hx of Rehab", "Yes", "No"), ("MDR-TB", "Yes", "No"), ("Hx Chronic Disease", "Yes", "No"), ("Hx Diabetes Melitus", "Yes", "No")]
lima_tb_classification_label = ["Treatment Outcome"]
lima_tb_time_to_default = "Time to Default (Days)"
lima_tb_missing_values_strategy = "remove"
lima_tb_enabled = True
append_data_set_details(lima_tb_path, lima_tb_numeric_columns, lima_tb_categorical_columns, lima_tb_binary_columns, lima_tb_classification_label, lima_tb_time_to_default,
                        lima_tb_missing_values_strategy,
                        lima_tb_enabled,
                        lima_tb_classifier_parameters, "Lima_TB", lima_tb_data_balancer_only_parameters, None, data_set_arr)

# CHW Attrition
chw_attrition_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/chw_attrition/chw_data_india.csv")
chw_attrition_numeric_columns = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23", "X24", "X25",
                                 "X26", "X27", "X28", "X29", "X30", "X31", "X32", "X33", "X34", "X35", "X36", "X37", "X38", "X39", "X40", "X41", "X42", "X43", "X44", "X45", "X46", "X47", "X48", "X49",
                                 "X50", "X51", "X52", "X53", "X54", "X55", "X56", "X57", "X58", "X59", "X60", "X61", "X62", "X63", "X64", "X65", "X66", "X67", "X68", "X69", "X70", "X71", "X72", "X73",
                                 "X74", "X75", "X76", "X77", "X78", "X79", "X80", "X81", "X82", "X83", "X84", "X85", "X86", "X87", "X88", "X89", "X90"]
chw_attrition_categorical_columns = ["projectCode", "sector"]
chw_attrition_binary_columns = []
chw_attrition_classification_label = ["attritted"]
chw_attrition_time_to_default = None
chw_attrition_missing_values_strategy = "remove"
chw_attrition_enabled = False
append_data_set_details(chw_attrition_path, chw_attrition_numeric_columns, chw_attrition_categorical_columns, chw_attrition_binary_columns, chw_attrition_classification_label,
                        chw_attrition_time_to_default, chw_attrition_missing_values_strategy, chw_attrition_enabled, india_attrition_classifier_parameters, "India_CHW_attrition", india_attrition_data_balancer_only_parameters,
                        "userCode", data_set_arr)

# German credit data set
german_credit_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/german_finance/german_dataset.csv")
german_credit_numeric_columns = ["Attribute 2", "Attribute 5", "Attribute 8", "Attribute 11", "Attribute 13", "Attribute 16", "Attribute 18"]
german_credit_categorical_columns = ["Attribute 1", "Attribute 3", "Attribute 4", "Attribute 6", "Attribute 7", "Attribute 9", "Attribute 10", "Attribute 12", "Attribute 14",
                                     "Attribute 15", "Attribute 17"]
german_credit_binary_columns = [("Attribute 19", "A192", "A191"), ("Attribute 20", "A201", "A202")]
german_credit_classification_label = ["Classification Label"]
german_credit_time_to_default = None
german_credit_missing_values_strategy = "remove"
german_credit_enabled = True
german_credit_parameters = german_classifier_parameters
append_data_set_details(german_credit_path, german_credit_numeric_columns, german_credit_categorical_columns, german_credit_binary_columns, german_credit_classification_label,
                        german_credit_time_to_default, german_credit_missing_values_strategy,
                        german_credit_enabled, german_credit_parameters, "German_credit",
                        german_data_balancer_only_parameters, None, data_set_arr)

# Australian credit data set
australian_credit_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/australian_finance/australian.csv")
australian_credit_numeric_columns = ["A2", "A3", "A7", "A10", "A13", "A14"]
australian_credit_categorical_columns = ["A1", "A4", "A5", "A6", "A12"]
australian_credit_binary_columns = [("A8", "t", "f"), ("A9", "t", "f"), ("A11", "t", "f")]
australian_credit_classification_label = ["Classification Label"]
australian_credit_time_to_default = None
australian_credit_missing_values_strategy = "remove"
australian_credit_enabled = True
australian_credit_parameters = australian_classifier_parameters
append_data_set_details(australian_credit_path, australian_credit_numeric_columns, australian_credit_categorical_columns, australian_credit_binary_columns, australian_credit_classification_label,
                        australian_credit_time_to_default, australian_credit_missing_values_strategy,
                        australian_credit_enabled, australian_credit_parameters, "Australian_credit",
                        australian_data_balancer_only_parameters, None, data_set_arr)