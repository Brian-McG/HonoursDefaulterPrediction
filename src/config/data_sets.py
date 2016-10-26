"""Data sets to test against"""
import os

from config import australian_data_balancer_only_parameters
from config import german_data_balancer_only_parameters
from config import lima_tb_classifier_parameters,  default_classifier_parameters, german_classifier_parameters, australian_classifier_parameters
from config import lima_tb_data_balancer_only_parameters
from config.constants import LOGISTIC_REGRESSION, BERNOULLI_NAIVE_BAYES, SVM_LINEAR, DECISION_TREE, RANDOM_FOREST, ANOVA_CHI2

data_set_arr = []


def append_data_set_details(data_set_path, numeric_columns, categorical_columns, classification_label, time_to_default, missing_values_strategy, feature_selection_strategy, status, classifier_parameters,
                            data_set_description, data_set_data_balancer_parameters, duplicate_removal_column,
                            data_sets):
    """Adds data set path, data set status and data set description to data_set_arr as a dictionary"""
    data_sets.append({"data_set_path": data_set_path, "numeric_columns": numeric_columns, "categorical_columns": categorical_columns, "classification_label": classification_label,
                      "time_to_default": time_to_default, "missing_values_strategy": missing_values_strategy,
                      "feature_selection_strategy": feature_selection_strategy, "status": status, "data_set_classifier_parameters": classifier_parameters,
                      "data_set_description": data_set_description, "data_set_data_balancer_parameters": data_set_data_balancer_parameters, "duplicate_removal_column": duplicate_removal_column})


# Lima TB data set
lima_tb_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/lima_tb/Lima TB Treatment Default Data.csv")
lima_tb_numeric_columns = []
lima_tb_categorical_columns = ["Age", "Sex", "Marital Status", "Prison Hx", "Completed Secondary Education", "Hx of Tobacco Use", "Alcohol Use at Least Once Per Week",
                               "History of Drug Use", "Hx of Rehab", "MDR-TB", "Body Mass Index", "Hx Chronic Disease", "HIV Status", "Hx Diabetes Melitus"]
lima_tb_classification_label = ["Treatment Outcome"]
lima_tb_time_to_default = "Time to Default (Days)"
lima_tb_missing_values_strategy = "remove"
lima_tb_feature_selection_strategy = None
lima_tb_enabled = False
append_data_set_details(lima_tb_path, lima_tb_numeric_columns, lima_tb_categorical_columns, lima_tb_classification_label, lima_tb_time_to_default, lima_tb_missing_values_strategy,
                        lima_tb_feature_selection_strategy, lima_tb_enabled,
                        lima_tb_classifier_parameters, "Lima_TB", lima_tb_data_balancer_only_parameters, None, data_set_arr)

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
german_credit_time_to_default = None
german_credit_missing_values_strategy = "remove"
german_credit_feature_selection_strategy = None
german_credit_enabled = False
german_credit_parameters = german_classifier_parameters
append_data_set_details(german_credit_path, german_credit_numeric_columns, german_credit_categorical_columns, german_credit_classification_label, german_credit_time_to_default,
                        german_credit_missing_values_strategy, german_credit_feature_selection_strategy, german_credit_enabled, german_credit_parameters, "German_credit", german_data_balancer_only_parameters, None, data_set_arr)

# Australian credit data set
australian_credit_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/australian_finance/australian.csv")
australian_credit_numeric_columns = ["A2", "A3", "A7", "A10", "A13", "A14"]
australian_credit_categorical_columns = ["A1", "A4", "A5", "A6", "A8", "A9", "A11", "A12"]
australian_credit_classification_label = ["Classification Label"]
australian_credit_time_to_default = None
australian_credit_missing_values_strategy = "remove"
australian_credit_feature_selection_strategy = None
australian_credit_enabled = False
australian_credit_parameters = australian_classifier_parameters
append_data_set_details(australian_credit_path, australian_credit_numeric_columns, australian_credit_categorical_columns, australian_credit_classification_label,
                        australian_credit_time_to_default,
                        australian_credit_missing_values_strategy, australian_credit_feature_selection_strategy, australian_credit_enabled, australian_credit_parameters, "Australian_credit", australian_data_balancer_only_parameters, None, data_set_arr)

# CHW Attrition
chw_attrition_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/chw_attrition/chw_data_india.csv")
chw_attrition_numeric_columns = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23", "X24", "X25", "X26", "X27", "X28", "X29", "X30", "X31", "X32", "X33", "X34", "X35", "X36", "X37", "X38", "X39", "X40", "X41", "X42", "X43", "X44", "X45", "X46", "X47", "X48", "X49", "X50", "X51", "X52", "X53", "X54", "X55", "X56", "X57", "X58", "X59", "X60", "X61", "X62", "X63", "X64", "X65", "X66", "X67", "X68", "X69", "X70", "X71", "X72", "X73", "X74", "X75", "X76", "X77", "X78", "X79", "X80", "X81", "X82", "X83", "X84", "X85", "X86", "X87", "X88", "X89", "X90"]
chw_attrition_categorical_columns = ["projectCode", "sector"]
chw_attrition_classification_label = ["attritted"]
chw_attrition_time_to_default = None
chw_attrition_missing_values_strategy = "remove"
chw_attrition_feature_selection_strategy = None
chw_attrition_enabled = True
chw_attrition_parameters = australian_classifier_parameters
append_data_set_details(chw_attrition_path, chw_attrition_numeric_columns, chw_attrition_categorical_columns, chw_attrition_classification_label,
                        chw_attrition_time_to_default,
                        chw_attrition_missing_values_strategy, chw_attrition_feature_selection_strategy, chw_attrition_enabled, lima_tb_classifier_parameters, "India_CHW_attrition", lima_tb_data_balancer_only_parameters, "userCode", data_set_arr)


# # HIV
# hiv_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/STEPS data PLOS ONE.csv")
# hiv_numeric_columns = ["CD4 cell ct", "Age"]
# hiv_categorical_columns = ["WHO Clinical Staging", "Gender", "Education", "Work Status", "Income", "Cell phone?", "Dist 0=near, 1=far, ", "Tobacco Use", "Alcohol Usage"]
# hiv_classification_label = ["Total retained in care"]
# hiv_time_to_default = None
# hiv_missing_values_strategy = "remove"
# hiv_enabled = False
# hiv_classifier_parameters = default_classifier_parameters
# append_data_set_details(hiv_path, hiv_numeric_columns, hiv_categorical_columns, hiv_classification_label,
#                         hiv_time_to_default,
#                         hiv_missing_values_strategy, hiv_enabled,
#                         hiv_classifier_parameters, "HIV data", data_set_arr)
#
# # Heart
# heart_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/heart.csv")
# heart_numeric_columns = ["1", "4", "5", "8", "10", "12", "11"]
# heart_categorical_columns = ["2", "6", "9", "7", "3", "13"]
# heart_classification_label = ["classification"]
# heart_time_to_default = None
# heart_missing_values_strategy = "remove"
# heart_enabled = False
# heart_classifier_parameters = default_classifier_parameters
# append_data_set_details(heart_path, heart_numeric_columns, heart_categorical_columns, heart_classification_label,
#                         heart_time_to_default,
#                         heart_missing_values_strategy, heart_enabled,
#                         heart_classifier_parameters, "Heart", data_set_arr)
#
# # Parkinsons
# parkinsons_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/parkinsons.csv")
# parkinsons_numeric_columns = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]
# parkinsons_categorical_columns = []
# parkinsons_classification_label = ["status"]
# parkinsons_time_to_default = None
# parkinsons_missing_values_strategy = "remove"
# parkinsons_enabled = False
# parkinsons_classifier_parameters = default_classifier_parameters
# append_data_set_details(parkinsons_path, parkinsons_numeric_columns, parkinsons_categorical_columns, parkinsons_classification_label,
#                         parkinsons_time_to_default,
#                         parkinsons_missing_values_strategy, parkinsons_enabled,
#                         parkinsons_classifier_parameters, "Parkinsons", data_set_arr)

