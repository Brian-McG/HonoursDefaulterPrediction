"""Data sets to test against"""
import os

from config import default_classifier_parameters

data_set_arr = []


def append_data_set_details(data_set_path, status, classifier_parameters, data_set_description, data_sets):
    """Adds data set path, data set status and data set description to data_set_arr as a dictionary"""
    data_sets.append({"data_set_path": data_set_path, "status": status, "data_set_classifier_parameters": classifier_parameters, "data_set_description": data_set_description})


# Lima TB data set
lima_tb_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/lima_tb/Lima-TB-Treatment-base.csv")
lima_tb_enabled = True
lima_tb_classifier_parameters = default_classifier_parameters
append_data_set_details(lima_tb_path, lima_tb_enabled, lima_tb_classifier_parameters, "Lima_TB", data_set_arr)

# Baobab data set
baobab_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "<TODO>")
baobab_enabled = False
baobab_classifier_parameters = default_classifier_parameters
append_data_set_details(baobab_path, baobab_enabled, baobab_classifier_parameters, "Baobab", data_set_arr)

# German credit data set
german_credit_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/german_finance/german_dataset_numberised.csv")
german_credit_enabled = False
german_credit_parameters = default_classifier_parameters
append_data_set_details(german_credit_path, german_credit_enabled, german_credit_parameters, "German_credit", data_set_arr)

# Australian credit data set
australian_credit_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/australian_finance/australian.csv")
australian_credit_enabled = False
australian_credit_parameters = default_classifier_parameters
append_data_set_details(australian_credit_path, australian_credit_enabled, australian_credit_parameters, "Australian_credit", data_set_arr)
