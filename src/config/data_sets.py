"""Data sets to test against"""
import os
import sys

data_set_arr = []


def append_data_set_details(data_set_path, status, data_set_description, data_sets):
    """Adds data set path, data set status and data set description to data_set_arr as a dictionary"""
    data_sets.append({"data_set_path": data_set_path, "status": status, "data_set_description": data_set_description})

# Lima TB data set
lima_tb_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/lima_tb/Lima-TB-Treatment-base.csv")
lima_tb_enabled = True
append_data_set_details(lima_tb_path, lima_tb_enabled, "Lima_TB", data_set_arr)

# Baobab data set
baobab_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "<TODO>")
baobab_enabled = False
append_data_set_details(baobab_path, baobab_enabled, "Baobab", data_set_arr)

# German credit data set
german_credit_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/german_finance/german_dataset_numberised.csv")
german_credit_enabled = False
append_data_set_details(german_credit_path, german_credit_enabled, "German_credit", data_set_arr)

# Australian credit data set
australian_credit_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../data/australian_finance/australian.csv")
australian_credit_enabled = False
append_data_set_details(australian_credit_path, australian_credit_enabled, "Australian_credit", data_set_arr)


