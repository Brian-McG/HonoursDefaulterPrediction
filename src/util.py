import psutil

from config.constants import VERBOSE_MODE


def verbose_print(message=""):
    if VERBOSE_MODE is True:
        print(message)


def get_number_of_processes_to_use():
    physical_cpu_count = psutil.cpu_count(logical=False)
    return physical_cpu_count * 2
