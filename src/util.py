from config.constants import VERBOSE_MODE


def verbose_print(message=""):
    if VERBOSE_MODE is True:
        print(message)