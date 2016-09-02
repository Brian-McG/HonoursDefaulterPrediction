"""Constants used throughout the project"""
NUMBER_OF_FOLDS = 5
TRAINING_ERROR = "training error"
VERBOSE_MODE = True


def verbose_print(message=""):
    if VERBOSE_MODE:
        print(message)
