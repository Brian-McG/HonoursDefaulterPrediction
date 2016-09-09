"""Constants used throughout the project"""
NUMBER_OF_FOLDS = 5
TRAINING_ERROR = "training error"
VERBOSE_MODE = True
RECORD_RESULTS = False
CUTOFF_RATE = 0.7
RETRY_COUNT = 3
TEST_REPEAT = 3


def verbose_print(message=""):
    if VERBOSE_MODE is True:
        print(message)
