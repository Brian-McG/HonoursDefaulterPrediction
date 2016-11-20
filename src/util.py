"""Contains miscellaneous functionality used throughout the project"""
import psutil

from constants import VERBOSE_MODE


def verbose_print(message=""):
    if VERBOSE_MODE is True:
        print(message)


def get_number_of_processes_to_use():
    physical_cpu_count = psutil.cpu_count(logical=False)
    return physical_cpu_count * 2


def bcr_scorer(classifier, X, y):
    if len(X[0]) < 3:
        # Prevent using less than 3 features to stop issues with data balancing
        return -1
    y_pred = classifier.predict(X)
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(y_pred)):
        if y[i] == y_pred[i] and y[i] == 1:
            TP += 1
        elif y[i] == y_pred[i] and y[i] == 0:
            TN += 1
        elif y[i] != y_pred[i] and y[i] == 1:
            FN += 1
        elif y[i] != y_pred[i] and y[i] == 0:
            FP += 1

    TPR = TP / float(TP + FN)
    TNR = TN / float(TN + FP)

    BCR = (TPR + TNR) / 2.0
    return BCR
