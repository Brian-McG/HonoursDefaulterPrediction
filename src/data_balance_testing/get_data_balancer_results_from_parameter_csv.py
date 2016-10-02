import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import classifier_parameter_testing.classifier_parameter_tester as classifier_parameter_tester
from config.constants import DATA_BALANCER_STR
from visualisation import plot_balancer_results_per_classifier


def main(parameter_results):
    # Load in data set
    classifiers = []
    for (classifier_name, classifier_path) in parameter_results:
        parameter_results = pd.DataFrame.from_csv(classifier_path, index_col=None, encoding="UTF-8")
        data_balancers = classifier_parameter_tester.data_balancers
        data_balance_results = []
        for data_balancer in data_balancers:
            data_balance_df = parameter_results.loc[parameter_results[DATA_BALANCER_STR] == (data_balancer.__name__ if data_balancer is not None else "None")]
            data_balance_results.append((data_balancer.__name__ if data_balancer is not None else "None", data_balance_df.loc[data_balance_df["Average true rate"].argmax()]))
        classifiers.append((classifier_name, data_balance_results))
    plot_balancer_results_per_classifier(classifiers, "Average true rate")
    plot_balancer_results_per_classifier(classifiers, "Average true positive rate")
    plot_balancer_results_per_classifier(classifiers, "Average true negative rate")


if __name__ == "__main__":
    if len(sys.argv) < 3 and len(sys.argv) % 2 == 0:
        print('Expected "cdn_perf.py <parameter_result_label> <parameter_results_path>"')
    else:
        classifier_arr = []
        for i in range(1, len(sys.argv), 2):
            classifier_arr.append((sys.argv[i], sys.argv[i + 1]))
        main(classifier_arr)
