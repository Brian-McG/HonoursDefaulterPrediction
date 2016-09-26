import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import data_sets
import visualisation as vis


def main():
    for data_set in data_sets.data_set_arr:
        if data_set["status"] and data_set["time_to_default"] is not None:
            # Load in data set
            input_defaulter_set = pd.DataFrame.from_csv(data_set["data_set_path"], index_col=None, encoding="UTF-8")

            defaulters = input_defaulter_set[~np.isnan(input_defaulter_set[data_set["time_to_default"]])]

            vis.plot_kaplan_meier_graph_of_time_to_default(defaulters[data_set["time_to_default"]], data_set_description=data_set["data_set_description"])


if __name__ == "__main__":
    main()
