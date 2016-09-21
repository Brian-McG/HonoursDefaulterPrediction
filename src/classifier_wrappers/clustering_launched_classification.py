import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime

import pandas as pd


class ClusteringLaunchedClassifier:
    def __init__(self, d=0.1):
        if sys.platform != 'win32':
            raise OSError("ClusteringLaunchedClassifier can only be run on Windows.")

        self.d = d
        self.clc_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/../../dependencies/CLC.exe')
        self.model = None
        self.current_time = None
        self.training_tmp_handle = None

    def fit(self, x_resampled, y_resampled):
        data = pd.DataFrame(data=x_resampled)
        data.insert(0, 'classification', y_resampled)

        self.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
        self.training_tmp_handle = tempfile.NamedTemporaryFile()
        training_data = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../dependencies/tmp/training_fold_{0}_{1}".format(os.path.basename(self.training_tmp_handle.name), self.current_time))
        self.model = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../dependencies/tmp/model_{0}_{0}".format(os.path.basename(self.training_tmp_handle.name), self.current_time))
        data.to_csv(path_or_buf=training_data, index=False, header=False, sep='\t')

        clc_arr = [self.clc_path, 'TRAIN', training_data, str(self.d), self.model]
        try:
            subprocess.check_output(clc_arr)
        except subprocess.CalledProcessError as e:
            os.remove(training_data)
            self.training_tmp_handle.close()
            raise e

        os.remove(training_data)

    def predict(self, x_testing):
        test_data = pd.DataFrame(data=x_testing)

        # The CLC tool expects an outcome value so it can calculate predictive accuracy. This however, is already calculated in our scripts so we just pass in a dummy value.
        y_arr = [-1000] * len(x_testing)
        test_data.insert(0, 'classification', y_arr)

        test_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../dependencies/tmp/testing_fold_{0}_{1}".format(os.path.basename(self.training_tmp_handle.name), self.current_time))
        test_data.to_csv(path_or_buf=test_path, index=False, header=False, sep='\t')

        prediction_output = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/../../dependencies/tmp/prediction_output_{0}_{1}".format(os.path.basename(self.training_tmp_handle.name), self.current_time))

        clc_arr = [self.clc_path, 'PREDICT', test_path, self.model, prediction_output]
        try:
            subprocess.check_output(clc_arr)
        except subprocess.CalledProcessError as e:
            os.remove(test_path)
            os.remove(self.model)
            self.training_tmp_handle.close()
            raise e

        predictions = [None] * len(x_testing)
        prediction = re.compile(r'\S*')
        index = 0
        for line in open(prediction_output, "r"):
            match = prediction.search(line)
            predictions[index] = int(match.group(0))
            index += 1

        os.remove(test_path)
        os.remove(self.model)
        os.remove(prediction_output)
        self.training_tmp_handle.close()

        return predictions
