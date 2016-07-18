from sknn.platform import cpu64, threading
import pandas as pd
import matplotlib.pyplot as plt
from sknn.mlp import Classifier, Layer

# Constant variables
NUMBER_OF_FOLDS = 5
TRAINING_ERROR = "training error"
TREATMENT_OUTCOME = "Treatment Outcome"

# Global variables
errors = [{} for i in range(NUMBER_OF_FOLDS)]
current_i = None


def store_stats(avg_train_error, **_):
    if TRAINING_ERROR not in errors[current_i]:
        errors[current_i][TRAINING_ERROR] = []
        errors[current_i]["training_error_count"] = 1
    errors[current_i][TRAINING_ERROR].append(avg_train_error)
    errors[current_i]["training_error_count"] += 1


def main():
    # Load in data set
    defaulter_set = pd.DataFrame.from_csv("../data/Lima-TB-Treatment-v2.csv", index_col=None, encoding="UTF-8")
    defaulter_set_len = defaulter_set.shape[0]

    classification_features = ['Age', 'Sex', 'Marital Status', 'Poverty Level', 'Prison Hx',
                               'Completed Secondary Education', 'Hx of Tobacco Use',
                               'Alcohol Use at Least Once Per Week', 'History of Drug Use',
                               'Hx of Rehab', 'MDR-TB', 'Body Mass Index', 'Hx Chronic Disease', 'HIV Status',
                               'Hx Diabetes Melitus']

    # Prepare data set
    input_set = defaulter_set[classification_features]
    output_set = defaulter_set['Treatment Outcome']

    # Basic data set statistics
    print("== Basic Data Set Stats ==")
    print("Total number of decision features: ", input_set.shape[1])
    print("Total number of rows: ", defaulter_set_len)
    print("Total number of defaulters: ", len(defaulter_set[defaulter_set[TREATMENT_OUTCOME] == 1]))

    # Apply k-fold cross validation
    fold_len = defaulter_set_len / NUMBER_OF_FOLDS
    for i in range(NUMBER_OF_FOLDS):
        global current_i
        current_i = i
        min_range = int(fold_len * i)
        max_range = int(fold_len * (i + 1))

        # Training data
        x_train_dataframe = pd.concat([input_set.iloc[0:min_range], input_set.iloc[max_range:defaulter_set_len]])
        y_train_dataframe = pd.concat([output_set.iloc[0:min_range], output_set.iloc[max_range:defaulter_set_len]])

        # Testing data
        test_dataframe = defaulter_set.iloc[min_range:max_range]

        # Assert that data is as expected
        assert (x_train_dataframe.shape[0] == y_train_dataframe.shape[0])
        assert (test_dataframe.shape[0] == defaulter_set_len - x_train_dataframe.shape[0])

        # Training fold specific statistics
        print("\n== Training Stats Fold {0} ==".format(i + 1))
        print("Number of rows for training fold {0}: ".format(i + 1), x_train_dataframe.shape[0])
        print("Number of defaulters for training fold {0}: ".format(i + 1),
              y_train_dataframe[y_train_dataframe == 1].shape[0])

        # Train classifier
        nn = Classifier(layers=[Layer("Sigmoid", units=100), Layer("Softmax")], learning_rate=0.01, n_iter=10000, n_stable=100,
                        callback={'on_epoch_finish': store_stats})
        nn.fit(x_train_dataframe.as_matrix(), y_train_dataframe.as_matrix())

        # Testing fold specific statistics
        print("== Testing Stats Fold {0} ==".format(i + 1))
        print("Number of rows for training fold {0}: ".format(i + 1), test_dataframe.shape[0])
        print("Number of defaulters for training fold {0}: ".format(i + 1),
              test_dataframe[test_dataframe[TREATMENT_OUTCOME] == 1].shape[0])

        # Test accuracy
        test_classification = nn.predict(test_dataframe[classification_features].as_matrix())
        actual_outcome = test_dataframe[TREATMENT_OUTCOME].as_matrix()

        true_positive_count = 0
        true_negative_count = 0
        false_positive_count = 0
        false_negative_count = 0

        assert (len(test_classification) == len(actual_outcome))
        for z in range(len(test_classification)):
            # True positive
            if test_classification[z] == 1 and actual_outcome[z] == 1:
                true_positive_count += 1
            # True negative
            elif test_classification[z] != 1 and actual_outcome[z] != 1:
                true_negative_count += 1
            # False positive (Type I)
            elif test_classification[z] == 1 and actual_outcome[z] != 1:
                false_positive_count += 1
            # False negative (Type II)
            elif test_classification[z] != 1 and actual_outcome[z] == 1:
                false_negative_count += 1
            # Debug - error detection
            else:
                raise RuntimeError("Unexpected classification")

        errors[i]["true positive rate"] = true_positive_count / float(len(test_classification))
        errors[i]["true negative rate"] = true_negative_count / float(len(test_classification))
        errors[i]["false positive rate"] = false_positive_count / float(len(test_classification))
        errors[i]["false negative rate"] = false_negative_count / float(len(test_classification))

    # Error rates
    print(errors)
    avg_true_positive_rate = 0
    avg_true_negative_rate = 0
    avg_false_positive_rate = 0
    avg_false_negative_rate = 0
    for error_dict in errors:
        avg_true_positive_rate += error_dict["true positive rate"]
        avg_true_negative_rate += error_dict["true negative rate"]
        avg_false_positive_rate += error_dict["false positive rate"]
        avg_false_negative_rate += error_dict["false negative rate"]

    avg_true_positive_rate /= float(NUMBER_OF_FOLDS)
    avg_true_negative_rate /= float(NUMBER_OF_FOLDS)
    avg_false_positive_rate /= float(NUMBER_OF_FOLDS)
    avg_false_negative_rate /= float(NUMBER_OF_FOLDS)

    print("Average true positive rate:", avg_true_positive_rate)
    print("Average true negative rate:", avg_true_negative_rate)
    print("Average false positive rate:", avg_false_positive_rate)
    print("Average false negative rate:", avg_false_negative_rate)

    # Plot training error
    error_map = {}
    for i in range(NUMBER_OF_FOLDS):
        error_map[i + 1] = errors[i][TRAINING_ERROR]

    error_df = pd.DataFrame({k: pd.Series(v) for k, v in error_map.items()})
    error_df.plot()
    plt.show()


if __name__ == "__main__":
    main()
