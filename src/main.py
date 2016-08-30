"""Primary script used to execute the defaulter prediction"""
import pandas as pd

# User imports
from artificial_neural_network import ArtificialNeuralNetwork
import constants as const


def main():
    # Load in data set
    defaulter_set = pd.DataFrame.from_csv("../data/Lima-TB-Treatment-v2.csv", index_col=None, encoding="UTF-8")

    # Basic data set statistics
    print("== Basic Data Set Stats ==")
    print("Total number of decision features: ", len(const.LIMA_CLASSIFICATION_FEATURES))
    print("Total number of rows: ", defaulter_set.shape[0])
    print("Total number of defaulters: ", len(defaulter_set[defaulter_set[const.LIMA_TREATMENT_OUTCOME] == 1]))

    # Train and test ML techniques
    ann = ArtificialNeuralNetwork()
    ann.train_and_evaluate(defaulter_set)

if __name__ == "__main__":
    main()
