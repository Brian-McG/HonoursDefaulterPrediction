from sknn.platform import cpu64, threading
import pandas as pd
from sknn.mlp import Classifier, Layer

# Constant variables
NUMBER_OF_FOLDS = 5

# Load in data set
defaulter_set = pd.DataFrame.from_csv("../data/Lima-TB-Treatment-v2.csv", index_col=None, encoding="UTF-8")
defaulter_set_len = len(defaulter_set)
fold_len = defaulter_set_len / NUMBER_OF_FOLDS

# Apply k-fold cross validation
for i in range(1, NUMBER_OF_FOLDS + 1):
    min_range = int(fold_len * (i - 1))
    max_range = int(fold_len * i)

    input_set = defaulter_set[['Age', 'Sex', 'Marital Status', 'Poverty Level', 'Prison Hx',
                               'Completed Secondary Education', 'Hx of Tobacco Use',
                               'Alcohol Use at Least Once Per Week', 'History of Drug Use',
                               'Hx of Rehab', 'MDR-TB', 'Body Mass Index', 'Hx Chronic Disease', 'HIV Status',
                               'Hx Diabetes Melitus']]
    output_set = defaulter_set['Treatment Outcome']

    x_train = pd.concat([input_set.iloc[0:min_range], input_set.iloc[max_range:defaulter_set_len]]).as_matrix()
    y_train = pd.concat([output_set.iloc[0:min_range], output_set.iloc[max_range:defaulter_set_len]]).as_matrix()
    x_test = input_set.iloc[min_range:max_range].as_matrix()
    y_test = output_set.iloc[min_range:max_range].as_matrix()

    nn = Classifier(layers=[Layer("Rectifier", units=100), Layer("Softmax")], learning_rate=0.01, n_iter=2500)
    nn.fit(x_train, y_train)

    print(nn.score(x_test, y_test))
