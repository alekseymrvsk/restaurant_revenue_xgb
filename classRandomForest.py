import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib


def dataset_split_train(INPUT_PATH_TRAIN='input/train.csv'):
    data_train = pd.read_csv(INPUT_PATH_TRAIN)

    y_train = data_train.revenue
    y_train = y_train.astype(int)
    x_train = data_train.drop(columns=['revenue', 'Id', 'City'])

    x_train = x_train.replace({'City Group': {'Other': 0, 'Big Cities': 1}})
    x_train = x_train.replace({'Type': {'FC': 0, 'IL': 1, 'DT': 2, 'MB': 3}})
    x_train['Open Date'] = pd.to_datetime(x_train['Open Date'], format='%m/%d/%Y')
    x_train['Open Date'] = pd.to_datetime(x_train['Open Date']).astype(np.int64)

    return [x_train, y_train]


def dataset_split_test(INPUT_PATH_TEST='input/test.csv'):
    data_test = pd.read_csv(INPUT_PATH_TEST)

    x_test = data_test.drop(columns=['Id', 'City'])
    x_test = x_test.replace({'City Group': {'Other': 0, 'Big Cities': 1}})
    x_test = x_test.replace({'Type': {'FC': 0, 'IL': 1, 'DT': 2, 'MB': 3}})
    x_test['Open Date'] = pd.to_datetime(x_test['Open Date'], format='%m/%d/%Y')
    x_test['Open Date'] = pd.to_datetime(x_test['Open Date']).astype(np.int64)
    return x_test


class MyRandomForest:
    def __init__(self, RANDOM_STATE_MODEL=1, TREES_COUNT=50, N_SPLITS=10, N_REPEATS=3, RANDOM_STATE_METRIC=1):
        self.RANDOM_STATE_METRIC = RANDOM_STATE_METRIC
        self.N_REPEATS = N_REPEATS
        self.N_SPLITS = N_SPLITS
        self.RANDOM_STATE_MODEL = RANDOM_STATE_MODEL
        self.TREES_COUNT = TREES_COUNT

    def fit_model(self, input_path_train='input/train.csv'):
        dataset = dataset_split_train(input_path_train)
        x_train = dataset[0]
        y_train = dataset[1]
        model = RandomForestRegressor(n_estimators=self.TREES_COUNT, random_state=self.RANDOM_STATE_MODEL)
        model.fit(x_train, y_train)
        return model

    @staticmethod
    def predict_data(model, input_path_test='input/test.csv'):
        dataset = dataset_split_test(input_path_test)
        x_test = dataset

        prediction = model.predict(x_test)

        return prediction

    @staticmethod
    def save_model(model):
        joblib.dump(model, "./model_random_forest.joblib")


mrf = MyRandomForest()
my_model = mrf.fit_model('input/train.csv')
predict = mrf.predict_data(my_model, 'input/test.csv')

output_file = open('output_random_forest.csv', 'w')
i = 0

print('Id,Prediction', file=output_file)
for element in predict:
    print(i, ',', element, file=output_file)
    i = i + 1

output_file.close()
