import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score
import time
from sklearn.ensemble import RandomForestRegressor
import joblib

RANDOM_STATE_MODEL = 1
TREES_COUNT = 50

N_SPLITS = 10
N_REPEATS = 3
RANDOM_STATE_METRIC = 1

INPUT_PATH_TRAIN = 'input/train.csv'
INPUT_PATH_TEST = 'input/test.csv'

# columns_for_hash = ['Open Date', 'City', 'City Group', 'Type']

data_train = pd.read_csv(INPUT_PATH_TRAIN)
data_test = pd.read_csv(INPUT_PATH_TEST)

y_train = data_train.revenue
y_train = y_train.astype(int)
x_train = data_train.drop(columns=['revenue', 'Id', 'City'])

x_train = x_train.replace({'City Group': {'Other': 0, 'Big Cities': 1}})
x_train = x_train.replace({'Type': {'FC': 0, 'IL': 1, 'DT': 2, 'MB': 3}})
tmp = x_train['Open Date'].str.split('/')
x_train['Open Date'] = tmp.str[1].astype(int) + tmp.str[0].astype(int) * 30 + tmp.str[2].astype(int) * 365

x_test = data_test.drop(columns=['Id', 'City'])
x_test = x_test.replace({'City Group': {'Other': 0, 'Big Cities': 1}})
x_test = x_test.replace({'Type': {'FC': 0, 'IL': 1, 'DT': 2, 'MB': 3}})
tmp = x_test['Open Date'].str.split('/')
x_test['Open Date'] = tmp.str[1].astype(int) + tmp.str[0].astype(int) * 30 + tmp.str[2].astype(int) * 365

model = RandomForestRegressor(n_estimators=TREES_COUNT, random_state=RANDOM_STATE_MODEL)
model.fit(x_train, y_train)

cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE_METRIC)

scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = abs(scores)

print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

start_time_prediction = time.time()
prediction = model.predict(x_test)
all_time_prediction = time.time() - start_time_prediction

print("Time of prediction: ", all_time_prediction)

output_file = open('output_random_forest.csv', 'w')
i = 0

print('Id,Prediction', file=output_file)
for element in prediction:
    print(i, ',', element, file=output_file)
    i = i + 1

output_file.close()

joblib.dump(model, "./model_random_forest.joblib")
