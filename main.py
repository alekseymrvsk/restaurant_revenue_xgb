import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score
import xgboost as xgb
import time

INPUT_PATH_TRAIN = 'input/train.csv'
INPUT_PATH_TEST = 'input/train.csv'

SUBSAMPLE_RATIO_COLUMNS = 0.8
SUBSAMPLE = 0.7
GRADIENT_BOOSTED_TREES_COUNT = 1000
MAX_DEPTH_TREE = 8
STEP_SIZE_DECREASE = 0.01

N_SPLITS = 10
N_REPEATS = 3
RANDOM_STATE = 1

columns_for_hash = ['Open Date', 'City', 'City Group', 'Type']

data_train = pd.read_csv(INPUT_PATH_TRAIN)
data_test = pd.read_csv(INPUT_PATH_TEST)

y_train = data_train.revenue
y_train = y_train.astype(int)
x_train = data_train.drop(columns=['revenue', 'Id'])

for column in columns_for_hash:
    x_train[column] = x_train[column].apply(hash)

x_test = data_test.drop(columns=['revenue', 'Id'])
for column in columns_for_hash:
    x_test[column] = x_test[column].apply(hash)

model = xgb.XGBRegressor(n_estimators=GRADIENT_BOOSTED_TREES_COUNT, max_depth=MAX_DEPTH_TREE, eta=STEP_SIZE_DECREASE,
                         subsample=SUBSAMPLE, colsample_bytree=SUBSAMPLE_RATIO_COLUMNS)
model.fit(x_train, y_train)

cv = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = abs(scores)

print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

start_time_prediction = time.time()
prediction = model.predict(x_test)
all_time_prediction = time.time() - start_time_prediction

print("Time of prediction: ", all_time_prediction)

output_file = open('output.csv', 'w')
i = 0

print('Id,Prediction', file=output_file)
for element in prediction:
    print(i, ',', element, file=output_file)
    i = i + 1

output_file.close()

model.save_model("model_xgb.json")
