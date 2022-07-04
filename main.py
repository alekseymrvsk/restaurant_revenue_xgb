'''
TODO:
output to file
constants for params
'''

import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_val_score
import xgboost as xgb
import time

GRADIENT_BOOSTED_TREES_COUNT = 1000

inputPathTrain = 'input/train.csv'
inputPathTest = 'input/train.csv'
columns_for_hash = ['Open Date', 'City', 'City Group', 'Type']

data_train = pd.read_csv(inputPathTrain)
data_test = pd.read_csv(inputPathTest)

y_train = data_train.revenue
y_train = y_train.astype(int)
x_train = data_train.drop(columns=['revenue', 'Id'])

for column in columns_for_hash:
    x_train[column] = x_train[column].apply(hash)

x_test = data_test.drop(columns=['revenue', 'Id'])
for column in columns_for_hash:
    x_test[column] = x_test[column].apply(hash)

model = xgb.XGBRegressor(n_estimators=GRADIENT_BOOSTED_TREES_COUNT, max_depth=8, eta=0.01, subsample=0.7,
                         colsample_bytree=0.8)
model.fit(x_train, y_train)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = abs(scores)

print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

start_time = time.time()
prediction = model.predict(x_test)
end_time = time.time()

print("Time of prediction: ", end_time - start_time)

model.save_model("model_xgb.json")
