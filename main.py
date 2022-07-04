import numpy as np
import os
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

plt.style.use("ggplot")

inputPathTrain = 'input/train.csv'
inputPathTest = 'input/train.csv'

data_train = pd.read_csv(inputPathTrain)
data_test = pd.read_csv(inputPathTest)

y_train = data_train.revenue
y_train = y_train.astype(int)
x_train = data_train.drop(columns=['revenue', 'Id', 'Open Date', 'City', 'City Group', 'Type'])

x_test = data_test.drop(columns=['revenue', 'Id', 'Open Date', 'City', 'City Group', 'Type'])
x_test = x_test.astype(int)

model = xgb.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
model.fit(x_train, y_train)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
scores = abs(scores)

print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

prediction = model.predict(x_test)

print('Prediction: ', prediction)
