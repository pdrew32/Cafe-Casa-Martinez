from sklearn import linear_model, svm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import Generator, SFC64


rain = pd.read_csv('data/rain.csv', index_col=0)
prod = pd.read_csv('data/production.csv', index_col=0)
lots = pd.read_csv('data/lots.csv', index_col=0)
tot_plants = pd.read_csv('data/tot_plants.csv', index_col=0)

rg = Generator(SFC64())

april_rain = rain.loc[rain.month=='ABRIL']
april_rain['may_rain_total'] = rain.loc[rain.month=='MAYO', 'total'].values

april_rain.reset_index(inplace=True, drop=True)

april_rain_inds = april_rain.index.values
rg.shuffle(april_rain_inds)

april_rain = april_rain.loc[april_rain_inds]

april_rain_train = april_rain.iloc[:np.floor(len(april_rain)*0.7).astype(int)]
april_rain_test = april_rain.iloc[np.floor(len(april_rain)*0.7).astype(int):]

regr = linear_model.LinearRegression()

april_rain_train_x = april_rain_train.loc[:, april_rain_train.columns[1]:april_rain_train.columns[30]].values
april_rain_train_y = april_rain_train.loc[:, april_rain_train.columns[32]]
april_rain_test_x = april_rain_test.loc[:, april_rain_test.columns[1]: april_rain_train.columns[30]].values
april_rain_test_y = april_rain_test.loc[:, april_rain_train.columns[32]].values

regr.fit(april_rain_train_x, april_rain_train_y)

april_rain_y_pred = regr.predict(april_rain_test_x)

plt.scatter(april_rain_test.year, april_rain_test_y, label='test')
plt.scatter(april_rain_test.year, april_rain_y_pred, label='prediction')
plt.legend(fontsize='x-small')
plt.show()
