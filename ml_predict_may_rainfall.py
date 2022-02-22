from sklearn import linear_model, svm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.random import Generator, SFC64


rain = pd.read_csv('data/rain.csv', index_col=0)
prod = pd.read_csv('data/production.csv', index_col=0)
lots = pd.read_csv('data/lots.csv', index_col=0)
tot_plants = pd.read_csv('data/tot_plants.csv', index_col=0)

# convert production weight to loads (cargas)
prod['cargas'] = prod.weight_kg/125

rg = Generator(SFC64())

# make new dataframe with may rainfall totals and year
may_rain = rain.loc[rain.month=='MAYO', ['total', 'year']]
may_rain.index = may_rain.year
may_rain = may_rain.loc[np.isin(may_rain.year, prod.year.unique())]

# create dataframe containing rainfall totals from the beginning of each year through the end of april.
# will train/test on that.
dF = pd.DataFrame()
for i in range(len(prod.year.values)):
    a = rain.loc[rain.year == prod.year.values[i]]
    dF[str(prod.year.values[i])] = np.hstack(a.iloc[0:4, 1:-2].values)

# shuffle the years around before training/testing
year_cols = dF.columns.values
rg.shuffle(year_cols)
dF = dF.loc[:, year_cols]
dF = dF.fillna(0)

# split into testing and training sets
dF_train = dF.iloc[:, :np.floor(len(year_cols)*0.7).astype(int)]
dF_test = dF.iloc[:, np.floor(len(year_cols)*0.7).astype(int):]

train_x = dF_train.transpose().values
train_y = may_rain.iloc[:np.floor(len(year_cols)*0.7).astype(int)]['total'].values
test_x = dF_test.transpose().values
test_y_truth = may_rain.iloc[np.floor(len(year_cols)*0.7).astype(int):]['total'].values

# specify the model
regr = linear_model.LinearRegression()

regr.fit(train_x, train_y)

test_y = regr.predict(test_x)

plt.scatter(may_rain.iloc[np.floor(len(year_cols)*0.7).astype(int):]['year'].values, test_y, label='predicted')
plt.scatter(may_rain.iloc[np.floor(len(year_cols)*0.7).astype(int):]['year'].values, test_y_truth, label='truth')
plt.legend(fontsize='x-small')
plt.xlabel('year')
plt.ylabel('may rainfall (cm)')
plt.show()


all_preds = regr.predict(dF.transpose().values)

rain_to_prod_best_fit_params = np.load('data/lin_reg_best_fit_may_rain_total_production.npy')
pred_prod = rain_to_prod_best_fit_params[0] * all_preds + rain_to_prod_best_fit_params[1]

plt.scatter(dF.columns.astype(float), pred_prod/125.0, label='predicted')
plt.scatter(prod.year, prod.weight_kg/125.0, label='truth')
plt.axhline(28.5, color='k', linewidth=2.0, label='break even')
plt.xlabel('year')
plt.ylabel('yearly production (cargas)')
plt.legend(fontsize='x-small')
plt.show()
