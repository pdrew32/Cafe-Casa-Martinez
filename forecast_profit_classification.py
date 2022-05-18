import numpy as np
import pandas as pd
import pickle
import constants_helper as c

"""
Use previously defined classification routine to predict whether there will be a profit in 2020 and the probability of the profitability for both forecasted rainfall totals and true rainfall totals
"""

best_fit_coef_path = 'data/lr_coefs_best_fit.pkl'
forecast_path = 'data/forecast_rainfall.npy'

tot_plants = pd.read_csv('data/tot_plants.csv', index_col=0)
# define the profit threshold and make a new binary column where 1 is a year with profit and 0 is without
profit_threshold = c.c.profit_thresh_kg
tot_plants['profit'] = ((tot_plants['total_production'] > profit_threshold)*1).values

lr = pickle.load(open(best_fit_coef_path, 'rb'))
forecast = np.load(forecast_path)

profit_pred = lr.predict(np.array([[forecast[5-1], forecast[8-1]]]))
profit_pred_prob = lr.predict_proba(np.array([forecast[8-1], forecast[5-1]]).reshape(1,-1))
true_profit = tot_plants.profit[12]

print('predicted 2020 profit classification based on forecasted rainfall:', profit_pred)
print('probability of occurance:', profit_pred_prob[0][0], '\n')

profit_pred_truth = lr.predict(np.array([[tot_plants.loc[12, 'may_rain_cm'], tot_plants.loc[12, 'aug_rain_cm']]]))
profit_pred_prob_truth = lr.predict_proba(np.array([[tot_plants.loc[12, 'may_rain_cm'], tot_plants.loc[12, 'aug_rain_cm']]]))

print('predicted 2020 profit classification based on true rainfall:', profit_pred_truth)
print('probability of occurance:', profit_pred_prob_truth[0][0])
