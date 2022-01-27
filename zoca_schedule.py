import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from itertools import islice, cycle


"""
Determine best crop renewal schedule
"""


def quadratic(x, a, b, c):
    """
    quadratic function to estimate the fraction of production in a plant as a function of time since planting or renewing.
    
    Returns:
    --------
    y : array of floats
        the fraction of the total production in a plant
    """
    return a + b*x**c


rain = pd.read_csv('data/rain.csv', index_col=0)
prod = pd.read_csv('data/production.csv', index_col=0)
lots = pd.read_csv('data/lots.csv', index_col=0)
tot_plants = pd.read_csv('data/tot_plants.csv', index_col=0)

# estimate the fractional production of plants 
# years, starting in 0
x = np.array([0.0, 1.0, 2.0])
# the observed fractional decay of production after the year of peak production
y = np.array([1, 0.7, 0.55])

# fit the exponential function to estimate the fractional production for years beyond that which data constrains.
g = curve_fit(quadratic, x, y)

# production fraction vs time for 16 years after the peak. This will be used later to estimate the production for fields that have gone too long without being renewed.
production_vs_time = quadratic(np.linspace(0, 15, 16), g[0][0], g[0][1], g[0][2])

# initialize a dataframe with years between 2008 and 2042 for columns
future_prod = pd.DataFrame(0.0, index=list(lots.index), columns=np.arange(2008, 2042).astype(str))

# fill fields that were producing when the farm was purchased with the lowest production estimated from the data
future_prod.loc[np.isnan(lots.cut_year.astype(float))] = production_vs_time[4]

prod_curve_zoca = [0, 0.25, 0.8, 1.0, 0.7, 0.55]
prod_curve_sow = [0, 0, 0.5, 0.7, 1.0, 0.7, 0.55]
for i in list(lots.index):
    # if previously renewed, fill with zoca productivity curve for years in the future and make it cyclical
    if ~np.isnan(lots.loc[i, 'cut_year']):
        zoca_cols = np.isin(future_prod.columns.astype(int), np.arange(lots.loc[i, 'cut_year'], future_prod.columns.astype(int)[-1]+1))
        future_prod.loc[i, zoca_cols] = list(islice(cycle(prod_curve_zoca), sum(zoca_cols)))
    # if not yet renewed fill it with the sowing productivity curve, followed by zoca productivity curve that repeats
    if ~np.isnan(lots.loc[i, 'sow_year']):
        sow_cols = np.isin(future_prod.columns.astype(int), np.arange(lots.loc[i, 'sow_year'], lots.loc[i, 'sow_year']+7))
        future_prod.loc[i, sow_cols] = list(islice(cycle(prod_curve_sow), sum(sow_cols)))
        sow_cols = np.isin(future_prod.columns.astype(int), np.arange(lots.loc[i, 'sow_year']+7, future_prod.columns.astype(int)[-1]+1))
        future_prod.loc[i, sow_cols] = list(islice(cycle(prod_curve_zoca), sum(sow_cols)))
    future_prod.loc[i] = future_prod.loc[i] * lots.loc[i, 'n_plants']

# recommend a schedule for zoca for the 4 fields that are due or overdue for zoca.
# The field with the most plants should be renewed first in order to get that productivity on line as soon as possible. 
# The following year the second largest should be renewed for the same reason
# the 3rd year the smallest two should be renewed together to minimize the standard deviation of the difference between yearly productivity and mean productivity
# These three rules result in the following zoca schedule for the 4 fields that are due/overdue
# Note this is figured out simply and manually from looking at the number of plants in each field that's due/overdue for zoca. 
# We want the most even split of plants possible while getting the highest production fields producing maximally again asap.
recommended_zoca_year_list = [2024, 2024, 2023, 2022]

# copy the theoretical future productivity dataframe
recommended_prod = future_prod.copy()
# calculate the mean productivity after 2028, a regime where all field zoca schedules should be adhered to
mean_prod_after_2028 = np.mean(future_prod.loc[:, np.isin(future_prod.columns.astype(int), np.arange(2028, future_prod.columns.astype(int)[-1]+1))].sum())

# fill recommended production dataframe with production from recommended zoca schedule
for i in range(len(recommended_zoca_year_list)):
    zoca_cols = np.isin(recommended_prod.columns.astype(int), np.arange(recommended_zoca_year_list[i], future_prod.columns.astype(int)[-1]+1))
    zoca_inds = lots.index[(lots.cut_year+6 < 2023) | (lots.sow_year+7 < 2023)][i]

    recommended_prod.loc[zoca_inds, zoca_cols] = list(islice(cycle(prod_curve_zoca), sum(zoca_cols)))
    recommended_prod.loc[zoca_inds, zoca_cols] = recommended_prod.loc[zoca_inds, zoca_cols] * lots.loc[zoca_inds, 'n_plants']

std_prod_minus_avg = np.std(recommended_prod.loc[:, np.isin(future_prod.columns.astype(int), np.arange(2028, future_prod.columns.astype(int)[-1]+1))].sum() - mean_prod_after_2028)
print(std_prod_minus_avg)

fig, ax = plt.subplots()
ax.plot(np.arange(np.float(future_prod.columns[0]), np.float(future_prod.columns[-1])+1), future_prod.sum(), color='k', label='If Zoca Schedule was Kept')
ax.plot(np.arange(np.float(future_prod.columns[0]), np.float(future_prod.columns[-1])+1), recommended_prod.sum(), color='r', label='Optimal Zoca Schedule')
ax.axhline(mean_prod_after_2028, color='lightgrey', label='mean effective n_plants')
ax2 = ax.twinx()
ax2.plot(prod.year.values, prod.weight_kg, label='Actual Production (kg)')
ax.set_xlabel('year')
ax.set_ylabel('effective number of plants producing at maximum')
ax2.set_ylabel('Actual Production (kg)')
ax.legend(fontsize='x-small')
ax2.legend(fontsize='x-small')
plt.show()


# save a dataframe with the recommended lot zoca schedule each year
zoca_schedule = pd.DataFrame(index=np.arange(2022, 2051), columns=np.arange(0, 6))

d = {}
for i in lots.index:
    if ~np.isnan(lots.loc[i, 'cut_year']):
        d[str(lots.index[i])] = np.arange(lots.loc[i, 'cut_year'], 2051, 6)
    if ~np.isnan(lots.loc[i, 'sow_year']):
        d[str(lots.index[i])] = np.arange(lots.loc[i, 'sow_year']+7, 2051, 6)


dF = pd.DataFrame(index=lots.index, columns=np.arange(0, 10))
j=0
for i in lots.index:
    if np.isin(i, lots.index[(lots.cut_year+6 < 2023) | (lots.sow_year+7 < 2023)]):
        dF.loc[i] = np.arange(recommended_zoca_year_list[j], recommended_zoca_year_list[j]+60, 6)
        j += 1
    else:
        if ~np.isnan(lots.loc[i, 'cut_year']):
            dF.loc[i] = np.arange(lots.loc[i, 'cut_year'], lots.loc[i, 'cut_year']+60, 6)
        if ~np.isnan(lots.loc[i, 'sow_year']):
            dF.loc[i] = np.arange(lots.loc[i, 'sow_year']+7, lots.loc[i, 'sow_year']+67, 6)
dF['lot_name'] = lots.lot_name
dF['n_plants'] = lots.n_plants

# print this to check that the correct years were assigned for our custom 4
# dF.loc[lots.index[(lots.cut_year+6 < 2023) | (lots.sow_year+7 < 2023)]]

print(dF.loc[np.isin(dF, 2022), ['lot_name', 'n_plants']])
