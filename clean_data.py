import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from datetime import date

"""
clean data from excel files and save as csv
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


write_files = True

save_path_rain = 'data/rain.csv'
save_path_production = 'data/production.csv'
save_path_lots = 'data/lots.csv'
save_path_tot_plants = 'data/tot_plants.csv'

# load rain data
rain = pd.read_excel('data/CONTROL_DE_LLUVIAS-1.xlsx', sheet_name=0)

# rename columns
rain.columns = rain.columns.str.replace('Unnamed: ', 'day_')
rain.rename(columns={rain.columns[0]:'month', rain.columns[-1]:'total'}, inplace=True)

# drop rows that only contain titles
rain.drop(index=rain.loc[(rain.month == rain.month.unique()[0]) | (rain.month == rain.month.unique()[1]) ].index, inplace=True) # | (rain.month == rain.month.unique()[2])

# get rows that correspond to years
row_is_a_year_inds = rain.loc[np.isnan(rain.month.str.isnumeric().astype(float))].index

# fill a new year column with years grabbed from rows
for i in range(len(row_is_a_year_inds)-1):
    rain.loc[row_is_a_year_inds[i]:row_is_a_year_inds[i+1]-1, 'year'] = rain.loc[row_is_a_year_inds, 'month'].values[i]
rain.loc[row_is_a_year_inds[i+1]:, 'year'] = rain.loc[row_is_a_year_inds, 'month'].values[i+1]

# drop rows that contained years only
rain.drop(index=row_is_a_year_inds, inplace=True)



# load production weight and year
production = pd.read_excel('data/PRODUCCION_CAFE.xlsx')
production = production[[production.columns[0], production.columns[5]]]

production.drop(index=0, inplace=True)
production.rename(columns={production.columns[0]:'year', production.columns[1]:'weight_kg'}, inplace=True)


# load lot data
lots = pd.read_excel('data/INVENTARIO_DE_CAFETALES.xlsx')

# rename columns
lots.rename(columns={lots.columns[0]:'lot_number', lots.columns[1]:'lot_name', lots.columns[2]:'sow_month', lots.columns[3]:'sow_year', lots.columns[4]:'cut_month', lots.columns[5]:'cut_year',  lots.columns[6]:'variety',  lots.columns[7]:'n_plants', lots.columns[8]:'planting_distance_streets_meters', lots.columns[10]:'planting_distance_grooves_meters', lots.columns[11]:'area_in_sq_meters'}, inplace=True)
lots.drop(columns='Unnamed: 9', index=0, inplace=True)

# array of years for which we have production data
year_arr = np.arange(production.year.values[0], production.year.values[-1]+1)
# add columns for each year, fill with 1 if producing, 0 if not.
lots = pd.concat([pd.DataFrame(columns=year_arr.astype(str)), lots])
# add columns for each year to be filled with theoretical 100% production
lots = pd.concat([pd.DataFrame(columns=np.core.defchararray.add(year_arr.astype(str), '_100_prod')), lots])

# for all lots in dataframe
for i in lots.index:
    prod_arr = np.zeros(len(year_arr))

    # if there was a zoca, year of cut has 0 production, all other years produce
    if ~np.isnan(lots.loc[i, 'cut_year']):
        productive_years_post_zoca = np.arange(lots.loc[i, 'cut_year']+1, date.today().year)
        productive_years_pre_zoca = np.arange(year_arr[0], lots.loc[i, 'cut_year'])

        prod_arr[np.isin(year_arr, productive_years_post_zoca)] = 1
        prod_arr[np.isin(year_arr, productive_years_pre_zoca)] = 1

    # if there was a sowing, years 0 and 1 have 0 production, all years after produce
    if ~np.isnan(lots.loc[i, 'sow_year']):
        productive_years_post_sow = np.arange(lots.loc[i, 'sow_year']+2, date.today().year)

        prod_arr[np.isin(year_arr, productive_years_post_sow)] = 1
    
    # multiply by number of plants
    prod_arr = prod_arr * lots.loc[i, 'n_plants']

    # fill each year in dataframe with the number of plants producing in each lot
    lots.iloc[i, np.where(lots.columns == str(year_arr[0]))[0][0]:np.where(lots.columns == str(year_arr[-1]))[0][0]+1] = prod_arr

# years, starting in 0
x = np.array([0.0, 1.0, 2.0, 10])
# the observed fractional decay of production after the year of peak production
y = np.array([1, 0.7, 0.55, 0.3240145672664953])

# fit the exponential function to estimate the fractional production for years beyond that which data constrains.
f = curve_fit(expon, x, y)
g = curve_fit(quadratic, x, y)

# production fraction vs time for 16 years after the peak. This will be used later to estimate the production for fields that have gone too long without being renewed.
# production_vs_time = expon(np.linspace(0, 15, 16), f[0][0], f[0][1])
production_vs_time = quadratic(np.linspace(0, 15, 16), g[0][0], g[0][1], g[0][2])

# for all lots in dataframe
for i in lots.index:
    prod_arr = np.zeros(len(year_arr))

    # year of cut, 0% production
    # in 1st year after cut, 25% production
    prod_arr[year_arr == lots.loc[i, 'cut_year']+1] = 0.25
    # in 2nd year after cut, 80% production
    prod_arr[year_arr == lots.loc[i, 'cut_year']+2] = 0.8
    # in third year after cut, returned to 100% production
    prod_arr[year_arr == lots.loc[i, 'cut_year']+3] = 1
    # 4th year after cut, down to 70%
    prod_arr[year_arr == lots.loc[i, 'cut_year']+4] = production_vs_time[1]
    # 5th year after cut, down to 50-60%
    prod_arr[year_arr == lots.loc[i, 'cut_year']+5] = production_vs_time[2]
    # continue to assume an exponential loss each year, as calculated based on three points (, 1, 0.7, 0.55)
    prod_arr[year_arr == lots.loc[i, 'cut_year']+6] = production_vs_time[3]
    prod_arr[year_arr == lots.loc[i, 'cut_year']+7] = production_vs_time[4]
    prod_arr[year_arr == lots.loc[i, 'cut_year']+8] = production_vs_time[5]

    # in the years before zoca it is unknown how old the plants are, other than that they are too old. 
    # Filling with the average of all plants in all fields before any cut/zoca or new field planting.
    # Percentage is a fraction of the expected 500g dried coffee per plant.
    prod_arr[year_arr == lots.loc[i, 'cut_year']-1] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-2] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-3] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-4] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-5] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-6] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-7] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-8] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-9] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-10] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-11] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-12] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-13] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-14] = production_vs_time[-1]
    prod_arr[year_arr == lots.loc[i, 'cut_year']-15] = production_vs_time[-1]

    # year of sowing, 0% production
    # in 1st year after sowing, insignificant production (calling it 0%)
    # in 2nd year after sowing, 50% production
    prod_arr[year_arr == lots.loc[i, 'sow_year']+2] = 0.5
    # in 3rd year after sowing, at 100% production
    prod_arr[year_arr == lots.loc[i, 'sow_year']+3] = 1
    # in 4th year after sowing, at 70% production
    prod_arr[year_arr == lots.loc[i, 'sow_year']+4] = 0.7
    # in 5th year after sowing, at 70% production
    prod_arr[year_arr == lots.loc[i, 'sow_year']+5] = 0.55
    prod_arr[year_arr == lots.loc[i, 'sow_year']+6] = production_vs_time[3]
    prod_arr[year_arr == lots.loc[i, 'sow_year']+7] = production_vs_time[4]
    prod_arr[year_arr == lots.loc[i, 'sow_year']+8] = production_vs_time[5]
    prod_arr[year_arr == lots.loc[i, 'sow_year']+9] = production_vs_time[6]
    prod_arr[year_arr == lots.loc[i, 'sow_year']+10] = production_vs_time[7]
    prod_arr[year_arr == lots.loc[i, 'sow_year']+11] = production_vs_time[8]
    
    # multiply fraction of production by number of plants
    prod_arr = prod_arr * lots.loc[i, 'n_plants']

    # fill each year in dataframe with the number of plants producing in each lot
    lots.loc[i, [lots.columns[0], lots.columns[1], lots.columns[2], lots.columns[3], lots.columns[4], lots.columns[5], lots.columns[6], lots.columns[7], lots.columns[8], lots.columns[9], lots.columns[10], lots.columns[11], lots.columns[12]]] = prod_arr

# create new dataframe with total number of producing plants per year
tot_plants = pd.DataFrame()
tot_plants['year'] = year_arr

# add total plants per year
tot_plants['tot_plants'] = lots[year_arr].sum().values

tot_plants['prod_per_plant_kg'] = production.weight_kg.values/tot_plants['tot_plants'].values

for i in range(len(year_arr)):
    tot_plants.loc[tot_plants.year == year_arr[i], 'total_rain_cm'] = rain.loc[rain.year == year_arr[i], 'total'].sum()
    tot_plants.loc[tot_plants.year == year_arr[i], 'jan_rain_cm'] = rain.loc[(rain.year == year_arr[i]) & (rain.month == 'ENERO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == year_arr[i], 'feb_rain_cm'] = rain.loc[(rain.year == year_arr[i]) & (rain.month == 'FEBRERO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == year_arr[i], 'mar_rain_cm'] = rain.loc[(rain.year == year_arr[i]) & (rain.month == 'MARZO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == year_arr[i], 'apr_rain_cm'] = rain.loc[(rain.year == year_arr[i]) & (rain.month == 'ABRIL'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == year_arr[i], 'may_rain_cm'] = rain.loc[(rain.year == year_arr[i]) & (rain.month == 'MAYO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == year_arr[i], 'jun_rain_cm'] = rain.loc[(rain.year == year_arr[i]) & (rain.month == 'JUNIO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == year_arr[i], 'jul_rain_cm'] = rain.loc[(rain.year == year_arr[i]) & (rain.month == 'JULIO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == year_arr[i], 'aug_rain_cm'] = rain.loc[(rain.year == year_arr[i]) & (rain.month == 'AGOSTO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == year_arr[i], 'sep_rain_cm'] = rain.loc[(rain.year == year_arr[i]) & (rain.month == 'SEPTIEMBRE'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == year_arr[i], 'oct_rain_cm'] = rain.loc[(rain.year == year_arr[i]) & (rain.month == 'OCTUBRE'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == year_arr[i], 'nov_rain_cm'] = rain.loc[(rain.year == year_arr[i]) & (rain.month == 'NOVIEMBRE'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == year_arr[i], 'dec_rain_cm'] = rain.loc[(rain.year == year_arr[i]) & (rain.month == 'DICIEMBRE'), 'total'].values.astype(float)


# plot # of fresh fields (zoca or plant) vs year. need to control for the fact that they're new
tot_plants['fresh_fields'] = 0
year_fresh = np.sort(np.append(lots.sow_year.values+3, lots.cut_year.values+2).astype(float))
year_fresh = year_fresh[~np.isnan(year_fresh)]

for i in range(len(year_fresh)):
    tot_plants.loc[tot_plants.year == year_fresh[i], 'fresh_fields'] = sum(year_fresh == year_fresh[i])
    
for i in range(1, len(tot_plants)):
    tot_plants.loc[i, 'fresh_fields'] = tot_plants.loc[i, 'fresh_fields'] + tot_plants.loc[i-1, 'fresh_fields']

# n plants before any plants are renewed or newly planted.
avg_starting_plants = lots.loc[lots[lots.columns[0]] > 0, 'n_plants'].sum()
average_starting_fraction = np.median(production.loc[1:4, 'weight_kg'] / avg_starting_plants) / 0.5

if write_files is True:
    rain.to_csv('data/rain.csv')
    production.to_csv('data/production.csv')
    lots.to_csv('data/lots.csv')
    tot_plants.to_csv('data/tot_plants.csv')
