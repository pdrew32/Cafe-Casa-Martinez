import numpy as np
import pandas as pd
from datetime import date


"""
clean data from excel files and save as csv
"""

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
rain.drop(index=rain.loc[(rain.month == rain.month.unique()[0]) | (rain.month == rain.month.unique()[1]) ].index, inplace=True)

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
lots.reset_index(inplace=True, drop=True)

# array of years for which we have production data
year_arr = np.arange(production.year.values[0], production.year.values[-1]+1)
# add columns for each year, fill with 1 if producing, 0 if not.
lots = pd.concat([pd.DataFrame(columns=year_arr.astype(str)), lots])
# add columns for each year to be filled with theoretical 100% production
lots = pd.concat([pd.DataFrame(columns=np.core.defchararray.add(year_arr.astype(str), '_frac_prod')), lots])

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
    lots.iloc[i, np.where(lots.columns == str(year_arr[0]))[0][0] : np.where(lots.columns == str(year_arr[-1]))[0][0]+1] = prod_arr

# production per plant each year in the time before any fresh plantings or cuttings
# This will serve as our estimate of the long term low production after many years without cutting
median_old_plant_prod_per_plant = np.median(production.loc[production.year < 2012, 'weight_kg'] / lots.iloc[:, np.where(lots.columns == str(year_arr[0]))[0][0] : np.where(lots.columns == str(year_arr[-1]))[0][0]+1].sum()[0])

production['prod_per_plant'] = production.weight_kg.values/lots.iloc[:, np.where(lots.columns == str(year_arr[0]))[0][0] : np.where(lots.columns == str(year_arr[-1]))[0][0]+1].sum().values


# create new dataframe with total number of producing plants per year
tot_plants = pd.DataFrame()
tot_plants['year'] = year_arr

# add total plants per year
tot_plants['tot_plants'] = lots.iloc[:, np.where(lots.columns == str(year_arr[0]))[0][0] : np.where(lots.columns == str(year_arr[-1]))[0][0]+1].sum().values # lots[year_arr].sum().values

tot_plants['total_production'] = production.weight_kg.values
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

# print 15 most highly correlated parameters that are not auto-correlations
corr = tot_plants.corr()
print(corr.unstack().sort_values()[corr.unstack().sort_values() < 1][-20:])

if write_files is True:
    rain.to_csv('data/rain.csv')
    production.to_csv('data/production.csv')
    lots.to_csv('data/lots.csv')
    tot_plants.to_csv('data/tot_plants.csv')
