import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

"""
clean data from excel files and save as csv
"""

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

lots.rename(columns={lots.columns[0]:'lot_number', lots.columns[1]:'lot_name', lots.columns[2]:'sow_month', lots.columns[3]:'sow_year', lots.columns[4]:'cut_month', lots.columns[5]:'cut_year',  lots.columns[6]:'variety',  lots.columns[7]:'n_plants', lots.columns[8]:'planting_distance_streets_meters', lots.columns[10]:'planting_distance_grooves_meters', lots.columns[11]:'area_in_sq_meters'}, inplace=True)

lots.drop(columns='Unnamed: 9', index=0, inplace=True)

lots.loc[~np.isnan(lots.cut_month.astype(float)), 'cut_year']
lots.loc[~np.isnan(lots.sow_year.astype(float)), 'sow_year']

# add columns for each year, fill with 1 if producing, 0 if not.
lots = pd.concat([pd.DataFrame(columns=np.arange(2008, 2021)), lots])

for i in range(len(np.arange(2008, 2021))):
    year = lots.columns[i]
    lots[year] = 0

    lots.loc[(~np.isnan((lots.cut_year).astype(float))) & ((year < lots.cut_year) | (year >= lots.cut_year+2)), year] = 1*lots.n_plants

    lots.loc[(~np.isnan((lots.sow_year).astype(float))) & (lots.sow_year+3 <= year), year] = 1*lots.n_plants
    
# create new dataframe with total number of producing plants per year
tot_plants = pd.DataFrame()
tot_plants['year'] = np.arange(2008, 2021)

# add total plants per year
tot_plants['tot_plants'] = lots[np.arange(2008, 2021)].sum().values

tot_plants['prod_per_plant_kg'] = production.weight_kg.values/tot_plants['tot_plants'].values

for i in range(len(np.arange(2008, 2021))):
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'total_rain_cm'] = rain.loc[rain.year == np.arange(2008, 2021)[i], 'total'].sum()
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'jan_rain_cm'] = rain.loc[(rain.year == np.arange(2008, 2021)[i]) & (rain.month == 'ENERO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'feb_rain_cm'] = rain.loc[(rain.year == np.arange(2008, 2021)[i]) & (rain.month == 'FEBRERO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'mar_rain_cm'] = rain.loc[(rain.year == np.arange(2008, 2021)[i]) & (rain.month == 'MARZO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'apr_rain_cm'] = rain.loc[(rain.year == np.arange(2008, 2021)[i]) & (rain.month == 'ABRIL'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'may_rain_cm'] = rain.loc[(rain.year == np.arange(2008, 2021)[i]) & (rain.month == 'MAYO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'jun_rain_cm'] = rain.loc[(rain.year == np.arange(2008, 2021)[i]) & (rain.month == 'JUNIO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'jul_rain_cm'] = rain.loc[(rain.year == np.arange(2008, 2021)[i]) & (rain.month == 'JULIO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'aug_rain_cm'] = rain.loc[(rain.year == np.arange(2008, 2021)[i]) & (rain.month == 'AGOSTO'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'sep_rain_cm'] = rain.loc[(rain.year == np.arange(2008, 2021)[i]) & (rain.month == 'SEPTIEMBRE'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'oct_rain_cm'] = rain.loc[(rain.year == np.arange(2008, 2021)[i]) & (rain.month == 'OCTUBRE'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'nov_rain_cm'] = rain.loc[(rain.year == np.arange(2008, 2021)[i]) & (rain.month == 'NOVIEMBRE'), 'total'].values.astype(float)
    tot_plants.loc[tot_plants.year == np.arange(2008, 2021)[i], 'dec_rain_cm'] = rain.loc[(rain.year == np.arange(2008, 2021)[i]) & (rain.month == 'DICIEMBRE'), 'total'].values.astype(float)


# plot # of fresh fields (zoca or plant) vs year. need to control for the fact that they're new
tot_plants['fresh_fields'] = 0
year_fresh = np.sort(np.append(lots.sow_year.values+3, lots.cut_year.values+2).astype(float))
year_fresh = year_fresh[~np.isnan(year_fresh)]

"""fresh_counter = 0
for i in range(len(tot_plants)):
    if tot_plants.loc[i, 'year'] == year_fresh[0]:
        tot_plants.loc[i, 'fresh_fields'] = sum(year_fresh == year_fresh[fresh_counter])
        fresh_counter += 1
    if tot_plants.loc[i, 'year'] > year_fresh[0]:
        tot_plants.loc[i, 'fresh_fields'] = fresh_counter + sum(year_fresh == year_fresh[fresh_counter]) + tot_plants.loc[i-1, 'fresh_fields']
        fresh_counter += 1"""

for i in range(len(year_fresh)):
    tot_plants.loc[tot_plants.year == year_fresh[i], 'fresh_fields'] = sum(year_fresh == year_fresh[i])
    
for i in range(1, len(tot_plants)):
    tot_plants.loc[i, 'fresh_fields'] = tot_plants.loc[i, 'fresh_fields'] + tot_plants.loc[i-1, 'fresh_fields']


corr = tot_plants.corr()

# print 15 most highly correlated parameters that are not auto-correlations
print(corr.unstack().sort_values()[corr.unstack().sort_values() < 1][-15:])

plt.scatter(tot_plants.total_rain_cm, tot_plants.prod_per_plant_kg)
plt.ylabel('production per plant (kg)')
plt.xlabel('total rain (cm)')
plt.show()

plt.scatter(tot_plants.may_rain_cm, tot_plants.prod_per_plant_kg, c=tot_plants.year)
plt.ylabel('production per plant (kg)')
plt.xlabel('total rain in May (cm)')
plt.show()

plt.scatter(tot_plants.year, tot_plants.prod_per_plant_kg)
plt.ylabel('production per plant (kg)')
plt.xlabel('year')
plt.show()
