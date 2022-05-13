import pandas as pd
import numpy as np

"""
Determine when to zoca each lot
"""

rain = pd.read_csv('data/rain.csv', index_col=0)
prod = pd.read_csv('data/production.csv', index_col=0)
lots = pd.read_csv('data/lots.csv', index_col=0)
tot_plants = pd.read_csv('data/tot_plants.csv', index_col=0)

# these years are chosen to evenly spread the number of plants in lots that are currently overdue for zoca
recommended_zoca_year_list = [2024, 2024, 2023, 2022]

# save a dataframe with years of when to sow
dF = pd.DataFrame(index=lots.index, columns=np.arange(0, 10))
j=0
for i in lots.index:
    # for every lot, if the overdue for zoca, fill with recommended_zoca_year_list going forward.
    # Else, continue with the previous zoca schedule
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
