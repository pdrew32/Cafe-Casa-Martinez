import pandas as pd
import numpy as np

"""
Determine when to zoca each lot
"""

save_frame = False
save_frame_path = 'data/zoca_year.csv'

lots = pd.read_csv('data/lots.csv', index_col=0)

# I manually chose these years to evenly spread the number of plants in lots that are currently overdue for zoca
recommended_zoca_year_list = [2024, 2024, 2023, 2022]

year_iterator = np.arange(2018, 2081).astype(int)

zoca_year = pd.DataFrame(data=0, index=year_iterator, columns=lots.lot_number.values)

j=0
for i in lots.index:
    # if the lot is overdue, assign it one of the manually recommended years to zoca
    if np.isin(i, lots.index[(lots.cut_year+6 < 2023) | (lots.sow_year+7 < 2023)]):
        zoca_year.loc[np.arange(recommended_zoca_year_list[j], recommended_zoca_year_list[j]+60, 6), lots.loc[i, 'lot_number']] = 1
        j += 1
    # if the lot is not overdue, fill 1 in for every 6 years after the recent zoca or if recently sown, zoca after 7 years and then every 6 after that
    else:
        if ~np.isnan(lots.loc[i, 'cut_year']):
            zoca_year.loc[np.arange(lots.loc[i, 'cut_year'], max(year_iterator)+1, 6), lots.loc[i, 'lot_number']] = 1
        elif ~np.isnan(lots.loc[i, 'sow_year']):
            zoca_year.loc[np.arange(lots.loc[i, 'sow_year']+7, max(year_iterator)+1, 6), lots.loc[i, 'lot_number']] = 1

zoca_year.drop(index=[2018, 2019, 2020, 2021], inplace=True)

if save_frame is True:
    zoca_year.to_csv(save_frame_path)