import numpy as np
import pandas as pd

"""
Estimate the increse in production over a period of 5 years considering the case of the overdue lots do not get renewed during that time.
"""

lots = pd.read_csv('data/lots.csv', index_col=0)

# info on overdue lots
lot_names = ['sancocho-2', 'sancocho-1', 'arrayanes', 'ceiba-3']
lot_number = ['11', '10', '9', '2']
n_plants = [5000, 1488, 906, 900]

lots['overdue'] = 0
lots.loc[np.isin(lots.lot_number, lot_number), 'overdue'] = 1

lots['not_overdue'] = 1
lots.loc[np.isin(lots.lot_number, lot_number), 'not_overdue'] = 0

# fraction of number of overdue plants to the total
n_old_to_total = sum(n_plants) / lots.n_plants.sum()

# let's assume a constant fractional production of 40% for old plants
frac_prod_old = 0.3
frac_prod_new = 0.60

frac_old = sum(lots.overdue * lots.n_plants * frac_prod_old)
frac_new = sum(lots.not_overdue * lots.n_plants * frac_prod_new)
frac_new_with_renewed = sum(lots.n_plants * frac_prod_new)

print(1 - frac_old / frac_new_with_renewed)
