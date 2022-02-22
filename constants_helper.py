import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler


tot_plants = pd.read_csv('data/tot_plants.csv', index_col=0)

scaler = RobustScaler()
data_scaled = scaler.fit_transform(tot_plants)
t = tot_plants.copy()
t.loc[:,:] = data_scaled

class c:
    """define class to hold constants we will repeatedly use in different scripts"""
    prod_curve_zoca = [0, 0.25, 0.8, 1.0, 0.7, 0.55]
    prod_curve_sow = [0, 0, 0.5, 0.7, 1.0, 0.7, 0.55]

    corr = t.corr()
    pval = corr.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(t.columns))

    profit_thresh_kg = 28*125
