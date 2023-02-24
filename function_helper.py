import pandas as pd
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


"""
Custom functions for analysis of Cafe Casa Martinez data.
"""

def correlations(DataFrame, drop_col='tot_prod_kg'):
    """
    Given a pandas dataframe of data return a dataframe with columns:
    pearsonr, pval, and whether or not the correlation is significant

    Parameters:
    -----------
    DataFrame : pandas dataframe
        dataframe containing all data
    drop_col : string or array of strings, default='tot_prod_kg'
        The auto-correlated column to be dropped. e.g. drop tot_prod_kg because
        correlations are calculated with respect to it and this row will always 
        be 1.

    Returns:
    --------
    stats : Pandas DataFrame
        dataframe with columns pearson_corr, pval, and is_significant, which is 1 
        if significant and 0 otherwise
    """
    
    corr = DataFrame.corr()
    pval = corr.corr(method=lambda x, y: pearsonr(x, y)[1])
    stats = pd.DataFrame(index=corr.tot_prod_kg.index,
                         data=np.array([corr.tot_prod_kg, pval.tot_prod_kg, pval.tot_prod_kg <= 0.05]).transpose(),
                         columns=['pearson_corr', 'pval', 'is_significant'])
    stats = stats.sort_values(by='pearson_corr', ascending=False)
    stats = stats.drop(drop_col)
    return stats


def thresholds(y_true, y_pred, pos_label):
    """
    Get true positive rate, false positive rate, and accompanying
    thresholds in a dataframe
    
    Parameters:
    -----------
    y_true : ndarray of shape associated with model
        X_test array or matrix the model was trained on
    y_pred : ndarray of shape associated with model
        y_test array or matrix the model was trained on
    pos_label : int or str
        The label of the positive class

    Returns:
    --------
    df : Pandas DataFrame
        pandas dataframe with columns, tpr, fpr, and thresh
    """
        
    fpr, tpr, thresh = roc_curve(y_true, y_pred, pos_label=pos_label)

    df = pd.DataFrame()
    df['tpr'] = tpr
    df['fpr'] = fpr
    df['thresh'] = thresh

    return df