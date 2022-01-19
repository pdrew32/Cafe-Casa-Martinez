import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


"""
Estimate the curve of fractional plant productivity as a function of year since the peak. 
This curve is known for the first three years and needs to be estimated from the data for later years.
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


# years, starting in 0
x = np.array([0.0, 1.0, 2.0])

# the observed fractional decay of production after the year of peak production
y = np.array([1, 0.7, 0.55])

