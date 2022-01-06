import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

"""
clean data from excel files and save as csv
"""

rain = pd.read_excel('data/CONTROL_DE_LLUVIAS-1.xlsx', sheet_name=0)
