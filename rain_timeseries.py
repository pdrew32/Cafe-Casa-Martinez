import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose


def trend_strength(tsd):
    """
    Takes a seasonal decomposition and returns the strength of the trend.
    Based on from Wang, X., Smith, K. A., & Hyndman, R. J. (2006). Characteristic-based clustering for time series data. Data Mining and Knowledge Discovery
    doi s10618-005-0039-x
    """
    one_term = 1-np.nanvar(tsd.resid)/np.nanvar(tsd.trend + tsd.resid)
    return max([0, one_term])

def seasonal_strength(tsd):
    """
    Takes a seasonal decomposition and returns the strength of the seasonal component.
    Based on Wang, X., Smith, K. A., & Hyndman, R. J. (2006). Characteristic-based clustering for time series data. Data Mining and Knowledge Discovery
    doi s10618-005-0039-x
    """
    one_term = 1-np.nanvar(tsd.resid)/np.nanvar(tsd.seasonal + tsd.resid)
    return max([0, one_term])


save_figures = False
seasonal_decomp_plot_path = 'figures/seasonal_decomposition_plot.png'
seasonal_trend_plot_path = 'figures/seasonal_trend_plot.png'
seasonal_seasonal_plot_path = 'figures/seasonal_curve_plot.png'

rain = pd.read_csv('data/rain.csv', index_col=0)
time_series = np.sum(rain.loc[rain.year >= 2007, rain.columns[1:-2]], axis=1).values
month_n = np.arange(1, len(time_series)+1)

tsd = seasonal_decompose(time_series, period=12)

# Plot time series decomposition in 4 subplots, observed, 
# trend, seasonal, and residuals and save the figure
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
for ax in fig.get_axes():
    ax.set_xticks(np.arange(0, max(month_n)+1, 12))
    ax.set_xticklabels(np.arange(0, max(month_n)/12+1), fontsize=15, rotation=45)
    ax.label_outer()
    ax.set_xlim(0, max(month_n)+1)
    for i in range(1, int(max(month_n)/12)):
        ax.axvline(12*i, color='k')
ax1.plot(month_n, tsd.observed)
ax2.plot(month_n, tsd.trend)
ax3.plot(month_n, tsd.seasonal)
ax4.scatter(month_n, tsd.resid)
plt.xlabel('Year')
ax1.set_ylabel('Observed')
ax2.set_ylabel('Trend')
ax3.set_ylabel('Seasonal')
ax4.set_ylabel('Resid.')
if save_figures is True:
    plt.savefig(seasonal_decomp_plot_path)
plt.show()

# plot the underlying trend
plt.plot(month_n, tsd.trend, linewidth=2.0, color='k')
plt.xlim(0, max(month_n)+1)
plt.xticks(ticks=np.arange(0, max(month_n)+1, 12*2), labels=np.arange(2007, 2007+max(month_n)/12+1, 2).astype(int), fontsize=15, rotation=45)
plt.yticks(fontsize=15)
plt.ylabel('Trend (arbitrary units)')
plt.xlabel('Year')
if save_figures is True:
    plt.savefig(seasonal_trend_plot_path)
plt.show()

# plot seasonal trend
plt.plot(month_n[:12], tsd.seasonal[:12], linewidth=2.0, color='k')
plt.xlim(1, 12)
plt.ylabel('Seasonal Trend (arbitrary units)')
plt.xlabel('Month')
if save_figures is True:
    plt.savefig(seasonal_seasonal_plot_path)
plt.show()


print(f"Trend Strength: {np.round(trend_strength(tsd), 4)}")
print(f"Seasonal Strength: {np.round(seasonal_strength(tsd), 4)}")
