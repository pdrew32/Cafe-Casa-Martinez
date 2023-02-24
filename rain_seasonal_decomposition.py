import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import hurst

"""
Perform a seasonal decomposition on monthly rainfall data and calculate strength of trend and seasonal components
"""

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
save_forecast = False
forecast_path = 'data/forecast_rainfall.npy'
seasonal_decomp_plot_path = 'figures/seasonal_decomposition_plot.png'
seasonal_trend_plot_path = 'figures/seasonal_trend_plot.png'
seasonal_seasonal_plot_path = 'figures/seasonal_curve_plot.png'
autocorrelation_plot_path = 'figures/autocorrelation.png'
patrial_autocorrelation_plot_path = 'figures/partial_autocorrelation.png'
forecast_vs_reality_path = 'figures/forecast_vs_reality.png'

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
plt.xlim(0.5, 12.5)
plt.ylabel('Seasonal Trend (arbitrary units)')
plt.xlabel('Month')
plt.xticks(np.arange(1, 13), ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], rotation=45, fontsize=14)
if save_figures is True:
    plt.savefig(seasonal_seasonal_plot_path)
plt.show()

years = np.arange(0, 16) * 12
fig, ax = plt.subplots()
ax.set_aspect(0.1)
plt.plot(tsd.observed)
for i in range(len(years)):
    plt.axvline(years[i], zorder=-1, color='k')
plt.xlabel('Month')
plt.ylabel('Rainfall Totals (mm)')
plt.show()

print(f"Trend Strength: {np.round(trend_strength(tsd), 4)}")
print(f"Seasonal Strength: {np.round(seasonal_strength(tsd), 4)}")

# split into testing and training at the year closest to 70% of the data
# split_at_70_percent = np.floor(len(time_series) * 0.7/12).astype(int) * 12
# train = time_series[:split_at_70_percent]
# test = time_series[split_at_70_percent:]
train = time_series[:-12]
test = time_series[-12:]

# check the stationarity of the training data using Kwiatkowski-Phillips-Schmidt-Shin Test
kpss_stat, p_value, lags, crit = kpss(train, nlags='auto')
print(f'KPSS statisitic: {np.round(kpss_stat, 4)}')
print(f'KPSS p value: {np.round(p_value, 4)}. If > 0.05 data consistent with stationary.')
if p_value > 0.05:
    print('KPSS says stationary')
else: 
    print('KPSS says non-stationary')

adf = adfuller(train, regression='c')
print(f'ADF statisitic: {np.round(adf[0], 4)}')
print(f'ADF p value: {np.round(adf[1], 4)}. If < 0.05 data consistent with stationary.')
if adf[1] < 0.05:
    print('ADF says stationary')
else: 
    print('ADF says non-stationary')

# calculate the hurst exponent, a measure of the memory of a time series
H, c, data = hurst.compute_Hc(train, simplified=True)
print(f"Hurst Exponent: {np.round(H, 4)}")
print("Values < 0.5 imply mean-reverting. Close to 0.5 imply random walk. > 0.5 imply trending.")


# auto correlation plot
plot_acf(train)
plt.xlabel('lag (Months)')
plt.ylabel('Autocorrelation')
if save_figures is True:
    plt.savefig(autocorrelation_plot_path)
plt.show()

plot_pacf(train)
plt.xlabel('lag (Months)')
plt.ylabel('Partial Autocorrelation')
if save_figures is True:
    plt.savefig(patrial_autocorrelation_plot_path)
plt.show()

# fit arima model
mod1 = ARIMA(train, order=(1, 0, 2), seasonal_order=(1, 0, 2, 12))
res1 = mod1.fit()
mod2 = ARIMA(train, order=(2, 0, 2), seasonal_order=(2, 0, 2, 12))
res2 = mod2.fit()
mod3 = ARIMA(train, order=(3, 0, 3), seasonal_order=(3, 0, 3, 12))
res3 = mod3.fit()
mod4 = ARIMA(train, order=(0, 0, 0), seasonal_order=(0, 0, 0, 12))
res4 = mod4.fit()
mod5 = ARIMA(train, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12))
res5 = mod5.fit()
mod6 = ARIMA(train, order=(1, 0, 3), seasonal_order=(1, 0, 3, 12))
res6 = mod6.fit()

# res1 has the lowest aicc, so check its residuals and the autocorrelation plot of the residuals to check if it's a good model
plt.plot(res1.resid)
plt.ylabel('Residuals')
plt.xlabel('Month')
plt.show()

plt.hist(res1.resid, bins=20)
plt.ylabel('Count')
plt.xlabel('Residuals')
plt.show()

plot_acf(res1.resid)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.show()

# since the residuals are uncorrelated with each other on the acf plot and there's no trend in the plot of the residuals we conclude this is a good forecasting model.
# res1 with order=(1, 0, 2), seasonal_order=(1, 0, 2, 12) is the best model we have with ARIMA

"""
# Now let's compare the results with a different model, ETS
# Actually, this didn't work. If I have time later maybe I could test other models but for now let's move on
ets_model = ETSModel(train, seasonal_periods=12)
ets_fit = ets_model.fit()

plt.plot(train)
plt.plot(ets_fit.fittedvalues)
plt.show()
"""

forecast_test = res1.forecast(steps=12)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
ax1.plot(forecast_test, label='forecast', linewidth=3)
ax1.plot(test, label='truth', linewidth=3)
ax2.scatter(np.arange(0,12), test-forecast_test, zorder=99, color='k')
ax2.axhline(0, color='k', linewidth=0.1)
ax1.set_ylabel('Rainfall Totals (mm)', fontsize=16)
ax2.set_ylabel('Residuals (mm)', fontsize=16)
plt.xlabel('Month')
ax1.legend(fontsize='x-small')
fig.subplots_adjust(wspace=0, hspace=0)
plt.xticks(np.arange(0, 12), ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], rotation=45, fontsize=14) # np.arange(1,13))
ax1.tick_params(axis='both', labelsize=14)
ax2.tick_params(axis='both', labelsize=14)
if save_figures is True:
    plt.savefig(forecast_vs_reality_path)
plt.show()

if save_forecast is True:
    np.save(forecast_path, forecast_test)
