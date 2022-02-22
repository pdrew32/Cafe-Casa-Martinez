import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler
from scipy.stats import pearsonr
import seaborn as sns

def normalized_regression(x, y, order=1):
    """
    Normalize the data and perform a linear regression
    """
    normx = (x-np.median(x))/(np.percentile(x, 75) - np.percentile(x, 25))
    normy = (y-np.median(y))/(np.percentile(y, 75) - np.percentile(y, 25))
    m, b = np.polyfit(normx, normy, order)
    return m, b

plot_it = True
save_fits = True
save_figures = True

save_path_may_rain_prod = 'figures/may_rain_vs_production.png'
save_path_may_rain_prod_per_plant = 'figures/may_rain_vs_production_per_plant.png'
save_path_n_plants_tot_prod = 'figures/n_plants_total_production.png'
save_path_median_rain_vs_month = 'figures/median_rainfall_vs_month.png'
save_path_may_rain_prod_perfect_prediction_with_year = 'figures/may_rain_vs_production_perfect_info_with_year.png'
save_path_may_rain_prod_perfect_prediction = 'figures/may_rain_vs_production_perfect_info.png'

tot_plants = pd.read_csv('data/tot_plants.csv', index_col=0)

scaler = RobustScaler()
data_scaled = scaler.fit_transform(tot_plants)
t = tot_plants.copy()
t.loc[:,:] = data_scaled

c = t.corr()
p = c.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(len(t.columns))

# quickly fit a line to get started values for m and before fitting our main model
m_rain_prod, b_rain_prod = np.polyfit(tot_plants.may_rain_cm.values, tot_plants.total_production.values, 1)
m_rain_prod_per_plant, b_rain_prod_per_plant = np.polyfit(tot_plants.may_rain_cm.values, tot_plants.prod_per_plant_kg.values, 1)
m_nplants_prod, b_nplants_prod = np.polyfit(tot_plants.tot_plants.values, tot_plants.total_production.values, 1)
m_year_prod, b_year_prod = np.polyfit(tot_plants.year.values, tot_plants.total_production.values, 1)

# m_rain_prod, b_rain_prod = normalized_regression(tot_plants.may_rain_cm.values, tot_plants.total_production.values)


if save_fits is True:
    np.save('data/lin_reg_best_fit_may_rain_total_production.npy', [m_rain_prod, b_rain_prod])
    np.save('data/lin_reg_best_fit_may_rain_production_per_plant.npy', [m_rain_prod_per_plant, b_rain_prod_per_plant])
    np.save('data/lin_reg_best_fit_total_plants_production.npy', [m_nplants_prod, b_nplants_prod])

profit_threshold = 28*125
x_ = tot_plants.may_rain_cm
y_ = m_rain_prod*x_ + b_rain_prod

if plot_it is True:
    x0 = np.linspace(0, max(tot_plants.may_rain_cm)+0.1*max(tot_plants.may_rain_cm))
    y0 = m_rain_prod*x0 + b_rain_prod
    plt.scatter(tot_plants.may_rain_cm, tot_plants.total_production, s=50, color=sns.color_palette('colorblind')[0])
    plt.axis([plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], plt.gca().get_ylim()[0], plt.gca().get_ylim()[1]])
    plt.axhline(profit_threshold, linewidth=2, color='red', zorder=-1)
    plt.plot(x0, y0, linewidth=2.0, color='k', zorder=-1)
    plt.xlabel('Total Rain in May (mm)')
    plt.ylabel('Total Production (kg)')
    if save_figures is True:
        plt.savefig(save_path_may_rain_prod)
    plt.show()

    fig, ax = plt.subplots()
    scat = plt.scatter(tot_plants.may_rain_cm, tot_plants.total_production, s=60, c=tot_plants.year, zorder=99, marker='s')
    cb = fig.colorbar(scat, ax=ax)
    cb.set_label('year')
    plt.scatter(x_, y_, s=50, color=sns.color_palette('colorblind')[5], label='Best Fit Production')
    for i in range(len(tot_plants.total_production)):
        # plt.text(tot_plants.may_rain_cm[i]+5, tot_plants.total_production[i], str(tot_plants.year[i]), fontsize='x-small')
        if ((tot_plants.total_production[i] > profit_threshold) & (y_[i] < profit_threshold)) | ((tot_plants.total_production[i] < profit_threshold) & (y_[i] > profit_threshold)):
            plt.plot([x_[i], x_[i]], [tot_plants.total_production[i], y_[i]], color='red', zorder=-1, linewidth=2)
            plt.scatter(tot_plants.may_rain_cm[i], tot_plants.total_production[i], s=130, color='red', zorder=-1, marker='s')
            plt.scatter(x_[i], y_[i], s=130, color='red', zorder=-1)
        else:
            plt.plot([x_[i], x_[i]], [tot_plants.total_production[i], y_[i]], color='grey', zorder=-1)
    plt.axis([plt.gca().get_xlim()[0], plt.gca().get_xlim()[1]+30, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1]])
    plt.scatter(0, 0, marker='s', label='True Production', color=sns.color_palette('viridis')[0])
    plt.axhline(profit_threshold, linewidth=2, color='black', zorder=-1)
    plt.plot(x0, y0, linewidth=2.0, color='k', zorder=-1)
    plt.xlabel('Total Rain in May (mm)')
    plt.ylabel('Total Production (kg)')
    plt.legend(fontsize='x-small')
    if save_figures is True:
        plt.savefig(save_path_may_rain_prod_perfect_prediction_with_year)
    plt.show()

    fig, ax = plt.subplots()
    plt.scatter(tot_plants.may_rain_cm, tot_plants.total_production, s=60, color=sns.color_palette('colorblind')[0], zorder=99, marker='s')
    plt.scatter(x_, y_, s=50, color=sns.color_palette('colorblind')[5], label='Best Fit Production')
    for i in range(len(tot_plants.total_production)):
        # plt.text(tot_plants.may_rain_cm[i]+5, tot_plants.total_production[i], str(tot_plants.year[i]), fontsize='x-small')
        if ((tot_plants.total_production[i] > profit_threshold) & (y_[i] < profit_threshold)) | ((tot_plants.total_production[i] < profit_threshold) & (y_[i] > profit_threshold)):
            plt.plot([x_[i], x_[i]], [tot_plants.total_production[i], y_[i]], color='red', zorder=-1, linewidth=2)
            plt.scatter(tot_plants.may_rain_cm[i], tot_plants.total_production[i], s=130, color='red', zorder=-1, marker='s')
            plt.scatter(x_[i], y_[i], s=130, color='red', zorder=-1)
        else:
            plt.plot([x_[i], x_[i]], [tot_plants.total_production[i], y_[i]], color='grey', zorder=-1)
    plt.axis([plt.gca().get_xlim()[0], plt.gca().get_xlim()[1]+30, plt.gca().get_ylim()[0], plt.gca().get_ylim()[1]])
    plt.scatter(0, 0, marker='s', label='True Production', color=sns.color_palette('viridis')[0])
    plt.axhline(profit_threshold, linewidth=2, color='black', zorder=-1)
    plt.plot(x0, y0, linewidth=2.0, color='k', zorder=-1)
    plt.xlabel('Total Rain in May (mm)')
    plt.ylabel('Total Production (kg)')
    plt.legend(fontsize='x-small')
    if save_figures is True:
        plt.savefig(save_path_may_rain_prod_perfect_prediction)
    plt.show()

    x0 = np.linspace(0, max(tot_plants.may_rain_cm)+0.1*max(tot_plants.may_rain_cm))
    y0 = m_rain_prod_per_plant*x0 + b_rain_prod_per_plant
    plt.scatter(tot_plants.may_rain_cm, tot_plants.prod_per_plant_kg, s=50, color=sns.color_palette('colorblind')[0])
    plt.axis([plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], plt.gca().get_ylim()[0], plt.gca().get_ylim()[1]])
    plt.plot(x0, y0, linewidth=2.0, color='k', zorder=-1)
    plt.xlabel('Total Rain in May (mm)')
    plt.ylabel('Production Per Plant (kg)')
    if save_figures is True:
        plt.savefig(save_path_may_rain_prod_per_plant)
    plt.show()

    x0 = np.linspace(0, max(tot_plants.tot_plants)+0.1*max(tot_plants.tot_plants))
    y0 = m_nplants_prod*x0 + b_nplants_prod
    plt.scatter(tot_plants.tot_plants, tot_plants.total_production, s=50, color=sns.color_palette('colorblind')[0])
    plt.axis([plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], plt.gca().get_ylim()[0], plt.gca().get_ylim()[1]])
    plt.plot(x0, y0, linewidth=2.0, color='k', zorder=-1)
    plt.xlabel('Number of Plants')
    plt.ylabel('Total Production (kg)')
    if save_figures is True:
        plt.savefig(save_path_n_plants_tot_prod)
    plt.show()

    rain_months = np.array([np.median(tot_plants.jan_rain_cm), np.median(tot_plants.feb_rain_cm), np.median(tot_plants.mar_rain_cm), np.median(tot_plants.apr_rain_cm), np.median(tot_plants.may_rain_cm), np.median(tot_plants.jun_rain_cm), np.median(tot_plants.jul_rain_cm), np.median(tot_plants.aug_rain_cm), np.median(tot_plants.sep_rain_cm), np.median(tot_plants.oct_rain_cm), np.median(tot_plants.nov_rain_cm), np.median(tot_plants.dec_rain_cm)])
    rain_months_std = np.array([np.std(tot_plants.jan_rain_cm), np.std(tot_plants.feb_rain_cm), np.std(tot_plants.mar_rain_cm), np.std(tot_plants.apr_rain_cm), np.std(tot_plants.may_rain_cm), np.std(tot_plants.jun_rain_cm), np.std(tot_plants.jul_rain_cm), np.std(tot_plants.aug_rain_cm), np.std(tot_plants.sep_rain_cm), np.std(tot_plants.oct_rain_cm), np.std(tot_plants.nov_rain_cm), np.std(tot_plants.dec_rain_cm)])
    months = np.arange(1,13)
    plt.errorbar(months, rain_months, yerr=rain_months_std, fmt='o', color=sns.color_palette('colorblind')[0], capsize=4, elinewidth=2, markersize=9, capthick=2)
    plt.xticks(ticks=months, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
    plt.ylabel('Median Rainfall Totals (mm)')
    if save_figures is True:
            plt.savefig(save_path_median_rain_vs_month)
    plt.show()


# find fraction of incorrect predictions if we had perfect information from machine learning the May rainfall
print('fraction of years where perfect information would lead to an accurate prediction of whether a year will be profitable:')
print(sum((y_ > profit_threshold) & (tot_plants.total_production > profit_threshold)) / sum(tot_plants.total_production > profit_threshold))
