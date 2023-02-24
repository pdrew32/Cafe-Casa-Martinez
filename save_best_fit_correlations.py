import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import constants_helper as c
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib.ticker import MaxNLocator


"""
Fit and save best fit correlations and make some plots using them for the analysis
"""

plot_it = True
save_fits = False
save_figures = False

save_path_may_rain_prod = 'figures/may_rain_vs_production.png'
save_path_may_rain_prod_per_plant = 'figures/may_rain_vs_production_per_plant.png'
save_path_n_plants_tot_prod = 'figures/n_plants_total_production.png'
save_path_median_rain_vs_month = 'figures/median_rainfall_vs_month.png'
save_path_may_rain_prod_perfect_prediction_with_year = 'figures/may_rain_vs_production_perfect_info_with_year.png'
save_path_may_rain_prod_perfect_prediction = 'figures/may_rain_vs_production_perfect_info.png'
save_path_year_vs_prod = 'figures/year_vs_production.png'
save_path_year_vs_prod_per_plant = 'figures/year_vs_prod_per_plant.png'
save_path_year_vs_prod_projection = 'figures/year_vs_production_projection.png'
save_path_year_vs_prod_per_plant_projection = 'figures/year_vs_prod_per_plant_projection.png'
save_path_year_vs_nplant = 'figures/year_vs_nplants.png'
save_path_field_actions_year_hist = 'figures/field_actions_year_hist.png'
save_path_renew_year_hist = 'figures/renewal_year_hist.png'
save_path_sow_year_hist = 'figures/sow_year_hist.png'

tot_plants = pd.read_csv('data/tot_plants.csv', index_col=0)
lots = pd.read_csv('data/lots.csv', index_col=0)

def lin_reg_params(x, y):
    """
    return m, b, variance, mse, r2 score from linear regression
    """
    reg = LinearRegression().fit(x, y)
    yhat = reg.predict(x)
    mse = mean_squared_error(y, yhat)
    r2 = r2_score(y, yhat)
    return reg.coef_, reg.intercept_, mse, r2, reg

# fit a line to get started values for m and before fitting our main model
m_rain_prod, b_rain_prod, mse_rain_prod, r2_m_prod, reg_rain_prod = lin_reg_params(tot_plants.may_rain_cm.values.reshape(-1,1), tot_plants.total_production.values)

m_rain_prod_per_plant, b_rain_prod_per_plant, mse_rain_prod_per_plant, r2_rain_prod_per_plant, reg_rain_prod_per_plant = lin_reg_params(tot_plants.may_rain_cm.values.reshape(-1,1), tot_plants.prod_per_plant_kg.values)

m_nplants_prod, b_nplants_prod, mse_nplants_prod, r2_nplants_prod, reg_nplants_prod = lin_reg_params(tot_plants.tot_plants.values.reshape(-1,1), tot_plants.total_production.values)

reg_year_prod, b_year_prod, mse_year_prod, r2_year_prod, reg_year_prod = lin_reg_params(tot_plants.year.values.reshape(-1,1), tot_plants.total_production.values)

if save_fits is True:
    np.save('data/lin_reg_best_fit_may_rain_total_production.npy', [m_rain_prod, b_rain_prod])
    np.save('data/lin_reg_best_fit_may_rain_production_per_plant.npy', [m_rain_prod_per_plant, b_rain_prod_per_plant])
    np.save('data/lin_reg_best_fit_total_plants_production.npy', [m_nplants_prod, b_nplants_prod])


profit_threshold = c.c.profit_thresh_kg
x_ = tot_plants.may_rain_cm
y_ = m_rain_prod*x_ + b_rain_prod

field_actions = np.hstack([lots.cut_year, lots.sow_year])

if plot_it is True:
    ax = plt.figure().gca()
    sns.scatterplot(x=tot_plants.year.values, y=tot_plants.total_production.values, s=60)
    plt.xlabel('Year')
    plt.ylabel('Total Production (kg)')
    plt.xticks(rotation=45) # fontsize=12, 
    # plt.yticks(fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if save_figures is True:
        plt.savefig(save_path_year_vs_prod)
    plt.show()

    ax = plt.figure().gca()
    sns.scatterplot(x=tot_plants.year.values, y=tot_plants.total_production.values, s=60)
    plt.xlabel('Year')
    plt.ylabel('Total Production (kg)')
    plt.xticks(rotation=45) # fontsize=12, 
    plt.xlim(min(tot_plants.year)-2, 2035)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if save_figures is True:
        plt.savefig(save_path_year_vs_prod_projection)
    plt.show()

    ax = plt.figure().gca()
    plt.hist(field_actions, bins=15)
    plt.ylabel('Number of Fields Sown or Renewed')
    plt.xlabel('Year')
    plt.xlim(min(tot_plants.year), max(tot_plants.year)+1)
    plt.ylim(0, 5)
    plt.xticks(rotation=45)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if save_figures is True:
        plt.savefig(save_path_field_actions_year_hist)
    plt.show()

    ax = plt.figure().gca()
    plt.hist(lots.cut_year, color=sns.color_palette('colorblind')[0])
    plt.ylabel('Number of Fields Renewed')
    plt.xlabel('Year')
    plt.ylim(0, 5)
    plt.xlim(min(tot_plants.year), max(tot_plants.year)+1)
    plt.xticks(rotation=45)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if save_figures is True:
        plt.savefig(save_path_renew_year_hist)
    plt.show()

    ax = plt.figure().gca()
    plt.hist(lots.sow_year, color=sns.color_palette('colorblind')[2])
    plt.ylabel('Number of Fields Sown')
    plt.xlabel('Year')
    plt.ylim(0, 5)
    plt.xlim(min(tot_plants.year), max(tot_plants.year)+1)
    plt.xticks(rotation=45)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if save_figures is True:
        plt.savefig(save_path_sow_year_hist)
    plt.show()

    ax = plt.figure().gca()
    sns.scatterplot(x=tot_plants.year.values, y=tot_plants.prod_per_plant_kg.values, s=60)
    plt.xlabel('Year')
    plt.ylabel('Production Per Plant (kg)')
    plt.xticks(rotation=45) # fontsize=12, 
    # plt.yticks(fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if save_figures is True:
        plt.savefig(save_path_year_vs_prod_per_plant)
    plt.show()

    ax = plt.figure().gca()
    sns.scatterplot(x=tot_plants.year.values, y=tot_plants.prod_per_plant_kg.values, s=60)
    plt.xlabel('Year')
    plt.ylabel('Production Per Plant (kg)')
    plt.xticks(rotation=45) # fontsize=12, 
    plt.xlim(min(tot_plants.year)-2, 2035)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if save_figures is True:
        plt.savefig(save_path_year_vs_prod_per_plant_projection)
    plt.show()

    ax = plt.figure().gca()
    sns.scatterplot(x=tot_plants.year, y=tot_plants.tot_plants, s=60)
    plt.xlabel('Year')
    plt.ylabel('Total Number of Plants')
    plt.xticks(rotation=45) # fontsize=12, 
    # plt.yticks(fontsize=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if save_figures is True:
        plt.savefig(save_path_year_vs_nplant)
    plt.show()

    x0 = np.linspace(0, max(tot_plants.may_rain_cm)+0.1*max(tot_plants.may_rain_cm))
    y0 = m_rain_prod*x0 + b_rain_prod
    fig, ax = plt.subplots()
    scat = plt.scatter(tot_plants.may_rain_cm, tot_plants.total_production, s=50, c=tot_plants.year)
    plt.axis([plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], plt.gca().get_ylim()[0], plt.gca().get_ylim()[1]])
    # plt.axhline(profit_threshold, linewidth=2, color='red', zorder=-1)
    cb = fig.colorbar(scat, ax=ax)
    cb.set_label('year')
    plt.plot(x0, y0, linewidth=2.0, color='k', zorder=-1)
    plt.xlabel('Total Rain in May (mm)')
    plt.ylabel('Total Production (kg)')
    if save_figures is True:
        plt.savefig(save_path_may_rain_prod)
    plt.show()

    x_hat = tot_plants.may_rain_cm
    y_hat = m_rain_prod*x_hat + b_rain_prod
    plt.hist(y_hat-tot_plants.total_production, bins=5)
    plt.show()

    plt.scatter(x_hat, y_hat-tot_plants.total_production)
    plt.axhline(-np.std(y_hat-tot_plants.total_production))
    plt.axhline(np.std(y_hat-tot_plants.total_production))
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
    fig, ax = plt.subplots()
    scat = plt.scatter(tot_plants.may_rain_cm, tot_plants.prod_per_plant_kg, s=50, c=tot_plants.year)
    plt.axis([plt.gca().get_xlim()[0], plt.gca().get_xlim()[1], plt.gca().get_ylim()[0], plt.gca().get_ylim()[1]])
    cb = fig.colorbar(scat, ax=ax)
    cb.set_label('year')
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
