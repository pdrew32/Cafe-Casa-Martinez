# Café Casa Martinez Data Science Consulting Project
## Table of Contents
* [Goals](https://github.com/pdrew32/Cafe-Casa-Martinez#goals)
* [Summary of Results](https://github.com/pdrew32/Cafe-Casa-Martinez#summary-of-results)
* [Methods](https://github.com/pdrew32/Cafe-Casa-Martinez#methods)
* [Products](https://github.com/pdrew32/Cafe-Casa-Martinez#products)
* [Background](https://github.com/pdrew32/Cafe-Casa-Martinez#coffee-background)
* [Exploratory Data Analysis and Machine Learning](https://github.com/pdrew32/Cafe-Casa-Martinez#exploratory-data-analysis-and-machine-learning)
* [Predicting Production](https://github.com/pdrew32/Cafe-Casa-Martinez#predicting-production)
* [Classification as Profitable vs Unprofitable](https://github.com/pdrew32/Cafe-Casa-Martinez#classification-as-profitable-vs-unprofitable)
* [Weather Forecasting](https://github.com/pdrew32/Cafe-Casa-Martinez#weather-forecasting)
* [Zoca Schedule](https://github.com/pdrew32/Cafe-Casa-Martinez#zoca-schedule)
* [Final Product – Timeline](https://github.com/pdrew32/Cafe-Casa-Martinez#final-product---timeline)

## Goals
[Café Casa Martinez](https://www.cafecasamartinez.com/) is a small, organic coffee company with a farm called Finca Tulipan in the Santander department of Colombia. The goals of this consulting project are to:

1. Find ways to increase production based on available data.
2. Predict production, profitability, and rainfall totals on the farm.
3. Look for evidence of the effect of global warming on rainfall totals over the last 15 years. 
    1. Some coffee farms in Colombia are closing because global warming has decreased rainfall making them unprofitable. If there is evidence of decreasing rainfall, determine which year the farm is projected to become unprofitable.
    2. Investigate the possibility to fully irrigate the farm by that date. 
    3. Investigate during which months rainfall is critical.

## Summary of Results
1. My statistical analysis of the correlations between production and all other data demonstrates that the most important thing Café Casa Martinez can do to increase production is by performing regular upkeep of their fields. This practice, known as renewal or zoca, has a stronger effect on total yearly production than rainfall. To optimize profits while minimizing short-term disruptions in production, I propose a renewal schedule for the farm. Farmers can access the recommended fields for renewal each year through the [dashboard](https://cafe-casa-martinez-app.herokuapp.com/).
2. My analysis indicates that the month of May is the only time when rainfall has a statistically significant positive correlation with total yearly production. Therefore, it is critical to water crops during this month if there is insufficient rainfall. While year-round irrigation of crops is prohibitively expensive, the farmers can use rainwater they collect for free in cisterns already located on the farm to water crops in the event of insufficient rainfall during May.
3. To forecast production based on monthly rainfall totals I use linear regression, elastic net, and LightGBM. I use logistic regression and LightGBM to predict the profitability of a given year. I use ARIMA and LightGBM to forecast daily rainfall totals.
4. There is no evidence that rainfall totals have changed with time. The data displays a mean-reverting pattern with no discernable statistical trend towards either an increase or decrease in rainfall.

## Methods
1. Python (Pandas, Scikit-learn, LightGBM, Optuna, Seaborn, Matplotlib, Plotly)
2. Machine learning – Regression and Classification
3. Pearson correlation and p-value estimation
5. Time series decomposition
6. ARIMA and LightGBM rain forecasting
7. Analysis of variance (ANOVA)
8. Dashboarding with plotly, dash, and heroku. Dashboard available [here](https://cafe-casa-martinez-app.herokuapp.com/).

## Products 
I provided Café Casa Martinez with a yearly timeline they can follow for steps they should perform to predict and maximize yearly profit. I also provided them with a [dashboard](https://cafe-casa-martinez-app.herokuapp.com/) so they can follow along with that timeline and predict production and profitability.

## Coffee Background
Café Casa Martinez grows three varieties of coffee, Castillo, Tabi, and Cenicafe 1. These varieties grow in the shade of trees and only fruit once per year. The plants typically flower between February and April and produce fully ripened fruit between October and December, from which the coffee beans are extracted.

Coffee plants do not produce the same quantity of beans every year, even without rainfall variability. The fruit production of any given plant depends on the number of years since that tree was planted. The production curve is as follows. In the year of sowing and the following year, no fruit is produced. In the year after that, the plant will produce fruit at 50% of its maximum. The fourth year sees maximum fruit production and each year after that sees decreasing production. For this reason, in the 6th year after sowing, the farmers perform a process called renewal, or zoca, where the plant is cut near its base. The plant then regrows from the stump and returns to maximum production after two more years. Plants may be renewed up to 5 times. The following figure shows the production fraction as a function of year covering the sowing cycle and three zoca cycles.

<img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/production_fraction_vs_year.png" width="400">

Farms like Finca Tulipan have multiple fields of coffee that were sown and renewed in different years and therefore, in any given year, they have fields in all stages of their renewal cycles. It is best practice to stagger renewal and sowing in different fields so that total production in each year stays as consistent as possible.

Café Casa Martinez provided me with daily rainfall records reaching back to October 2006, yearly coffee production totals since 2008, records of the month and day of sowing and renewal, and the number of plants in each of their 14 lots.

## Exploratory Data Analysis and Machine Learning
First, I cleaned the data, formatted them as pandas dataframes, and saved them for future use in other scripts. The script that performs these steps can be found [here](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/clean_data.ipynb). Next, I performed an exploratory data analysis to identify features that may have a correlation with target variables of interest, such as total production and production per plant. The script that performs these steps can be found [here](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/exploratory_data_analysis.ipynb). I calculated Pearson correlation coefficients and P-values between all parameters. The Pearson correlation coefficient measures the linear correlation between features in the dataset. Pearson correlation coefficient values closer to 1 (or -1) are highly correlated (or anti-correlated) and values closer to 0 indicate no correlation. The following table summarizes the strongest correlations between different features and total yearly production:

| Parameter 1      | Parameter 2 | Pearson Correlation Coefficient | P-value | Statistically Significant? |
| ---------------- | ----------- | ------------------------------- | ------ | --- |
|  Production Per Plant  | Total Production  |  0.96  |  9.4e-20 |  yes  |
|  Year  |  Total Production |  0.80 |  1.2e-13 |  yes  |
|  Total # Plants  |  Total Production  |  0.74  |  1.5e-12 |  yes  |
|  May Rainfall  |  Total Production  |  0.61  |  4.18e-7 |  yes  |
|  October Rainfall  |  Total Production  |  -0.42  |  9.5e-4 |  yes  |
|  August Rainfall  |  Total Production  |  -0.45  |  4.1e-8 |  yes  |

Production per plant and total production are highly correlated, and this makes sense because production per plant is just the production divided by the number of plants. Year and total production are also highly correlated which seems surprising at first glance. Naively, there is no reason to expect that production correlates with year, but this correlation exists because the coffee plants that were inherited with the purchase of the farm in 2006 had not been renewed recently and were therefore not producing optimally. While it is not shown in previous figure, the downward trend in plant production continues if a field is not renewed. This is why one wants to renew in the first place. Between 2008 and 2021 the owners began to renew some old fields, which increased production in a manner that happens to correlate with year. Eventually this correlation will disappear entirely if the farmers stick with the optimal renewal schedule that I will recommend later in this write-up. Since this is the second year that I’ve done this analysis I’ve already seen this correlation decrease since last year.

When the farm was purchased, there were also several empty lots that had no coffee plants sown yet. Over time, an increasing number of these empty lots were sown, which boosted total production in a manner that also correlates with year. Though now shown in the table above, the total number of plants are highly correlated with year. **The high correlations between total production and year, production per plant and year, and total number of plants and year illustrate one of our key takeaways. Sticking with the optimal plant renewal schedule and filling all available lots with coffee plants are the two most important factors that will lead to maximum production. These activities have a stronger effect on production than rainfall totals and unlike rainfall, they are completely in the farmers control.** In the coming years these correlations with year will disappear because the farm will have reached an optimal renewal schedule and they will run out of empty lots to plant or long-overdue fields to renew. At that point they will reach a stable production equilibrium and the correlation between rainfall and production is expected to become the strongest.

Our next key result is that **May is the only month where rainfall has a statistically significant positive correlation with production.** This was surprising to the owners of Café Casa Martinez because it was previously known that rainfall in the month of February causes the plants to flower. They believed rainfall during that month would correlate with production. There is, however, no correlation between February rainfall totals and production. February’s rainfall may cause the plants to flower, but this has no apparent effect on the amount of coffee that is later produced from those flowers. Two other months, October and August have statistically significant negative correlations with production. This is possible because too much rainfall can cause the coffee to rot. The plants are highly sensitive to too much rainfall. The figure below on the left shows the correlation between total production and May rainfall and the figure on the right shows production per plant versus May rainfall. The color of the point indicates the year and you can readily see the correlation between total production and year. At fixed rainfall totals of say 200 mm, you have a higher production per plant in later years that had many both more total plants and a higher fraction of renewed plants. Nonetheless, there is still a clear trend between higher total rain in May and higher production and production per plant.

<p float="left">
  <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/may_rain_vs_production.png" width="400">  
  <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/may_rain_vs_production_per_plant.png" width="400">
</p>

The black lines show the best fit linear relations between total production or production per plant and May rainfall totals as derived using linear regression. These figures are simply used for illustrative purposes. When re-running this analysis to include the 2023 data I updated the machine learning models used to predict production. Multiple features are now important rather than just May rainfall. I will go into further detail about this model below.

Using the updated best fit model on the [dashboard]( https://cafe-casa-martinez-app.herokuapp.com/predict-production) page, Café Casa Martinez can now predict their total production each year and to keep an eye on the rainfall totals during May to decide if they should be irrigating the crops with water from their cisterns.

## Predicting Production
Given there are correlations between rainfall totals and production the first machine learning model I employ is a regression model to predict total yearly production based on monthly rainfall totals. Last year I fit a simple linear regression model with just one feature, May rainfall totals, to the total production and produced the above figure on the left. This was used on the dashboard to predict yearly production, but the model was not as accurate as it could have been. The previous model performed poorly for two reasons: 1) There are only 15 years of monthly rainfall totals and production available (so only 15 observations), which is very small, and 2) there are confounding variables we cannot model that affect the relationship between rainfall totals and production. 

Probable confounding variables include total sunlight received by the plants, temperature, humidity, wind speed and direction, fraction of fields producing optimally, pest load, and likely others we cannot think of. I tried, but cannot gather better weather data for the farm because what we have represents the best information that exists. The nearest weather station to the farm is close as the crow flies, but it’s at a much lower altitude than the farm and at a much drier climate. There is no way to get information such as wind speed and direction, humidity, cloud cover, etc. We only have rainfall so we will have to make due with that. 

We cannot wait a few hundred years to get enough observations, however we can simulate more observations. Making the assumption that the observed covariances and means of all the data are the true means and covariances, I simulate 500 new observations so we have more data to train the models on. This is a logical solution given the impossibility of gathering more data from internal/external sources. The only caveat that needs to be remembered is that there is likely a difference between the observed means and covariances between parameters, and there is also likely a time component that is unaccounted for here. I perform the simulation and modeling of production in [this script]( https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/model_production.ipynb) but I’ll summarize key points below.

The first improvement over last years model comes from combining correlated features together. March and April rainfall totals have a Pearson correlation coefficient of 0.5 and June and July have a coefficient of 0.75. There are a few ways to handle this, but what I chose to do is simply engineer new features by adding March and April together than June and July together. I then deleted the individual features of each of these months in order to remove the correlations. Correlated features will be a problem for linear regression but not for LightGBM regression. Still, I later fit LightGBM with these engineered features instead of the original because it won’t make much of a difference. I also remove December rainfall from the feature set because December is when the coffee is picked. Rain that falls after a plant is picked will not contribute to how much coffee is grown, and I found previously that December is not an important month.

First I fit a linear regression model with k-fold cross validation with 10 folds and find a mean R-squared for the linear regression model of 0.6523. R-squared explains the fraction of the variance in the target variable that is explained by the model. This is already a big improvement over last year’s model, which was above 0.4. The RMSE is 1115.6 kg, and for reference, the median production value is about 3600 kg. This corresponds to an average uncertainty of order 30%. The most important features are May and October rainfall. May is positively correlated with production and October is negatively correlated.

Next I fit the LightGBM model with hyperparameter tuning by Optuna. Optuna is a hyperparameter optimization framework that finds the best hyperparameters. Again using 10-fold cross validation I find the average RMSE is about 1215.4 and the average R-squared is about 0.58. The most important features are October, then May rainfall. 

Finally, since a linear regression model performed better than LightGBM, I fit an elastic net model to see if I can improve on the baseline model. I use a randomized search to find the best hyperparameters and find an l1 ratio of 0.25 and a learning rate of 0.01 to be optimal. After 10-fold cross validation I find an R-squared of 0.6524 and an average RMSE of 1115.5, which slightly beats the baseline linear regression model. The important features are again May and October, with May being positive and October being negative. Because elastic net performed slightly better in the average RMSE and MAE across folds, I re-train the elastic net model on all the data and save it for use in the dashboard.

## Classification as Profitable vs Unprofitable
Another important question to address is whether a year will be profitable or not. The script that performs the classification analysis is hosted [here](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/model_simulated_profitability.ipynb). First, I engineer a new target variable that is True if the production is greater than 3500 kg and False if it is less than. This profitability cutoff was provided to me by Café Casa Martinez. My goal is to fit classification models to monthly rainfall data and predict whether a year will be profitable. I follow the same procedure as I did when predicting production from monthly rainfall totals. I use the same simulated data as before and I use the combined march and April and June and July features. There is a balance between profitable and unprofitable years with 47% of years being profitable and 53% of years being unprofitable. This is cause for concern, indicating that the company might fail in the long run, but since it is based on simulated data, some of which is generated from unprofitable years that were the result of old crops on the farm and having few fields with crops, it is not possible to read too much into it at this point. The farmers should certainly pay attention to the profitability in the coming years though to make sure they should still stay open. I fit logistic regression and LightGBM models. First I performed hyperparameter tuning with via randomized search on the logistic regression model and found the best fit regularization coefficient of C=0.17 and a best fit penalty type of LASSO. The most important positive feature coefficient is May rainfall, and the most important negative feature is October rainfall, similar to what we saw in the regression models. No other features are particularly important. 

Next I use the best hyperparameters to fit a 10-fold cross validation model to get a good estimate of our model performance. The performance across all folds is as follows:

| | precision | recall | f1-score | support | 
|--- | --- | --- | --- | --- | 
|False | 0.78 | 0.80 | 0.79 | 239 | 
|True | 0.82 | 0.80 | 0.81 | 276 | 
| |  |  |  |  | 
|accuracy | | | 0.80 | 515 | 
|macro avg | 0.80 | 0.80 | 0.80 | 515 | 
|weighted avg | 0.80 | 0.80 | 0.80 | 515 | 

These metrics are decent, with all metrics hovering around 80%. I fit a LightGBM model next, to see if we can improve. I also compare the two models using the area under the receiver operator curve (AUROC) in order to determine which model is best given that we will not use a probability threshold of 0.5. I will go into this in more detail below.

Next I use Optuna to optimize the hyperparameters of the LightGBM classification model, and I fit the model with the best fit hyperparameters using 10-fold cross validation. The most important features are again May and October rainfall, but this time October is more important than May. The metrics summary is as follows:

| | precision | recall | f1-score | support | 
|--- | --- | --- | --- | --- | 
|False | 0.78 | 0.77 | 0.77 | 239 | 
|True | 0.80 | 0.81 | 0.80 | 276 | 
| |  |  |  |  | 
|accuracy | | | 0.79 | 515 | 
|macro avg | 0.79 | 0.79 | 0.79 | 515 | 
|weighted avg | 0.79 | 0.79 | 0.79 | 515 | 

This is slightly worse performance than the logistic regression model. These metrics and those from the logistic regression model are computed using a probability threshold of 0.5, but we do not want to use a threshold of 0.5 because false positives and false negatives are not equal in this case. A false negative implies that there would have been enough rain during the month of May to make the year profitable, but the farm watered anyway. The downside for a false negative is that you simply increase the production by increasing the water each plant has access to. There is a limit to this, in that overwatering plants can cause the roots to rot, leading to decreased production. The farmers are acutely aware that they should not water plants just because our model recommends it, if they are at risk of being over-watered. A false positive implies that the model predicts there will be enough rain for a profit, and therefore the farmers do not water the crops, which results in an unprofitable year. This is much worse than a false negative, and therefore we want to choose a threshold that minimizes false positives. The farm is willing to accept a false positive rate of 10%. For the logistic regression model, this corresponds to a threshold of 0.68, which has an associated 63% true positive rate. For the LightGBM model a 10% false positive rate has a true positive rate of 61% with a threshold of 0.71. Since logistic regression has a higher true positive rate at its threshold that is our best model, which we will use in the dashboard for predictions.

Similar to the regression model I describe above, the key takeaway is that if rainfall is low during the month of May, the farmers should water the crops with the water collected in their cisterns. This is the most important thing they can do to increase production after following the optimal renewal or zoca schedule.

## Weather Forecasting
Café Casa Martinez has been collecting daily rainfall totals on their farm between November 2006 and the present day. This time series data can be decomposed into seasonal and trend components, which can provide valuable insights and form the basis for a forecasting model. The current seasonal decomposition and the old forecasting model can be found in [this script ]( https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/rain_seasonal_decomposition.py), and the updated 2023 forecasting model can be found in [this notebook](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/rain_time_series_forecasting.ipynb). 

In the first script, the seasonal decomposition is performed using the seasonal decomposition routine from the statsmodels Python package. This is performed on monthly rainfall totals instead of daily rainfall since it makes more sense to predict the monthly total than the rain on any given day. It does not matter much whether it rained on the third or the fourth of a given month, but rather what the total rainfall is in a month, so it’s better to aggregate the rainfall in this way. The first figure below shows the observed rainfall totals for each month, the underlying rainfall trend, the seasonal trend, and the residuals of the decomposition. The seasonal decomposition is done using the seasonal_decompose from the statsmodels python package. The bottom left figure shows a zoom in on the trend curve, and the right figure shows the seasonal trend repeating for a single year for better visualization.

<p float="left">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/seasonal_decomposition_plot.png" width="600">
</p>

<p float="left">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/seasonal_trend_plot.png" width="400">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/seasonal_curve_plot.png" width="450">
</p>

The median of the residuals of the decomposition is -4 with a standard deviation of 51. This is consistent with 0, implying the decomposition is a good fit to the data. The seasonal trend component closely matches the median rainfall in each month across all years, which confirms the validity of this decomposition. Rainfall is lowest in January and highest in May. [Wang et al. 2006]( https://www.researchgate.net/publication/220451959_Characteristic-Based_Clustering_for_Time_Series_Data) describe metrics for evaluating the strength of the seasonal component and trend component strengths. Each ranges from 0 to 1, where 1 implies a strong component strength. The trend strength is weak at 0.13 and the seasonal strength is moderately strong at 0.54, indicating that our data has a seasonality trend with no additional underlying trend. This is reassuring because it suggests that there is no evidence of a long-term trend of decreasing rainfall due to global warming, as is observed in some places in Colombia. 

The updated forecasting model is performed in the [second script](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/rain_time_series_forecasting.ipynb). Both LightGBM and ARIMA models are used for forecasting, with LightGBM performing slightly better. The ARIMA forecasting model works best on data that is stationary with respect to time, meaning that the statistical properties of the data do not depend on time. To confirm stationarity, I run KPSS and adfuller tests, which confirm that the rainfall data are indeed stationary. Next, I calculate the Hurst exponent, which reveals that the data are mean reverting, meaning that there are deviations away from the mean seasonal trend, but the data generally revert to the mean. This is reassuring because it implies that there is no evidence of a long-term trend of decreasing rainfall. Next, I examined the autocorrelation and partial autocorrelation plots for the data at different lags and observed significant autocorrelation, indicating the presence of a seasonal trend in the data. This seasonality has a period of 12 months, so I fit several different ARIMA models using a seasonality order with a period of 12 months. To model the data, I experimented with several ARIMA models, each using a seasonality order with a period of 12 months. I varied the inputted order and seasonal order coefficients for each model, and ultimately found the ARIMA model with autoregressive coefficient p=1, differences coefficient d=0, moving average coefficient q=2, and seasonal order coefficients P=1, D=0, Q=2, and seasonal period s=12 had the lowest AICC.

I fit both models 5 times using 5-fold time series cross validation, which preserves the time ordering of the data during training and prevents data leakage. For the LightGBM model I used a learning rate of 0.01 with a maximum depth of 3, 2000 estimators, and an early stopping round of 50, meaning that training would stop if the validation set metrics did not improve for 50 rounds. The mean RMSE for LightGBM models is 52.8, and for ARIMA it is 53.5. Therefore I selected the LightGBM and saved it for future forecasting.

## Zoca Schedule
We found during the exploratory analysis that sowing all lots with coffee plants and renewing them every 6 years is critical for maximizing production. Surprisingly, these factors have a stronger impact on production than rainfall totals. After examining the lot renewal schedule, I identified four lots that are currently overdue for renewal. While it is possible to renew all four lots immediately, it would not be optimal because this would result in many plants not producing at the same time. To maintain production consistency between years, it is best to renew an equal number of plants every year, if possible. With this in mind, I created an optimal zoca schedule for the next 60 years by dividing the four overdue lots into bins with close-to-equal numbers of plants. I wrote a script posted [here]( https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/zoca_schedule.py) that creates lists the years of suggested renewal for each field for the next 60 years. Additionally, I provided Café Casa Martinez with an [interactive map](https://cafe-casa-martinez-app.herokuapp.com/) where they select a year and see the location and ancillary information about the lots to renew in the given year. Below is a table showing the suggested lots to renew each year for the next 4 years. 

| 2022 |  |  2023 |  | 2024 | | 2025 |  | 
| --- | --- | --- | --- | ---- |  ---- | ---- |  --- |
| Lot | # plants | Lot | # plants | Lot | # plants | Lot | # plants |
| Sancocho 2 | 5000 | Ceiba 4 | 243 |  Ceiba 3|  900 | Ceiba 1 | 925 | 
|                       |            | Sancocho 1 | 1488 | Anacos 2 | 680 | Cedral | 2025 |
|                       |            |  | | La Hoya | 1130 | Anacos 1 | 2487 |
|                       |            |  | | Arrayanes | 906 |  |  |
|  |
| Total | 5000 |  | 1731 |  | 3616 |  | 5437 |

We can see that the number of plants renewed each year are not consistent, as they would be in the ideal case. The reason for this is that it is better to keep up with the current renewal schedule for fields that were already on schedule. Café Casa Martinez would lose production, unnecessarily, if they failed to keep up with the renewal schedule for lots that are not currently overdue. The current schedule for renewal for the 10 fields that are not late is not perfect, meaning that the number of plants renewed each year is not very close to uniform, but it is good enough and not worth the loss of production to correct.

## Final Product – Timeline
Every year, I recommend that Café Casa Martinez take the following steps:
1. In January, use the ARIMA model to predict monthly rainfall for the entire year. Next, predict total production with the May rainfall estimate and predict whether the year will be profitable with May and August rainfall estimates. This is their first, coarsest estimate of production/profitability for the year.
2. During the month of May, if rainfall is abnormally low irrigate at least some of the crops with water from the rainfall cisterns on the farm. This will mitigate the negative effect on production and profitability.
3. In June, use the observed May rainfall totals to run a more accurate projection of the production for the year.
4. In September, use the observed May and August rainfall totals to run a more accurate projection of the profitability for the year.
5. In December when the year’s production and rainfall totals are recorded, we will update the models with new data for better predictions going forward.
