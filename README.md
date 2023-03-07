# Café Casa Martinez Data Science Consulting Project
## Table of Contents
* [Goals](https://github.com/pdrew32/Cafe-Casa-Martinez#goals)
* [Summary of Results](https://github.com/pdrew32/Cafe-Casa-Martinez#summary-of-results)
* [Methods](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/README.md#methods)
* [Products](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/README.md#products)
* [Background](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/README.md#background)
* [Exploratory Data Analysis and Machine Learning](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/README.md#exploratory-data-analysis-and-machine-learning)
* [Classification as Profitable vs Unprofitable](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/README.md#classification-as-profitable-vs-unprofitable)
* [Weather Forecasting](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/README.md#weather-forecasting)
* [Zoca Schedule](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/README.md#zoca-schedule)
* [Final Product – Timeline](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/README.md#final-product---timeline)

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

I performed k-fold cross validation with 10 folds and find a mean R$^2$ for the linear regression model of 0.65. R$^2$ explains the fraction of the variance in the target variable that is explained by the model. This is already a big improvement over last year’s model, which was above 0.4. The RMSE is about 1000 kg, and for reference, the median production value is about 3600 kg. This corresponds to an average uncertainty of order 30%. The most important features are May and October rainfall. May is positively correlated with production and October is negatively correlated.

Next I fit the LightGBM model with hyperparameter tuning by Optuna. Optuna is a hyperparameter optimization framework that finds the best hyperparameters. Again using 10-fold cross validation I find the average RMSE is about 1200 and the average R$^2$ is about 0.6. The most important features are again May and October rainfall. May is positively correlated with production and October is negatively correlated.

Finally, since a linear regression model performed better than LightGBM, I fit an elastic net model to see if I can improve on the baseline model. I use a randomized search to find the best hyperparameters and find an l1 ratio of 0.25 and a learning rate of 0.01 to be optimal. After 10-fold cross validation I find an R$^2$ of 0.652 and an average RMSE of 1115, which slightly beats the baseline linear regression model. The important features are again May and October, with May being positive and October being negative. I re-train the elastic net model on all the data and save it for use in the dashboard.

## Classification as Profitable vs Unprofitable
The script that performs the classification analysis is hosted [here](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/classification_profit.py). First, I create a new target variable that is 1 if the production is greater than 3500 kg and 0 if it is less than. This profitability cutoff was provided to me by Café Casa Martinez. Then, because I am working with numerical features and a categorical target, I run an ANOVA and compute the f-statistic to find the features that correlate most strongly with the target variable. As we would expect, I find that rainfall during the month of May is significantly correlated with profitable years (p-value < 0.05), but surprisingly I also find that August rainfall is significantly anti-correlated with profitable years. While this anti-correlation may not be causal, it is significant and therefore I use this feature in the classification routine. During testing, I ran logistic regression on all features (monthly rainfall for all months) with total production as a target variable and found worse accuracy, precision, and recall than fits with just May and August rainfall totals. I will provide more detail on this cross-validation routine next.

I ran numerous classification algorithms to determine which provides the highest accuracy, including logistic regression, Gaussian naïve Bayes, k nearest neighbors, support vector machines, decision tree, and random forest. I run each of these after standardizing the features using the standard scaler from sklearn. The highest accuracy is achieved with logistic regression, as determined by stratified k-fold cross validation with 4 folds. I used stratified k-fold because there is a 70/30% split in the target variable, where 70% of years were unprofitable.  Using regular k-fold cross validation results in worse performance across the board. The table below shows the accuracy of each of these methods.

| Algorithm      | Mean Stratified K-fold Cross Validation Accuracy | 
| ---------------- | ----------- |
| Logistic Regression  |  83%  |
| Support Vector Machine |  71%  |
| K Nearest Neighbors   | 69%   |
| Decision Tree  |  69%  | 
| Random Forest |  69%   |
| Gaussian Naïve Bayes |   63%      |

After determining that logistic regression is the best algorithm in this case, I trained using all data and found the following confusion matrix:

Overall logistic Regression Accuracy = 92%
| Target | Precision | Recall | F1-score | 
| ---- | ---- | ---- | ---- | 
| 0 (unprofitable) | 0.89 | 1.0 | 0.94 |  
| 1 (profitable) | 1.0 | 0.8 | 0.89 |  

These values suggest logistic regression does acceptably well. The overall accuracy, precision, recall, and F1-scores are all high. A grid search of optimal hyperparameters reveals the following values are best, C=1, penalty=l2. Now that we have an acceptable model, I create a model to forecast rainfall for the current year to predict the production from the beginning of the year, rather than waiting until the end of August.

## Weather Forecasting
Café Casa Martinez has daily rainfall totals on the farm from November 2006 through the present day. This time series data can be decomposed into seasonal and trend components which provide useful insights as well as form the basis for a forecasting model. The script where I run the forecasting model can be found [here]( https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/rain_seasonal_decomposition.py). First, I run a seasonal decomposition using the seasonal decomposition routine from the statsmodels Python package. The first figure below shows the observed rainfall totals for each month, the underlying rainfall trend, the seasonal trend, and the residuals of the decomposition. This seasonal decomposition is done with the seasonal_decompose from the statsmodels python package. The bottom left figure shows a zoom in on the trend curve, and the right figure shows the seasonal trend repeating for a single year so more detail may be seen.

<p float="left">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/seasonal_decomposition_plot.png" width="600">
</p>

<p float="left">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/seasonal_trend_plot.png" width="400">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/seasonal_curve_plot.png" width="450">
</p>

The median of the residuals of the decomposition is -4 with a standard deviation of 51. This is consistent with 0, implying the decomposition is a good fit to the data. The seasonal trend component closely matches the median rainfall in each month across all years, which confirms the validity of this decomposition. Rainfall is lowest in January and highest in May. [Wang et al. 2006]( https://www.researchgate.net/publication/220451959_Characteristic-Based_Clustering_for_Time_Series_Data) describe metrics for evaluating the strength of the seasonal component and trend component strengths. Each range from 0 to 1, where 1 implies a strong component strength. The trend strength is weak at 0.13 and the seasonal strength is moderately strong at 0.54, implying our data has a seasonality trend with no additional underlying trend. This is reassuring because it suggests rainfall is not decreasing with time due to global warming, as it is in some places in Colombia. 

Next, I begin the forecast modeling. I split the data into training and testing sets with a split of the first 12 years in training and the last year in testing, as we will want to predict a single, whole year at a time, and it is better to train the forecasting model on as much data as possible. I use the ARIMA forecasting model, which works best on data that is stationary with respect to time. That is to say, the statistical properties of the data do not depend on time. To confirm stationarity, I run KPSS and adfuller tests, which confirm that the rainfall data are indeed stationary. Next, I calculate the Hurst exponent, which reveals if the data is mean reverting, has properties similar to a random walk, or if there is a trend to the data with respect to time.  I find that the data are mean reverting, meaning that there are deviations away from the mean seasonal trend, but the data generally revert to the mean. This is, again, reassuring because it implies that there is no evidence of a long-term trend of decreasing rainfall. Next, I check the autocorrelation and partial autocorrelation plots for the data as a function of different lags and find significant autocorrelation that corresponds to the seasonal trend, which is to say that there is a seasonality trend in the data. This seasonality has a period of 12 months, so I fit several different ARIMA models using a seasonality order with a period of 12 months. For each model I vary the inputted order and seasonal order coefficients. I find the lowest AICC corresponds to the ARIMA model with autoregressive coefficient p=1, differences coefficient d=0, moving average coefficient q=2, and seasonal order coefficients P=1, D=0, Q=2, and seasonal period s=12.

The top panel of the figure below shows the forecasted rainfall for 2020 as a function of month in orange with true rainfall totals in blue. The bottom panel shows the residuals, or reality minus forecast. We see the forecast is not perfect but does well at modeling the true rainfall totals. The forecast is quite accurate for May but is less accurate in August. The forecast over predicts the August rainfall by about 30%. 

<p float="left">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/forecast_vs_reality.png" width="400">
</p>

Using these forecasted rainfall totals to predict the profitability of the year 2020 results in a prediction that 2020 will be unprofitable with 93% confidence. This is however incorrect. The year 2020 was the most profitable year of all time for Café Casa Martinez. If instead we use real, observed data, the prediction is that 2020 would be profitable, with 93% confidence. The fact that the forecast over-predicts August rainfall totals is what results in the classification prediction of no profit with very high confidence since profits are anti-correlated with rainfall during the month of August.

## The (Im)Possibility of Inexpensively Gathering More Data
In an ideal world we would gather more data for both the classification and forecasting routines to improve accuracy of both models. For the case of improving the classification accuracy, the only way to increase the amount of data is to wait years to gather more rainfall and production data. As I mentioned previously, now that Café Casa Martinez will be following the optimal zoca schedule, within a few years they will reach a maximum equilibrium production state because all fields will be planted and will be renewed every 6 years. Retraining regression and/or classification models on data taken only after this equilibrium state is reached should eliminate the confounding variables associated with having fields that are not sown or producing optimally. Another avenue for gathering more data would be to start collecting other ancillary weather data, such as temperature, cloud cover, and humidity.

With regard to improving the weather forecasting, more data may increase accuracy, however the rainfall totals recorded by Café Casa Martinez represent the best available data for a few reasons. First, the nearest weather station to the farm is at a much lower altitude in a place with a drier climate, even though it is only about a 20-minute drive away. Scraping weather data from the web or purchasing access to a weather API would not be helpful in this case because the available weather stations do not record the true weather conditions at high elevation where the farm is. The farm has its own microclimate. At any rate, the rainfall data available online for the region surrounding Finca Tulipan does not extend as far back in time as the rainfall totals recorded by Café Casa Martinez. The data they presented me represents the best data in existence, unless a neighboring farm has recorded rainfall data for longer. For this reason, I do not believe it is possible to increase forecasting accuracy without waiting years for more data to become available.

## Zoca Schedule
We saw during the exploratory analysis that the most important thing Café Casa Martinez can do to maximize production is sow all their lots with plants and keep up with the renewal schedule every 6 years.  Surprisingly, these have a stronger effect on production than rainfall totals. Upon inspection of the lot renewal schedule, I found four lots that are currently overdue for renewal. While it would be possible to renew all four lots immediately, it would not be ideal because that would lead to many plants not producing at once. It is better to keep the production as consistent as possible between years which requires an equal number of plants to be renewed every year, in the ideal case. With this in mind, I tried to divide the four overdue lots into bins with close-to-equal numbers of plants in order to make an optimal zoca schedule for the coming years. I wrote a script posted [here]( https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/zoca_schedule.py) that creates a pandas dataframe listing the years of suggested renewal for each field for the next 60 years. Below is a table showing the suggested lots to renew each year for the next 4 years. Additionally, I provided Café Casa Martinez with an [interactive map](https://cafe-casa-martinez-app.herokuapp.com/) where they select a year and see the location and ancillary information about the lots to renew in the given year.

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

## Final Product - Timeline
Every year, I recommend that Café Casa Martinez take the following steps:
1. In January, use the ARIMA model to predict monthly rainfall for the entire year. Next, predict total production with the May rainfall estimate and predict whether the year will be profitable with May and August rainfall estimates. This is their first, coarsest estimate of production/profitability for the year.
2. During the month of May, if rainfall is abnormally low irrigate at least some of the crops with water from the rainfall cisterns on the farm. This will mitigate the negative effect on production and profitability.
3. In June, use the observed May rainfall totals to run a more accurate projection of the production for the year.
4. In September, use the observed May and August rainfall totals to run a more accurate projection of the profitability for the year.
5. In December when the year’s production and rainfall totals are recorded, we will update the models with new data for better predictions going forward.
