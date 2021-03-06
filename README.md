# Café Casa Martinez Data Science Consulting Project
## Summary
[Café Casa Martinez](https://www.cafecasamartinez.com/) is a small, organic coffee company with a farm called Finca Tulipan in the Santander department of Colombia. 

## Goals
1. Look for evidence of the effect of global warming on rainfall totals over the last 15 years. 
    1. Some coffee farms in Colombia are closing because global warming has decreased rainfall making them unprofitable. 
    2. If there is evidence of decreasing rainfall, try to determine which year the farm is projected to become unprofitable.
    3. Investigate the possibility to fully irrigate the farm by that date. Investigate during which months rainfall is critical.
2. Find ways to increase production based on available data.

## Summary of Results
1. There is no evidence that rainfall totals have changed with time. The data are characterized as mean-reverting with no statistical trend towards increasing or decreasing rainfall.
2. A statistical analysis demonstrates the most important thing Café Casa Martinez can do to increase production is regular upkeep of their fields. This practice, called renewal or zoca, has a stronger effect on total yearly production than rainfall. 
3. The only month where rainfall correlates significantly with total yearly production is the month of May. It is critical to water crops only during this month in the event of low rainfall totals. This is critical information because year-round irrigation of crops is prohibitively expensive but some crops may be watered using existing rainwater cisterns for the period of a month.

## Methods
1. Python (pandas, numpy, scikit-learn, matplotlib, plotly)
2. Pearson correlation and p-value estimation
2. Machine learning – linear regression
3. Machine learning – classification 
4. Time series decomposition
5. ARIMA weather forecasting
6. Dashboarding with plotly, dash, and heroku. Dashboard available [here](https://cafe-casa-martinez-app.herokuapp.com/).

## Products 
I provided Café Casa Martinez with a yearly timeline they can follow for steps they perform to predict and maximize yearly profit. I also provided them with a [dashboard](https://cafe-casa-martinez-app.herokuapp.com/) so they can follow along with that timeline and predict production and profitability.

## Analysis
They grow three varieties of coffee, Castillo, Tabi, and Cenicafe 1. These varieties grow in the shade of trees and only fruit once per year. The plants typically flower between February and April and produce fully ripened fruit between October and December, from which the coffee beans are extracted.

Coffee plants do not produce the same quantity of beans every year, even without rainfall variability. The fruit production of any given plant depends on the number of years since that tree was planted. The production curve is as follows. In the year of sowing and the following year, no fruit is produced. In the year after that, the plant will produce fruit at 50% of its maximum. The fourth year sees maximum fruit production and each year after that sees decreasing production. For this reason, in the 6th year after sowing, the farmers perform a process called renewal, or zoca, where the plant is cut near its base. The plant then regrows from the stump and returns to maximum production after two more years. Plants may be renewed up to 5 times. The following figure shows the production fraction as a function of year covering the sowing cycle and three zoca cycles.

<img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/production_fraction_vs_year.png" width="400">

Farms like Finca Tulipan have multiple fields of coffee that were sown and renewed in different years and therefore, in any given year, they have fields in all stages of their renewal cycles. It is best practice to stagger renewal and sowing in different fields so that total production in each year stays as consistent as possible.

Café Casa Martinez provided me with daily rainfall records reaching back to October 2006, yearly coffee production totals since 2008, records of the month and day of sowing and renewal, and the number of plants in each of their 14 lots. With this data I worked toward the following key goals:

# Exploratory Data Analysis and Machine Learning
First, I cleaned the data, formatted them as pandas dataframes, and saved them for future use in other scripts. The script that performs these steps can be found [here](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/clean_data.py). Next, I performed exploratory data analysis to determine which features might correlate with target variables of interest, such as total production and production per plant. I calculated Pearson correlation coefficients and P-values between all parameters. The Pearson correlation coefficient is a measure of the linear correlation between features in the dataset. Pearson correlation coefficient values closer to 1 (or -1) are highly correlated and values closer to 0 are uncorrelated. The following table summarizes the strongest correlations between features and target variables of interest:

| Parameter 1      | Parameter 2 | Pearson Correlation Coefficient | P-value | Statistically Significant? |
| ---------------- | ----------- | ------------------------------- | ------ | --- |
|  Total Production  |  Year  |  0.88 |  2.2e-10 |  yes  |
|  Production Per Plant  |  Year  |  0.79  |  5.5e-8 |  yes  |
|  Total Number of Plants  |  Year  |  0.75  |  3.8e-8 |  yes  |
|  Total Number of Plants  |  Total Production  |  0.68  |  5.0e-7 |  yes  |
|  Production Per Plant  |  Total May Rainfall  |  0.66  |  1.4e-5 |  yes  |
|  Total Production  |  Total May Rainfall  |  0.64  |  3.6e-5 |  yes  |

The total production and production per plant are both highly correlated with year. At first glance this may seem surprising, but these correlations are seen because the coffee plants that were inherited with the purchase of the farm in 2006 had not been renewed recently, and were therefore not producing optimally.  While it is not shown in the figure above, the downward trend in plant production continues if a field is not renewed, which is why one wants to renew in the first place. Between 2008 and 2021 the owners began to renew some old fields, which increased production in a way that correlates with year.

When the farm was purchased, there were also several lots that had no coffee plants yet. Over time, an increasing number of these empty lots were sown, which boosted total production in a manner that correlates with year. This is verified by the fact that the third highest correlation in the data is between the total number of plants and year. **These three strongest correlations illustrate our first key takeaway. Sticking with the optimal plant renewal schedule and filling all available lots with coffee plants are the two most important factors when trying to maximize production. Surprisingly, these have a stronger effect on production than rainfall totals.** In the coming years these correlations with year will disappear because the farm will have reached an optimal renewal schedule and they will run out of empty lots to plant or long-overdue fields to renew. At this point they will reach a stable production equilibrium and the correlation between rainfall and production should become the strongest.

Our next main result is that **May is the month where rainfall correlates most strongly with production.** This is surprising to the owners of Café Casa Martinez because it was previously known that rainfall in the month of February causes the plants to flower. It was therefore assumed that rainfall during that month would correlate with production. There is, however, no correlation between February rainfall totals and production. The Pearson correlation coefficient between February rainfall and production is -0.2, which is much closer to 0 than to positive or negative 1. The probability that that is a chance correlation is high, given that the associated P-value associated is higher than 0.05 (actual value is 0.14). February’s rainfall may cause the plants to flower, but it is rainfall during the month of May that contributes most strongly to the production of coffee beans. I find no statistically significant correlations between any other month and total production or production per plant. The figure below on the left shows the correlation between total production and May rainfall and the figure on the right shows production per plant versus May rainfall.

<p float="left">
  <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/may_rain_vs_production.png" width="400">  
  <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/may_rain_vs_production_per_plant.png" width="400">
</p>

The points in each figure represent individual years of May rainfall totals versus total production for the year (left figure) and total production per plant (right figure). The color of each data point shows the year. From this figure we can see the strong correlation between year and total production. We see that the higher the rainfall in May, the higher the total production and the production per plant for the year.  The black lines show the best fit correlations between total production or production per plant and May rainfall totals as derived using ordinary least squares linear regression.

With this measured correlation, Café Casa Martinez can now predict their total production after observing the rainfall totals for the month of May each year. This is helpful to know because total production is not known until the end of the year. However, the data is scattered considerably about the best fit so the prediction may not be as accurate as one would like. For instance, for May rainfall totals of around 200 mm, corresponding values of production range from ~2000 to ~5500 kg. The square root of the mean square error is 1450 kg, which corresponds to about 50% of the median of production across all years. That is to say, a characteristic uncertainty associated with this model is about 50%.  The R<sup>2</sup> score for the regression fit between May rainfall and total production is 0.41, meaning only 41% of the variance seen in this correlation explainable by the variance in May rainfall.  There are several confounding factors that contribute in unknown ways toward production.  
Because a typical uncertainty in the regression is of order 50%, we may have better predictive power if we simplify the problem by recasting it in terms of a classification problem where years are classified in terms of profitable versus non profitable, rather than trying to predict the number of kilograms of roasted coffee produced.

The script that performs the classification analysis is hosted [here](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/classification_profit.py). First, I create a new target variable that is 1 if the production is greater than 3500 kg and 0 if it is less than. Then because I am working with numerical features and a categorical target, I run an ANOVA and compute the f-statistic to find the features that correlate most strongly with the target variable. As I expected, I find that rainfall during the month of May is significantly correlated with profitable years (p-value < 0.05), but surprisingly I also find that August rainfall is significantly anti-correlated with profitable years. While this anti-correlation is not causal, because all available evidence suggests that a lack of rainfall does not lead to increased production in plants, this correlation is significant and therefore I use this feature in the classification routine. During testing, I ran logistic regression on all features (monthly rainfall for all months) with total production as a target variable and found worse accuracy, precision, and recall than fits with just May and August rainfall totals. I will provide more detail on this cross-validation routine in the next paragraph.

Next, I ran numerous classification algorithms to determine which provides the best accuracy, including logistic regression, Gaussian naïve Bayes, k nearest neighbors, support vector machines, decision tree, and random forest. I run each of these after standardizing the features using the standard scaler from sklearn. The highest accuracy is achieved with logistic regression, as determined by stratified k-fold cross validation with 4 folds. I used stratified k-fold because there is a 70/30% split in the target variable, where 70% of years were unprofitable.  Using regular k-fold cross validation results in worse performance across the board. The table below shows the accuracy of each of these methods.

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

These values suggest logistic regression does acceptably well. A grid search of optimal hyperparameters reveals the default values are best, C=1, penalty=l2. Now that we have an acceptable model I want to create a model to forecast rainfall for the current year in order to predict the production from the beginning of the year, rather than waiting until the end of August.

# Weather Forecasting
Café Casa Martinez has daily rainfall totals on the farm from November 2006 through the present dat. This time series data can be decomposed into seasonal and trend components which may provide useful insights as well as form the basis for a forecasting model. The script where I run the forecasting model can be found [here]( https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/rain_seasonal_decomposition.py). First I run a seasonal decomposition using the seasonal decomposition routine from the statsmodels Python package. The first figure below shows the observed rainfall totals for each month, the underlying rainfall trend, the seasonal trend, and the residuals of the decomposition. This seasonal decomposition is done with the seasonal_decompose from the statsmodels python package. The bottom left figure shows a zoom in on the trend curve, and the right figure shows the seasonal trend repeating for a single year so more detail may be seen.

<p float="left">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/seasonal_decomposition_plot.png" width="600">
</p>

<p float="left">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/seasonal_trend_plot.png" width="400">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/seasonal_curve_plot.png" width="450">
</p>

The median of the residuals of the decomposition is -4 with a standard deviation of 51. This is consistent with 0, implying the decomposition is a good fit to the data. The seasonal trend component closely matches the median rainfall in each month across all years, which confirms the validity of this decomposition. Rainfall is lowest in January and highest in May. [Wang et al. 2006]( https://www.researchgate.net/publication/220451959_Characteristic-Based_Clustering_for_Time_Series_Data) describe metrics for evaluating the strength of the seasonal component and trend component strengths. Each ranges from 0 to 1, where 1 implies a strong component strength. The trend strength is weak at 0.13 and the seasonal strength is moderately strong at 0.54, implying our data has a seasonality trend with no additional underlying trend. This is reassuring because one of my hypotheses going into this project was that rainfall could be decreasing with time due to global warming, as it is in some places in Colombia. This would have meant that the farm would eventually fail to produce crops, but this analysis shows there is no such trend.

With the seasonal decomposition done, next I begin the forecast modeling. I split the data into training and testing sets with a split of the first 12 years in training and the last year in testing, as we will want to predict a single, whole year at a time and it is better to train the forecasting model on as much data as possible. I use the ARIMA forecasting model, which works best on data that is stationary with respect to time. That is to say, the statistical properties of the data do not depend on time. To confirm stationarity, I run KPSS and adfuller tests, which confirm that the rainfall data are indeed stationary. Next I calculate the Hurst exponent, which reveals if the data is mean reverting, has properties similar to a random walk, or if there is a trend to the data with respect to time.  I find that the data are mean reverting, meaning that there are deviations away from the mean seasonal trend, but the data generally revert to the mean. This is, again, reassuring because it implies that there is no evidence of a long term trend of decreasing rainfall. Next I check the auto-correlation and partial auto-correlation plots for the data as a function of different lags and find significant auto-correlation that corresponds to the seasonal trend, which is to say that there is a seasonality trend in the data. This seasonality has a period of 12 months, so I fit a number of different ARIMA models using a seasonality order with a period of 12 months. For each model I vary the inputted order and seasonal order coefficients. I find the lowest AICC corresponds to the ARIMA model with autoregressive coefficient p=1, differences coefficient d=0, moving average coefficient q=2, and seasonal order coefficients P=1, D=0, Q=2, and seasonal period s=12.

The top panel of the figure below shows the forecasted rainfall for 2020 as a function of month in orange with true rainfall totals in blue. The bottom panel shows the residuals, or reality minus forecast. We see the forecast is not perfect but does fairly well modeling the true rainfall totals. The forecast is quite accurate for May, but is less accurate in August. The forecast over predicts the August rainfall by about 30%. 

<p float="left">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/forecast_vs_reality.png" width="400">
</p>

Using these forecasted rainfall totals to predict the profitability of the year 2020 results in a prediction that 2020 will be unprofitable with 95% confidence. This is however incorrect. The year 2020 was the most profitable year for Café Casa Martinez in their existence. The first thing I checked upon realizing this is what the classification prediction would be if the forecast perfectly predicted the rainfall totals in May and August. The prediction is actually that 2020 would be unprofitable, though with 55% confidence. The fact that the forecast over-predicts August rainfall totals is what results in the classification prediction of no profit with very high confidence, since profits are anti-correlated with rainfall during the month of August.

# The (Im)Possibility of Inexpensively Gathering More Data
In an ideal world we would gather more data for both the classification and forecasting routines in order to improve accuracy of both models. For the case of improving the classification accuracy, the only way to increase the amount of data is to wait years to gather more rainfall and production data. As I mentioned previously, now that Café Casa Martinez will be following the optimal zoca schedule, within a few years they will reach a maximum equilibrium production state because all fields will be planted and will be renewed every 6 years. Retraining regression and/or classification models on data after this equilibrium is reached should eliminate the confounding variables associated with having lots that are not sown or producing optimally, which should increase the accuracy associated with the use of rainfall feature variables. Another avenue for gathering more data would be to start collecting other ancillary weather data, such as temperature, cloud cover, and humidity.

With regard to improving the weather forecasting, more data may increase accuracy, however the rainfall totals recorded by Café Casa Martinez represent the best available data for a few reasons. First, the nearest weather station to the farm is at a much lower altitude in a place with a drier climate, despite the fact that it is only about a 20-minute drive away. Scraping weather data from the web or purchasing access to a weather API would not be helpful in this case because the available weather stations do not record the true weather conditions at high elevation where the farm is. The farm has its own micro climate. At any rate, the rainfall data available online for the region surrounding Finca Tulipan does not extend as far back in time as the rainfall totals recorded by Café Casa Martinez. The data they presented me represents the best data in existence, barring the possibility that a neighboring farm rigorously recorded rainfall data for longer. For this reason, I do not believe it is possible to increase forecasting accuracy without waiting years for more data to become available.

# Zoca Schedule
We saw during the exploratory analysis that the most important thing Café Casa Martinez can do to maximize production is sow all their lots with plants and keep up with the renewal schedule every 6 years.  Surprisingly, these have a stronger effect on production than rainfall totals. Upon inspection of the lot renewal schedule, I found four lots that are currently overdue for renewal. While it would be possible to renew all four lots immediately, it would not be ideal ideal because that would lead to many plants not producing at once. It is better to keep the production as consistent as possible between years which requires an equal number of plants to be renewed every year, in the ideal case. With this in mind, I tried to divide the four overdue lots into bins with close-to-equal numbers of plants in order to make an optimal zoca schedule for the coming years. I wrote a script posted [here]( https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/zoca_schedule.py) that creates a pandas dataframe listing the years of suggested renewal for each field for the next 60 years. Below is a table showing the suggested lots to renew each year for the next 4 years.

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
