# Café Casa Martinez
[Café Casa Martinez](https://www.cafecasamartinez.com/) is a small, organic coffee company with a farm called Finca Tulipan in the Santander department of Colombia. They grow three varieties of coffee, Castillo, Tabi, and Cenicafe 1. These varieties grow in the shade of trees and only fruit once per year. The plants typically flower between February and April and produce fully ripened fruit between October and December, from which the coffee beans are extracted.

Coffee plants do not produce the same quantity of beans every year, even without rainfall variability. The fruit production of any given plant depends on the number of years since that tree was planted. The production curve is as follows. In the year of sowing and the following year, no fruit is produced. In the year after that, the plant will produce fruit at 50% of its maximum. The fourth year sees maximum fruit production and each year after that sees decreasing production. For this reason, in the 6th year after sowing, the farmers perform a process called renewal, or zoca, where the plant is cut near its base. The plant then regrows from the stump and returns to maximum production after two more years. Plants may be renewed up to 5 times. The following figure shows the production fraction as a function of year covering the sowing cycle and three zoca cycles.

<img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/production_fraction_vs_year.png" width="400">

Farms like Finca Tulipan have multiple fields of coffee that were sown and renewed in different years and therefore, in any given year, they have fields in all stages of their renewal cycles. It is best practice to stagger renewal and sowing in different fields so that total production in each year stays as consistent as possible.

Café Casa Martinez provided me with daily rainfall records reaching back to October 2006, yearly coffee production totals since 2008, records of the month and day of sowing and renewal, and the number of plants in each of their 14 lots. With this data I worked toward the following key goals:

# Key Goals
1. Determine which quantities correlate with yearly production.
2. Train a machine learning algorithm to predict whether the company should expect to make a profit in any given year.
3. Optimize the renewal schedule to minimize inter-year variation in production.


# Summary of Results
1.	Planting new fields and renewing fields on time are the two most important factors Café Casa Martinez can do to maximize production. These factors correlate more strongly with production than rainfall.
2.	May rainfall totals correlate with production per plant and total production. Rainfall in this month is the only one that correlates significantly with production.
3.	I present a weather forecasting model to predict rainfall on the farm. This may be used to predict whether a profit should be expected at the end of the year.

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

Our next main result is that **May is the month where rainfall correlates most strongly with production.** This is surprising to the owners of Café Casa Martinez because it was previously known that rainfall in the month of February causes the plants to flower. It was therefore assumed that rainfall during that month would correlate with production. There is, however, no correlation between February rainfall totals and production. The Pearson correlation coefficient between February rainfall and production is -0.2, which is much closer to 0 than to positive or negative 1. The probability that that is a chance correlation is high, given that the associated P-value associated is higher than 0.05 (actual value is 0.14). February’s rainfall may cause the plants to flower, but it’s rainfall during the month of May that contributes most strongly to the production of coffee beans. I find no statistically significant correlations between any other month and total production or production per plant. The figure below on the left shows the correlation between total production and May rainfall and the figure on the right shows production per plant versus May rainfall.

<p float="left">
  <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/may_rain_vs_production.png" width="400">  
  <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/may_rain_vs_production_per_plant.png" width="400">
</p>

The blue points in each figure represent individual years of May rainfall totals versus total production for the year (left figure) and total production per plant (right figure).  We see that the higher the rainfall in May, the higher the total production and the production per plant for the year.  The black lines show the best fit correlations between total production or production per plant and May rainfall totals as derived using ordinary least squares linear regression. The red line in the left figure shows the estimated production required to make a profit, according to Café Casa Martinez (3500 kg). May rainfall totals greater than 203.2 mm are expected to result in a profit. 

With this measured correlation, Café Casa Martinez can predict their total production after observing the rainfall totals in the month of May. This is helpful to know because total production is not known until the end of the year. However, the data is scattered considerably about the best fit so the prediction may not be as accurate as one would like.  The square root of the mean square error is 1450 kg, which corresponds to about 50% of the median of production across all years. That is to say, a characteristic uncertainty associated with this model is about 50%.  The R<sup>2</sup> score for the regression fit between May rainfall and total production is 0.41, meaning only 41% of the variance seen in this correlation explainable by the variance in May rainfall.  There are a number of confounding factors that contribute in unknown ways toward production.  

The figure below shows the same figure as the one on the left above, however with residuals drawn as lines connecting observed production with estimated production. The color of the data points corresponds to the year. Circles are the predicted value and squares are the true values. 

<p float="left">
  <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/may_rain_vs_production_perfect_info_with_year.png" width="400">  
</p>

The three red lines denote years where the adoption of the best fit value would result in an incorrect classification of profitable or unprofitable, when compared with the truth. In other words, for these three data points the predicted value on the best fit relation is on the other side of the profit threshold from the true production value. Because a typical uncertainty in the regression is of order 50%, we may have better predictive power if we simplify the problem by recasting it in terms of a classification problem where years are classified in terms of profitable versus non profitable, rather than trying to predict the number of kilograms of roasted coffee produced.

The script that performs the classification analysis is hosted [here](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/classification_profit.py). First I create a new target variable that is 1 if the production is greater than 3500 kg and 0 if it is less than. Then because I am working with numerical features and a categorical target, I run an ANOVA and compute the f-statistic to find the features that correlate most strongly with the target variable. As I expected, I find that rainfall during the month of may is significantly correlated with profitable years (p-value < 0.05), but surprisingly I also find that August rainfall is significantly anti-correlated with profitable years. While this anti-correlation is not causal, because all available evidence suggests that a lack of rainfall does not lead to increased production in plants, this correlation is significant and therefore I use this feature in the classification routine. During testing, I ran logistic regression on all features (monthly rainfall for all months) with total production as a target variable and found worse accuracy, precision, and recall than fits with just May and August rainfall totals. I will provide more detail on this cross validation routine in the next paragraph.

Next I ran numerous classification algorithms to determine which provides the best accuracy, including logistic regression, Gaussian naïve Bayes, k nearest neighbors, support vector machines, decision tree, and random forest. I run each of these after standardizing the features using the standard scaler from sklearn. The highest accuracy is achieved with logistic regression, as determined by stratified k-fold cross validation with 4 folds. I used stratified k-fold because there is a 70/30% split in the target variable, where 70% of years were unprofitable.  Using regular k-fold cross validation results in worse performance across the board. The table below shows the accuracy of each of these methods.

| Algorithm      | Mean Stratified K-fold Cross Validation Accuracy | 
| ---------------- | ----------- |
|  Logistic Regression  |  83%  |
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

The top panel of the figure below shows the forecasted rainfall for 2020 in orange with true rainfall totals in blue as a function of month. The bottom panel shows the residuals, or reality minus forecast. We see the forecast is not perfect but does fairly well modeling the true rainfall totals. The forecast is quite accurate for May, but is less accurate in August. The forecast over predicts the August rainfall by about 30%. Taking the 
<p float="left">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/forecast_vs_reality.png" width="400">
</p>

# Zoca Schedule
We saw during the exploratory analysis that the most important thing Café Casa Martinez can do to maximize production is sow all their lots with plants and keep up with the renewal schedule every 6 years.  Surprisingly, these have a stronger effect on production than rainfall totals. Upon inspection of their renewal schedule I found four lots that are currently overdue for renewal. While it would be possible to renew all four lots immediately, it would not be ideal ideal because that would lead to many plants not producing at once. It is better to keep the production as consistent as possible between years which requires an equal number of plants to be renewed every year in the ideal case. With this in mind, I tried to divide the four overdue lots into bins with close to equal numbers of plants in order to make an optimal zoca schedule for Café Casa Martinez for the coming years. I wrote a script posted [here]( https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/zoca_schedule.py) that creates a pandas dataframe listing the years of suggested renewal for each field for the next 60 years. Below is a table showing the lots to renew each year for the next 4 years.

| 2022 |  |  2023 |  | 2024 | | 2025 |  | 
| --- | --- | --- | --- | ---- |  ---- | ---- |  --- |
| Lot | # plants | Lot | # plants | Lot | # plants | Lot | # plants |
| Sancocho 2 | 5000 | Ceiba 4 | 243 |  Ceiba 3|  900 | Ceiba 1 | 925 | 
|                       |            | Sancocho 1 | 1488 | Anacos 2 | 680 | Cedral | 2025 |
|                       |            |  | | La Hoya | 1130 | Anacos 1 | 2487 |
|                       |            |  | | Arrayanes | 906 |  |  |
|  |
| Total | 5000 |  | 1731 |  | 3616 |  | 5437 |

We can see that the number of plants renewed each year are not consistent, as would be ideal. The reason for this is that it is better to keep up with the current renewal schedule for fields that were already on schedule. Café Casa Martinez would lose production unnecessarily if they failed to keep up with the renewal schedule that was previously determined, in the case of fields that are not currently overdue for renewal. The current schedule for renewal for the 10 fields that are not late is not perfect, in that the number of plants renewed each year is not very close to uniform, but it is good enough and not worth the loss of production to correct.

