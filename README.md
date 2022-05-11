# Café Casa Martinez
[Café Casa Martinez](https://www.cafecasamartinez.com/) is a small, organic coffee company with a farm called Finca Tulipan in the Santander department of Colombia. They grow three varieties of coffee, Castillo, Tabi, and Cenicafe 1. These varieties grow in the shade of trees and only fruit once per year. The plants typically flower between February and April and produce fully ripened fruit between October and December, from which the coffee beans are extracted.

Coffee plants do not produce the same quantity of beans every year, even without rainfall variability. The fruit production of any given plant depends on the number of years since that tree was planted. The production curve is as follows. In the year of sowing and the following year, no fruit is produced. In the year after that, the plant will produce fruit at 50% of its maximum. The fourth year sees maximum fruit production and each year after that sees decreasing production. For this reason, in the 6th year after sowing, the farmers perform a process called renewal, or zoca, where the plant is cut near its base. The plant then regrows from the stump and returns to maximum production after two more years. Plants may be renewed up to 5 times. The following figure shows the production fraction as a function of year covering the sowing cycle and three zoca cycles.

<img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/production_fraction_vs_year.png" width="400">

Farms like Finca Tulipan have multiple fields of coffee that were sown and renewed in different years and therefore, in any given year, they have fields in all stages of their renewal cycles. It is best practice to stagger renewal and sowing in different fields so that total production in each year stays as consistent as possible.

Café Casa Martinez provided me with daily rainfall records reaching back to October 2006, yearly coffee production totals since 2008, records of the month and day of sowing and renewal, and the number of plants in each of their 14 lots. With this data I worked toward the following key goals:

# Key Goals
1. Optimize the renewal schedule to minimize inter-year variation in production.
2. Determine which quantities correlate with yearly production.
3. Train a machine learning algorithm to predict whether the company should expect to make a profit in any given year.

# Summary of Results
1.	Planting new fields and renewing fields on time are the two most important factors Café Casa Martinez can do to maximize production. These factors correlate more strongly with production than rainfall.
2.	May rainfall totals correlate with production per plant and total production. Rainfall in this month is the only one that correlates with production.
3.	I present a weather forecasting model to predict rainfall on the farm. This may be used to predict whether a profit should be expected at the end of the year.

# Steps
First, I cleaned the data, formatted them as pandas dataframes, and saved them for future use in other scripts. The script that performs these steps can be found [here](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/clean_data.py). Next, I performed exploratory data analysis to determine which features might correlate with interesting target variables, such as total production and production per plant. I calculated Pearson correlation coefficients and P-values between all parameters. The Pearson correlation coefficient is a measure of the linear correlation between features in the dataset. Pearson correlation coefficient values closer to 1 (or -1) are highly correlated and values closer to 0 are uncorrelated. The following table summarizes the strongest correlations between features and target variables of interest:

| Parameter 1      | Parameter 2 | Pearson Correlation Coefficient | P-value | Statistically Significant? |
| ---------------- | ----------- | ------------------------------- | ------ | --- |
|  Total Production  |  Year  |  0.88 |  2.2e-10 |  yes  |
|  Production Per Plant  |  Year  |  0.79  |  5.5e-8 |  yes  |
|  Total Number of Plants  |  Year  |  0.75  |  3.8e-8 |  yes  |
|  Total Number of Plants  |  Total Production  |  0.68  |  5.0e-7 |  yes  |
|  Production Per Plant  |  Total May Rainfall  |  0.66  |  1.4e-5 |  yes  |
|  Total Production  |  Total May Rainfall  |  0.64  |  3.6e-5 |  yes  |

The total production and production per plant are both highly correlated with year. At first glance this may seem surprising, but these correlations are seen because the coffee plants that were inherited with the purchase of the farm in 2006 had not been renewed recently, and were therefore not producing optimally.  While it is not shown in the figure above, the downward trend in plant production continues if a field is not renewed, which is why one wants to renew in the first place. Between 2008 and 2021 the owners began to renew some old fields, which increased production in a way that correlates with year.

When the farm was purchased, there were also several lots that had no coffee plants yet. Over time, an increasing number of these empty lots were sown, which boosted total production in a manner that correlates with year. This is verified by the fact that the third highest correlation in the data is between the total number of plants and year. In the coming years these correlations will disappear because the farm will have reached an optimal renewal schedule and they will run out of empty lots to plant. **These three strongest correlations illustrate our first key takeaway. Sticking with the optimal plant renewal schedule and filling all available lots with coffee plants are the two most important factors when trying to maximize production. Surprisingly, these have a stronger effect on production than rainfall totals.**

Our next main result is that **rainfall in the month of May is most highly correlated with production.** This is surprising to the owners of Café Casa Martinez because it was previously known that rainfall in the month of February causes the plants to flower. It was therefore assumed that rainfall during that month would correlate with production. There is, however, no correlation between February rainfall totals and production. The Pearson correlation coefficient between February rainfall and production is -0.2, which is much closer to 0 than to positive or negative 1. The probability that that is a chance correlation is high, given that the associated P-value associated is higher than 0.05 (actual value is 0.14). February’s rainfall may cause the plants to flower, but it’s rainfall during the month of May that contributes most strongly to the production of coffee beans. I find no statistically significant correlations between any other month and total production or production per plant. The figure below on the left shows the correlation between total production and May rainfall and the figure on the right shows production per plant versus May rainfall.

<p float="left">
  <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/may_rain_vs_production.png" width="400">  
  <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/may_rain_vs_production_per_plant.png" width="400">
</p>


The blue points represent individual years of May rainfall totals versus total production for the year (left figure) and total production per plant (right figure).  We see that the higher the rainfall in May, the higher the total production and the production per plant at the end of the year.  The black lines show the best fit correlations between total production or production per plant and may rainfall totals derived using ordinary least squares linear regression. The red line in the left figure shows the estimated production required to make a profit, according to Café Casa Martinez. May rainfall totals greater than 203.2 mm are expected to result in a profit. With this information, Café Casa Martinez can predict their production after observing the rainfall totals in the month of May. However, the data is scattered considerably about the best fit.  The square root of the mean square error is 1450 kg, which is about 50% of the median production of all years.  The R<sup>2</sup> score for the regression fit between May rainfall and total production is 0.41, meaning 41% of the variance seen in total production is explainable by the variance in May rainfall. We will see that we have better predictive power if we recast this as a classification problem where years are classified in terms of profitable versus non profitable and then classification algorithms are used.

The script that performs the classification analysis is hosted [here](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/classification_profit.py). First I add a new binary column to the pandas dataframe that is 1 if the production is greater than 3500 kg and 0 if less than. Then because I am working with numerical features and a categorical target I run an ANOVA and compute the f-statistic to find the features that correlate most strongly with the target variable. I find that rainfall during the month of may is significantly correlated with profitable years (p-value < 0.05), and august rainfall is significantly anti-correlated with profitable years. While this anti-correlation is not causal because all available evidence suggests that a lack of rainfall does not lead to increased production in plants, this correlation is significant and therefore I use this feature in my classification routine. During testing, I ran logistic regression on all features (monthly rainfall for all months) with total production as a target variable and found worse accuracy, precision, and recall than fits with just May and August rainfall totals.

Next I ran numerous classification algorithms to determine which provides the best accuracy, including logistic regression, Gaussian naïve Bayes, k nearest neighbors, support vector machines, decision tree, and random forest. I run each of these after standardizing the features using the standard scaler from sklearn. I achieve the highest accuracy with logistic regression, as determined by stratified k-fold cross validation with 4 folds. I used stratified k-fold because there is a 70/30% split in the target variable, where 70% of years were unprofitable.  Using regular k-fold cross validation performs worse. The table below shows the accuracy of each of these methods.

| Algorithm      | Mean Stratified K-fold Cross Validation Accuracy | 
| ---------------- | ----------- |
|  Logistic Regression  |  83%  |
| Support Vector Machine |  71%  |
| K Nearest Neighbors   | 69%   |
| Decision Tree  |  69%  | 
| Random Forest |  69%   |
| Gaussian Naïve Bayes |   63%      |

After determining that logistic regression is the best algorithm in this case, I trained using all data and found the following confusion matrix:

Logistic Regression Accuracy = 92%
| Target | Precision | Recall | F1-score | 
| ---- | ---- | ---- | ---- | 
| 0 | 0.89 | 1.0 | 0.94 |  
| 1 | 1.0 | 0.8 | 0.89 |  

These values suggest logistic regression does acceptably well. A grid search of optimal hyperparameters reveals the default values are best, C=1, penalty=l2.

First, I investigate the accuracy of this approach assuming we could perfectly predict May rainfall totals using machine learning. This is obviously not achievable, but it is important to investigate given the scatter in the relation between May rainfall totals and total production, as this will set the upper limit on our accuracy. The first constraint we want to impose, given the intrinsic scatter in the relation between May rainfall and total production, is to predict whether a year will be profitable rather than to predict the production in kg. In the figure below I adopt the value of the best fit relation between May rainfall totals and total production for each year and ask whether the true value and the predicted value are above or below the profit threshold.

<img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/may_rain_vs_production_perfect_info_with_year.png" width="500">

Here the squares show the true production and the circles show the production that would be predicted if we could perfectly predict the true May rainfall totals. The color of the square denotes the year according to the color bar on the right. Grey lines connect data where the true and predicted production are both below or both above the profitability threshold, which is shown as the black line. Red lines connect years where one would predict profitability when there would be a net loss or vice versa. I find that 80% of the time (8 out of 15 years) we accurately predict whether a year would be profitable, so this sets our expected upper limit of accuracy before we proceed with machine learning to predict May rainfall totals based on prior rain data.

Additionally, we have daily rainfall totals from November 2006 until January 2022. This time series data can be decomposed into seasonal and trend components which may provide useful insights. The first figure below shows the observed rainfall totals for each month, the underlying trend, the seasonal trend, and the residuals of the decomposition. The bottom left figure shows a zoom in on the trend curve, and the right figure shows the seasonal trend repeating for a single year so more detail may be seen.

<p float="left">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/seasonal_decomposition_plot.png" width="800">
</p>

<p float="left">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/seasonal_trend_plot.png" width="400">
    <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/seasonal_curve_plot.png" width="400">
</p>

The median of the residuals of the decomposition is -4 with a standard deviation of 51. This is consistent with 0, implying the decomposition is a good fit to the data. The seasonal trend component closely matches the median rainfall in each month across all years, which confirms the validity of this decomposition. Rainfall is lowest in January and highest in May. [Wang et al. 2006]( https://www.researchgate.net/publication/220451959_Characteristic-Based_Clustering_for_Time_Series_Data) describe metrics for evaluating the strength of the seasonal component and trend component strengths. Each range from 0 to 1, where 1 implies a strong component strength. The trend strength is weak at 0.13 and the seasonal strength is moderately strong at 0.54, implying our data has a seasonality trend with no additional underlying trend.


