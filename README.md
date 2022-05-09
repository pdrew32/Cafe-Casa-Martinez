# Café Casa Martinez
[Café Casa Martinez](https://www.cafecasamartinez.com/) is a small, organic coffee company with a farm called Finca Tulipan in the Santander department of Colombia. They grow three varieties of coffee, Castillo, Tabi, and Cenicafe 1. These varieties grow in the shade and only fruit once per year. The plants typically flower between February and April and produce fully ripened fruits between October and December, from which the coffee beans are extracted.

These coffee plants produce fruit at a fraction of their maximum productivity which depends on the number of years since the tree has been planted. No fruit is produced in the year of sowing or the following year. In the year after that, the plant will produce fruit at 50% of its maximum. The fourth year sees maximum fruit production and each year after that sees decreasing production. For this reason, in the 6th year after sowing, the farmers perform a process called renewal, or zoca, where the plant is cut near its base. The plant then regrows from the stump and returns to maximum production after two more years. Plants may be renewed up to 5 times. The following figure shows the production fraction as a function of year covering the sowing cycle and three zoca cycles.

<img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/production_fraction_vs_year.png" width="400">

Farms like Finca Tulipan have multiple fields of coffee that were sown in different years and are therefore, in any given year, they have fields in different stages of their renewal cycles. It is best practice to stagger renewal and sowing in  different fields in this way so that total production in each year stays as consistent as possible.

# Key Goals
The first goal was to determine which quantities correlate with yearly production and to consult with Café Casa Martinez about using this information to maximize production. Secondary goals include investigating what is required to make a profit each year.

# Summary of Results
1.	Sticking with the optimal plant renewal schedule is the single most important factor to maximize production.
2.	May rainfall totals correlate with production per plant and total production.
# Steps
Café Casa Martinez has daily rainfall records reaching back to October 2006, yearly coffee production totals since 2008, records of the month and day of sowing and renewal, and the number of plants in each of their 14 lots. 

First, I cleaned the data, formatted them as pandas dataframes, and saved them for future use in other scripts. The script that performs these steps can be found [here](https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/clean_data.py). Next, I performed exploratory data analysis to determine which features might correlate with interesting target variables. I calculated Pearson correlation coefficients and P-values between all parameters. The Pearson correlation coefficient is a measure of the linear correlation between features in the dataset. Pearson correlation coefficient values closer to 1 (or -1) are highly correlated and values closer to 0 are uncorrelated. The following table summarizes the strongest correlations between features and potential target variables of interest:

| Parameter 1      | Parameter 2 | Pearson Correlation Coefficient | P-value | Statistically Significant? |
| ---------------- | ----------- | ------------------------------- | ------ | --- |
|  Total Production  |  Year  |  0.88 |  2.2e-10 |  yes  |
|  Production Per Plant  |  Year  |  0.79  |  5.5e-8 |  yes  |
|  Total Number of Plants  |  Year  |  0.75  |  3.8e-8 |  yes  |
|  Total Number of Plants  |  Total Production  |  0.68  |  5.0e-7 |  yes  |
|  Production Per Plant  |  Total May Rainfall  |  0.66  |  1.4e-5 |  yes  |
|  Total Production  |  Total May Rainfall  |  0.64  |  3.6e-5 |  yes  |

The total production and production per plant are both highly correlated with year. At first glance this may be surprising, but these trends are seen because the coffee plants that were inherited with the purchase of the farm in 2006 had not been renewed recently and were therefore not producing optimally.  While it is not shown in the figure above, the downward trend in plant production continues if a field is not renewed, which is why one wants to renew in the first place. Between 2008 and 2021 the owners began to renew some old fields, which increased production in a way that correlates with year.

When the farm was purchased, there were also several lots that had no coffee plants yet. Over time, an increasing number of these empty lots were sown, thereby boosting total production in a manner that is correlates with year. This is verified by the fact that the third highest correlation in the data is between the total number of plants and year.

The fact that the 4th strongest correlation is between total number of plants and total production. This seems logical, as more plants equals more coffee, but it is interesting to note that this is not the strongest correlation in the data. In the coming years these correlations will disappear because the farm will have reached an optimal renewal schedule and they will run out of empty lots to plant. **These correlations illustrate our first key takeaway. Sticking with the optimal plant renewal schedule and filling all available lots with coffee plants are the two most important factors when trying to maximize production. Surprisingly, these are more significant than rainfall totals.**

Our next main result is that **rainfall in the month of May is most highly correlated with production.** This is surprising to the owners of Café Casa Martinez because it is known that rainfall in the month of February is what causes the plants to flower, so it was assumed that rainfall during that month is important. There is no correlation between February rainfall totals and production. The Pearson correlation coefficient is -0.2, which is much closer to 0 than to positive or negative 1. Additionally, the probability that this is a chance correlation is high, given that the P-value associated with the correlation is higher than 0.05 (actual value is 0.14). February rainfall may be the rain that causes the plant to flower, but the rainfall that contributes most strongly to the production of coffee beans occurs during the month of May. Rainfall totals do not have a significant correlation with production in any other month. The figure below on the left shows the correlation between total production and May rainfall totals and the figure on the right shows production per plant versus May rainfall.

<p float="left">
  <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/may_rain_vs_production.png" width="400">  
  <img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/may_rain_vs_production_per_plant.png" width="400">
</p>


The black lines show the best fit correlations between the parameters and the red line in the left figure shows the estimated production required to make a profit, according to Café Casa Martinez. May rainfall totals greater than 203.2 mm will result in a profit.

This leads us to our second recommendation to maximize production. **If rainfall in the month of May is going to be below 203.2 mm, the farm should consider watering their lots.** One never knows how much rain is going to fall in May ahead of time, so next I explore whether machine learning can accurately predict May rainfall totals, and thus yearly production.

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

