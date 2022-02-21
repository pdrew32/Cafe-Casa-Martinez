# Café Casa Martinez
[Café Casa Martinez](https://www.cafecasamartinez.com/) is a small, organic coffee company with a farm called Finca Tulipan in the Santander department of Colombia. They grow three varieties of coffee, Castillo, Tabi, and Cenicafe 1. These varieties grow in the shade and only fruit once per year. The plants typically flower between February and April and produce fully ripened fruits between October and December, from which the coffee beans are extracted.

These coffee plants produce fruit at a fraction of their maximum productivity that depends on the number of years since the tree has been planted. No fruit is produced in the year of sowing or the following year. In the year after that the plant will produce fruit at 50% of its maximum. The fourth year sees maximum fruit production and each year after that sees decreasing production. For this reason, in the 6th year after sowing the farmers perform a process called renewal, or zoca, where the plant is cut near its base. The plant then regrows from the stump and returns to maximum production after two more years. Plants may be renewed up to 5 times. The following figure shows the production curve as a function of year covering the sowing cycle and three zoca cycles.

<img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/production_fraction_vs_year.png" width="500">

Farms like Finca Tulipan have multiple fields of coffee that were sown in different years and are therefore in different years of their renewal cycles. It is best practice to stagger renewal and sowing across different fields so that total production stays as close as uniform as possible across years.

# Available Data
Café Casa Martinez has daily rainfall records reaching back to October 2006, yearly coffee production totals since 2008, records of the month and day of sowing and renewal, and the number of plants in each of their 14 lots. 

# Key Goals
The first goal was to determine which quantities correlate with yearly production and to consult with Café Casa Martinez about  practices could increase or maximize production. Secondary goals include investigating what requirements there are to make a profit each year. 

# Summary of Results
1.	Sticking with the optimal plant renewal schedule is the single most important factor to maximize production.
# Steps
First, I cleaned the data by translating the columns from Spanish to English and removing the excel formatting of each file, as Café Casa Martinez works with each file in excel. I saved the clean pandas data frames for use in other scripts. 

Next, I performed an exploratory data analysis. First, I normalized the data using the robust scaler from sklearn and then calculated Pearson correlation coefficients and P-values for all parameters. This coefficient is a measure of the linear correlation between parameters with values closer to 1 (or -1) being highly correlated and values closer to 0 being uncorrelated. I found the following:

| Parameter 1      | Parameter 2 | Pearson Correlation Coefficient | P-value | Statistically Significant? |
| ---------------- | ----------- | ------------------------------- | ------ | --- |
|  Total Production  |  Year  |  0.88 |  2.2e-10 |  yes  |
|  Production Per Plant  |  Year  |  0.79  |  5.5e-8 |  yes  |
|  Total Number of Plants  |  Year  |  0.75  |  3.8e-8 |  yes  |
|  Total Number of Plants  |  Total Production  |  0.68  |  5.0e-7 |  yes  |
|  Production Per Plant  |  Total May Rainfall  |  0.66  |  1.4e-5 |  yes  |
|  Total Production  |  Total May Rainfall  |  0.64  |  3.6e-5 |  yes  |

The total production and production per plant are both highly correlated with year. At first glance this may be surprising, but these trends are seen because the coffee plants that were inherited with the purchase of the farm in 2006 had not been renewed recently and were therefore not producing optimally.  While it is not shown in the figure above, the downward trend in plant production continues if a field is not renewed, hence the point of renewal. Between 2008 and 2021 the owners began to renew some old fields, which increased production in a way that correlates with year.

When the farm was purchased, there were also several lots that had no coffee plants yet. Over time, an increasing number of these empty lots were sown, thereby boosting total production in a manner that is correlates with year. This is verified by the fact that the third highest correlation in the data is between the total number of plants and year.

The fact that the 4th strongest correlation is between total number of plants and total production. This seems logical, as more plants equals more coffee, but it is interesting to note that this is not the strongest correlation in the data. In the coming years these correlations will disappear because the farm will have reached an optimal renewal schedule and they will run out of empty lots to plant. **These correlations illustrate our first key takeaway. Sticking with the optimal plant renewal schedule and filling all available lots with coffee plants are the two most important factors when trying to maximize production. Surprisingly, these are more significant than rainfall totals.**

Our next main result is that **rainfall in the month of May is most highly correlated with production.** 
