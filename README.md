# Cafe Casa Martinez
[Cafe Casa Martinez](https://www.cafecasamartinez.com/) is a small, organic coffee company with a farm called Tulipan in the Santander department of Colombia. They grow three varieties of coffee, Castillo, Tabi, and Cenicafe 1. These varieties grow in the shade and only fruit once per year. The plants typically flower between February and April and produce fully ripened fruits between October and December, from which the coffee beans are extracted.

These coffee plants produce fruit at a fraction of their maximum productivity that depends on the number of years since the tree has been planted. No fruit is produced in the year of sowing or the following year. In the year after that the plant will produce fruit at 50% of its maximum. The fourth year sees maximum fruit production and each year after that sees decreasing production. For this reason, in the 6th year after sowing the farmers perform a process called renewal, or zoca, where the plant is cut near its base. The plant then regrows from the stump and returns to maximum production after two more years. Plants may be renewed up to 5 times. The following figure shows the production curve as a function of year covering the sowing cycle and three zoca cycles.

<img src="https://github.com/pdrew32/Cafe-Casa-Martinez/blob/master/figures/production_fraction_vs_year.png" width="500">

It is best practice to stagger the renewal or sowing of plants across different lots so that a high fraction of the farm is always producing.

# Available Data
Cafe Casa Martinez has daily rainfall records reaching back to October 2006, yearly coffee production totals since 2008, and information about the month and day of sowing, renewal, and number of plants in each of the 11 lots. 

# Key Goals
The first set of key goals are to determine which quantities correlate with yearly production and what practices could increase or maximize production. Secondary goals include investigating what requirements there are to make a profit each year. 

# Results
The strongest correlation in the dataset is between total production and year with a Pearson correlation coefficient of 0.88. This can be explained by two effects, the first being that the total number of plants is also highly correlated with year. The production per plant is also highly correlated with year because of the renewal processes in different lots returning the plants to high production. When the farm was purchased in 2006 the existing coffee plants were not producing well because they had not been refreshed recently. Also, there were several lots that had no coffee plants yet. As year increased from 2008 to 2021 the fraction of new and refreshed plants to the total number of plants increased. This has the strongest effect on the production of coffee. This leads us to the first main result. To maximize production, Cafe Casa Martinez needs to keep up with their renewal and planting schedules. New and recently refreshed plants have a stronger effect on production than rainfall levels.

Rainfall in the month of May is most highly correlated with production. This result is arrived at after controlling for the decreased production of lots that are not in their maximum year of production. The correlation has a strength of .

The maximum production of a lot depends on the soil, rainfall, climate, and available sunlight. It is not known in general without empirical measurement from the plants in each lot. Tulipan farm has 11 lots in which coffee is grown on, each with a different number of plants. The available production data is not split out into production per lot. There is only a record of the total production in a year for the entire farm, so I needed to find a way to estimate the theoretical maximum production from the data, while trying to control for different levels of production due to staggered sowing and zoca schedules. This is complicated by the fact that rainfall totals are different every year and the number of plants of a given age vary, with production totals tied to both.

# Steps
First, I cleaned the data by translating the columns from Spanish to English and removing the excel formatting of each file, as Cafe Casa Martinez works with each file in excel. I saved the clean pandas data frames for use in other scripts. 

Next, I performed an exploratory data analysis. First, I calculated Pearson correlation coefficients for all parameters. This coefficient is a measure of the linear correlation between parameters with values closer to 1 (or -1) being highly correlated and values closer to 0 being uncorrelated. I found the following:

| Parameter 1      | Parameter 2 | Pearson Correlation Coefficient | P-value | Statistically Significant? |
| ---------------- | ----------- | ------------------------------- | ------ | --- |
|  Total Production  |  Year  |  0.88 |  8.8e-5 |  yes  |
|  Production Per Plant  |  Year  |  0.79  |  1.3e-5 |  yes  |
|  Total Number of Plants  |  Year  |  0.75  |  3.1e-3 |  yes  |
|  Total Number of Plants  |  Total Production  |  0.68  |  1.1e-2 |  yes  |
|  Production Per Plant  |  Total May Rainfall  |  0.66  |  1.4e-2 |  yes  |
|  Total Production  |  Total May Rainfall  |  0.64  |  1.8e-2 |  yes  |
 
: a footnote

Here I do not include self-correlations such as the correlation between total production per plant and total production nor am I including the correlations between rainfall totals in different months because these correlations are not interesting for our purposes. I manually reviewed all the other correlations to make sure there was nothing else interesting to look at.
The total production and production per plant are highly correlated with year because Cafe Casa Martinez has been renewing their fields and have not yet reached the stable equilibrium that will be achieved after all fields are renewed on schedule. The total number of plants producing each year is also correlated with year, which further contributes to the total production. In the coming years this correlation will drop away because they will be on the optimal renewal schedule. This is the first key takeaway. It is more important to stick with the renewal schedule than anything else, even to have adequate rainfall.

