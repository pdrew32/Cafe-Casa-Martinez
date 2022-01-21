# Cafe Casa Martinez
[Cafe Casa Martinez](https://www.cafecasamartinez.com/) is a small, organic coffee company with a farm called Tulipan in the Santander department of Colombia. They grow three varieties of coffee, Castillo, Tabi, and Cenicafe 1. These varieties grow in the shade and only fruit once per year. The plants typically flower between February and April and produce fully ripened fruits between October and December.

These coffee plants produce fruit (from which beans are extracted) at a fraction of their max productivity that depends on the number of years since the tree has been planted or renewed. No fruit is produced in the year of sowing or the year after. In the second year after sowing the plant will produce fruit at 50% of its maximum production. The third year sees maximum fruit production and each year after that sees decreasing production. For this reason, in the 6th year after sowing the plant is cut near its base. The plant then regrows from the stump. This process is called renewing, or zoca, and it returns the plant to maximum production after two more years. Plants may be renewed up to 5 times. It is best practices to stagger the renewal or sowing of plants in different lots so that some of the farm is always producing.

# Available Data
Cafe Casa Martinez has daily rainfall totals reaching back to October 2006, yearly coffee production totals since 2008, and information about the month and day of sowing, renewal, and number of plants in each of the 11 lots. 

# Key Goals
The first set of key goals are to determine which quantities correlate with yearly production and what practices could increase or maximize production. Secondary goals include investigating what requirements there are to make a profit in a given year. 

# Results
The first result is that rainfall in the month of May is most highly correlated with production. This result is arrived at after controlling for the decreased production of lots that are not in their maximum year of production. The correlation has a strength of .

The maximum production of a lot depends on the soil, rainfall, climate, and available sunlight. It is not known in general without empirical measurement from the plants in a given lot. Tulipan farm has 11 lots in which coffee is grown on, each with a different number of plants. The available production data is not split out into production per lot. There is only a record of the total production in a year for the entire farm, so I needed to find a way to estimate the theoretical maximum production from the data, while trying to control for different levels of production due to staggered sowing and zoca schedules. This is complicated by the fact that rainfall totals are different every year and the number of plants of a given age vary, with production totals tied to both of these.
