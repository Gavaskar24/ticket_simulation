import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import expon
import matplotlib.pyplot as plt

# Passenger arrival times are to be modelled from inter-arrival time data from file.csv
df=pd.read_csv('file.csv')

# First column Ben is interarrival times of passengers towards A
# Second column Ben is interarrival times of passengers towards B

# Plotting histogram of inter-arrival times towards A
plt.hist(df['Ben'], bins=20)
plt.title("Inter-arrival times towards A")
plt.xlabel("Inter-arrival times")
plt.ylabel("Frequency")
plt.show()

#It looks like exponential distribution, lets try to fit exponential distribution to this data

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

# Create your time series data as a NumPy array or a pandas Series
# For example, assuming you have a variable 'inter_arrival_times' containing your data

# Calculate the Ljung-Box test statistics and p-values
lb_test_stat, lb_p_value = acorr_ljungbox(df['Ben'], lags=None)

# Display the results for each lag
for lag, test_stat, p_value in zip(range(1, len(lb_test_stat) + 1), lb_test_stat, lb_p_value):
    print(f"Lag {lag}: Ljung-Box Test Statistic = {test_stat}, P-Value = {p_value}")

# You can examine each test statistic and its corresponding p-value to assess autocorrelation at different lags.


# You can also check whether the p-values are below a significance level (e.g., 0.05) to reject the null hypothesis of no autocorrelation.


# sns.set(color_codes=True)
# sns.set(rc={'figure.figsize':(5,5)})
# x = np.linspace(0, 50, 100)
# data=df['Ben']
# ax = sns.distplot(data,
#                   kde=True,
#                   bins=100,
#                   color='skyblue',
#                   hist_kws={"linewidth": 15,'alpha':1}) 
# ax.set(xlabel='Exponential Distribution ', ylabel='Frequency')
# params = expon.fit(data)
# print("Parameters of exponential distribution",params)
# plt.show()
# plt.figure()

# # Plotting histogram of inter-arrival times towards B
# plt.hist(df['Tum'], bins=20)
# plt.title("Inter-arrival times towards B")
# plt.xlabel("Inter-arrival times")
# plt.ylabel("Frequency")
# plt.show()

# # As the data seem to follow Normal distribution fitting Normal distribution to the inter-arrival times towards B

# sns.set(color_codes=True)
# sns.set(rc={'figure.figsize':(5,5)})
# x = np.linspace(0, 50, 100)
# data=df['Tum']
# ax = sns.distplot(data,
#                   kde=True,
#                   bins=100,
#                   color='skyblue',
#                   hist_kws={"linewidth": 15,'alpha':1})
# ax.set(xlabel='Normal Distribution ', ylabel='Frequency')
# params = norm.fit(data)
# print("Parameters of Normal distribution",params)

# plt.show()

