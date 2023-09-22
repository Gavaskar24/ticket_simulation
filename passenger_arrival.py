import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import expon
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
from scipy.stats import *
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

sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(5,5)})
x = np.linspace(0, 50, 100)
data=df['Ben']
ax = sns.distplot(data,
                  kde=True,
                  bins=100,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1}) 
ax.set(xlabel='Exponential Distribution ', ylabel='Frequency')
params = expon.fit(data)
print("Parameters of exponential distribution",params)
plt.show()
plt.figure()

# # Plotting histogram of inter-arrival times towards B
# plt.hist(df['Tum'], bins=20)
# plt.title("Inter-arrival times towards B")
# plt.xlabel("Inter-arrival times")
# plt.ylabel("Frequency")
# plt.show()

# As the data seem to follow Normal distribution fitting Normal distribution to the inter-arrival times towards B

sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(5,5)})
x = np.linspace(0, 50, 100)
data=df['Tum']
ax = sns.distplot(data,
                  kde=True,
                  bins=100,
                  color='skyblue',
                  hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Normal Distribution ', ylabel='Frequency')
params = norm.fit(data)
print("Parameters of Normal distribution",params)

plt.show()

