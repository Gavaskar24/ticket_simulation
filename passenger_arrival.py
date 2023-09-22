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

# check for sample independence
# plotting autocorrelation 
pd.plotting.autocorrelation_plot(df['Ben'])
plt.title("Autocorrelation plot of inter-arrival times towards A")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()

# Define all available distributions as candidate distributions
candidate_distributions = [beta, norm, expon, exponweib, exponpow, gamma, lognorm, pearson3, weibull_max, weibull_min, pareto, genextreme, gumbel_r, gumbel_l
                           , genlogistic, gennorm, halfgennorm, halflogistic, invgauss, invweibull, johnsonsb, johnsonsu, laplace, levy, levy_l, levy_stable, 
                           norminvgauss, powerlognorm, truncexpon, truncnorm, tukeylambda, vonmises, vonmises_line, wald, wrapcauchy]

# Fit each distribution and perform a KS test against the empirical data

best_fit = None
best_p_value = 0

for distribution in candidate_distributions:
    params = distribution.fit(df['Ben'])
    _, p_value = st.kstest(df['Ben'], distribution.cdf, args=params)

    if p_value > best_p_value:
        best_fit = distribution
        best_p_value = p_value



# visualize the best distribution
plt.hist(df['Ben'], bins=20, density=True, label='Data')
x=np.linspace(min(df['Ben']), max(df['Ben']), 100)
plt.plot(x, best_fit.pdf(x, *params), color='red', label='Best fit')
plt.legend()
plt.title("Inter-arrival times towards A")
plt.xlabel("Inter-arrival times")
plt.ylabel("Frequency")
plt.show()

# printing the best fit distribution
print("Best fit distribution: ", best_fit.name)
print("Best fit parameters: ", params)

# KS-test value
print("KS test value: ", best_p_value)






# Plotting histogram of inter-arrival times towards B
plt.hist(df['Tum'], bins=20)
plt.title("Inter-arrival times towards B")
plt.xlabel("Inter-arrival times")
plt.ylabel("Frequency")
plt.show()

# check for sample independence
# plotting autocorrelation 
pd.plotting.autocorrelation_plot(df['Tum'])
plt.title("Autocorrelation plot of inter-arrival times towards A")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()


# Fitting a distribution for inter-arrival times towards B

best_fit = None
best_p_value = 0

for distribution in candidate_distributions:
    params = distribution.fit(df['Tum'])
    _, p_value = st.kstest(df['Tum'], distribution.cdf, args=params)

    if p_value > best_p_value:
        best_fit = distribution
        best_p_value = p_value

Data=df['Tum']

# visualize the best distribution
plt.hist(df['Tum'], bins=20, density=True, label='Data')
x=np.linspace(min(df['Tum']), max(df['Tum']), 100)
plt.plot(x, best_fit.pdf(x, *params), color='red', label='Best fit')
plt.legend()
plt.title("Inter-arrival times towards B")
plt.xlabel("Inter-arrival times")
plt.ylabel("Frequency")
plt.show()

# printing the best fit distribution
print("Best fit distribution: ", best_fit.name)
print("Best fit parameters: ", params)

# KS-test value
print("KS test value: ", best_p_value)


