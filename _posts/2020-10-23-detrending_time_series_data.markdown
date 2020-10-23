---
layout: post
title:      "Detrending Time Series Data"
date:       2020-10-23 10:29:58 +0000
permalink:  detrending_time_series_data
---


Before you can begin modeling on time series data, you need to make sure that the data you have is stationary. Stationarity is important because if the mean or variance of a time series fluctuates throughout time, a model describing the data will vary in accuracy at different time points due to the changing mean/variance. Since most time series data is not stationary, this is an important step when dealing with time series data.

Let's go over a couple methods to make time series data stationary. I'll be using housing value data from Zillow and will be showing housing data from one zipcode in Los Angeles starting from 2010. Here's a quick glance at our data:

![](https://i.imgur.com/6EcjByW.jpg)

Taking a quick glance at our data, it is clear that there is an upward linear trend in our data. The trend might also be exponential due to how quickly the value of housing is rising.

### Methods to Check Stationarity

Before we begin detrending our data, we need a way to determine whether or not a series is stationary. We'll be using the Dickey-Fuller test which is a statistical test that tests for stationarity. The null hypothesis for the Dickey-Fuller test is that the series is not stationary. So if the test statistic is less than the critical value, we reject the null hypothesis and conclude that our time series is stationary. 

Here is the code to run the Dickey-Fuller test in python.
```
from statsmodels.tsa.stattools import adfuller

dftest = adfuller(ts)

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)
```

Running the Dickey-Fuller test for our data above, these are our results:

```
Results of Dickey-Fuller Test: 

Test Statistic                 -0.667612
p-value                         0.854961
#Lags Used                      9.000000
Number of Observations Used    90.000000
Critical Value (1%)            -3.505190
Critical Value (5%)            -2.894232
Critical Value (10%)           -2.584210
dtype: float64
```

With a p-value of 0.85, we fail to reject the null hypothesis and conclude that our data is not stationary. Now lets try to make our time series stationary!

### 1. Log Transformation

One method to make your series stationary is to log transform your data to make your series more uniform over time. Transformations are used to stabilize the variance of a series. Let's take the log transformation of our series:
```
log_ts = np.log(ts)
log_ts.plot(figsize=(8, 5));
```
![](https://i.imgur.com/dLhelUB.jpg)

Just by taking a look at the graph, we can see that our data has not become stationary by taking the log transformation. For some time series, log transformation is the way to go, but it is not the case for this time series so lets move on to another method.

### 2. Subtracting the Rolling Mean

Another method to make your time series stationary is to subtract the rolling mean from your data. The rolling mean is the moving average,  at any point *t* you can take the average of the *m* last time periods. *m* is known as the window size.
```
roll_mean = ts.rolling(window=4).mean()
fig = plt.figure(figsize=(11, 7))
plt.plot(ts, color='blue', label='Original')
plt.plot(roll_mean, color='red', label='Rolling Mean')
plt.legend()
plt.show();
```
![](https://i.imgur.com/mwjFXrn.jpg)

Here's how to subtract the rolling mean from you data and plot it:
```
data_minus_roll_mean = ts - roll_mean
data_minus_roll_mean.dropna(inplace=True)
data_minus_roll_mean.plot(figsize=(8, 5))
```
![](https://i.imgur.com/g8IJFHM.jpg)

Our data certainly looks more stationary by looking at the graph. Let's see if it passes the Dickey-Fuller test.
```
Results of Dickey-Fuller Test: 

Test Statistic                 -2.934873
p-value                         0.041440
#Lags Used                     12.000000
Number of Observations Used    84.000000
Critical Value (1%)            -3.510712
Critical Value (5%)            -2.896616
Critical Value (10%)           -2.585482
dtype: float64
```
With a p-value less than 0.05 we reject the null hypothesis and conclude that our time series is stationary! Let's take a look at another method to enforce stationarity.

### 3. Differencing

The last method and one of the most common methods to make your time series stationary is differencing. Differencing can help stabilize the mean of a time series by removing changes in the level of a time series to eliminate or reduce trends and seasonality. 
```
diff_ts = ts.diff(periods=5)
diff_ts.dropna(inplace=True)

fig = plt.figure(figsize=(8,5))
plt.plot(diff_ts, color='blue')
plt.title('Differenced series')
plt.show()
```
![](https://i.imgur.com/IHKGY0Z.jpg)

Differencing subtracts the previous observation from the current observation. The periods parameter is the lag at which to difference, a 1 period lag takes the difference between consecutive observations. A 12 period lag with monthly data will take the difference between yearly data.

```
Results of Dickey-Fuller Test: 

Test Statistic                 -3.297344
p-value                         0.014998
#Lags Used                     11.000000
Number of Observations Used    83.000000
Critical Value (1%)            -3.511712
Critical Value (5%)            -2.897048
Critical Value (10%)           -2.585713
dtype: float64
```
With a p-value of 0.01, we can reject the null hypothesis and conclude that our differenced series is stationary!

## Conclusion

Now that we our data is stationary it is ready to be used for time series modeling. I hope this was helpful in finding ways to make your time series data stationary.

