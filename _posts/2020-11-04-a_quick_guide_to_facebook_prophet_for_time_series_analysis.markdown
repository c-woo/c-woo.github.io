---
layout: post
title:      "A Quick Guide to Facebook Prophet for Time Series Analysis"
date:       2020-11-04 07:38:05 -0500
permalink:  a_quick_guide_to_facebook_prophet_for_time_series_analysis
---


Facebook prophet is an open source forecasting tool available for Python and R. Prophet makes it easier for experts and non-experts to make high quality forecasts at scale by simplifying the forecasting process and providing an improved predicting ability. Prophet uses an additive regression model which represents a time series with an overall trend as well as a combination of patterns at different scales, such as daily, weekly, or monthly. 

Before you can use Prophet you'll need to install it if you haven't already. 
To install from terminal:
```
pip install pystan
pip install fbprophet
```
To install from jupyter notebook:
```
conda install pystan -c conda-forge
conda install -c conda-forge fbprophet
```

Now that we've installed Prophet let's try using it on a dataset. The dataset I'll be using is this [Air Passengers](https://www.kaggle.com/chirag19/air-passengers) dataset which provides monthly totals of a US airline from 1949 to 1960. 

Let's take a quick look at this dataset and see what it looks like.
```
df = pd.read_csv('AirPassengers.csv')
df.head()
```
![](https://i.imgur.com/b5XxRyc.jpg)

The input to Prophet is always a dataframe with two columns `ds`(the time column) and `y`(the metric column). Let's rename the columns of our dataframe so we can use it with Prophet.
```
df.rename(columns={'Month': 'ds', '#Passengers': 'y'}, inplace=True)
df.head()
```
![](https://i.imgur.com/oXTmIoe.jpg)

Now that our columns have been properly renamed, lets take a look at our data.
```
df.set_index('ds').plot(figsize=(13, 6))
```
![](https://i.imgur.com/pQwQUGU.jpg)

Taking a look at this graph, we can see that there is an upward trend and seasonality in our data. Now let's begin running our data through Prophet.
```
from fbprophet import Prophet

Model =Prophet()
Model.fit(df)
```

In order to make predictions with Prophet, we need to make a new dataframe with a column `ds` containing the dates for which a prediction is to be made. Prophet provides an easy way to make this dataframe so we don't have to do it manually.
```
future = Model.make_future_dataframe(periods=36, freq='MS')
future.tail()
```
![](https://i.imgur.com/0p492DZ.jpg)

Since our data is monthly data we'll specify the desired frequency of the timestamps with the `freq` parameter. Now that we have our dataframe, we can use the `.predict()` method to make predictions.
```
forecast = Model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```
![](https://i.imgur.com/gq6821R.jpg)

Prophet returns a table that includes these columns:
`ds`: datestamp of the forecasted value
`yhat`: the forecasted value
`yhat_lower`: the lower bound of our forecast
`yhat_upper`: the upper bound of our forecast

We can also plot the results of the forecast using Prophet.
```
Model.plot(forecast)
plt.show()
```
![](https://i.imgur.com/178cZ0H.jpg)

Prophet plots the observed values of our data (the black dots), the forecasted values (the blue line), and the uncertainty intervals (the shaded region). Prophet also can plot the components of our forecast to show trend and daily, weekly, and yearly patterns of our data.
```
Model.plot_components(forecast)
plt.show()
```
![](https://i.imgur.com/51jJCh0.jpg)

We can see the upward trend in our data as well as the yearly seasonality. Since our data is monthly we see the yearly seasonality, but if you were working with daily data, you would also see a weekly seasonality plot. 

Using Prophet, we were able to quickly and easily model and forecast our time series data and hopefully you'll be able to use Prophet to model and forecast your time series data as well.
