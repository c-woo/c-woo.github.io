---
layout: post
title:      "Bar Plots for Analysis"
date:       2019-11-02 15:01:47 -0400
permalink:  bar_plots_for_analysis
---


For our first project I had my first attempt at data analysis. The data I used was the Kings County House Sales dataset which includes the data for houses that were sold in Kings County, Washington in 2014 and 2015. This dataset contains many different variables along with the prices each house was sold for and during my analysis I came across a problem trying to figure out the best way to see the relationship between bedrooms/bathrooms and price. 

My first thought was to use a scatter plot graphing both variables to see if there was any insight I could gain.

![Scatter plot](https://i.imgur.com/mxfZssq.png)

While you can see that there is a positive linear relationship between the number of bedrooms/bathroom and price, theres not much else you can infer from the graph. 

This is where bar plots come in and seaborn is a great simple tool that you can use to create multiple different bar plots to suit your needs. We'll begin by creating the same bar plots using this line of code

```
plt.figure(figsize=(22, 8))

ax1 = plt.subplot(1, 2, 1)
sns.barplot(x='bedrooms', y='price', data=df);
ax1.set_title('Price of Homes Based on Number of Bedrooms');
plt.xlabel('Bedrooms');
plt.ylabel('Price');

ax2 = plt.subplot(1, 2, 2)
sns.barplot(x='bathrooms', y='price', data = df);
ax2.set_title('Price of Homes Based on Number of Bathrooms');
plt.xlabel('Bathrooms');
plt.ylabel('Price');
```
![Bar Plot Example 1](https://i.imgur.com/NRfIi8x.png)

As you can see, this graph is much easier to digest than the scatterplot giving us the average price of homes based on the number of bedrooms/bathrooms.

You can also seperate the bar plot even more using categorical variables by introducing the hue parameter to your code.

```
sns.barplot(x='bedrooms', y='price', hue='waterfront', data=df);
sns.barplot(x='bathrooms', y='price', hue='waterfront', data = df);
```

![Bar Plot Example 2](https://i.imgur.com/ZSjKYTZ.png)

And if you'd like, you can get rid of the confidence intervals, using the ci parameter, to make the bar plot easier to read.

```
sns.barplot(x='bedrooms', y='price', hue='waterfront', ci=None, data=df);
sns.barplot(x='bathrooms', y='price', hue='waterfront', ci=None, data = df);
```

![Bar Plot Example 3](https://i.imgur.com/SclUVc8.png)

There are many more parameters that you can use to further refine your bar graphs which can be found in the [seaborn bar plot documentation](https://seaborn.pydata.org/generated/seaborn.barplot.html).

Using bar plots allows us to see the relationship between bedrooms/bathrooms and price much more clearly, and also allows us to see the effect different categorical variables have on the pricing of homes with respect to the number of bedrooms/bathrooms.
