---
layout: post
title:      "Hypothesis Testing on Northwind Database"
date:       2020-01-28 12:23:01 +0000
permalink:  hypothesis_testing_on_northwind_database
---


As part of a project, I was given the Northwind database to generate valuable analytical insights that could be of value to the company. The Northwind database is a sample database created by Microsoft for a fictitious company called Northwind Traders which imports and exports consumables around the world. For this, I would need to come up with a couple questions to be used for hypothesis testing. The first question I was to answer was:

> **Does discount have a statistically significant effect on the quantity of a product in an order? If so, at what level(s) of discount?**

**First, we'll define our hypotheses:**

Null Hypothesis: There is no difference in quantities between discounts and no discount

Alternative Hypothesis: There is a difference in quantities between discounts and no discount

The alpha level is set at 0.05

**Obtaining the data**

To answer the first part of the question I needed to take a look at the database and extract the correct data.
![](https://i.imgur.com/mz0Pw4m.png)

To answer the question I needed to get Quantity, and Discount from OrderDetail table
```
cur.execute("""SELECT Quantity, Discount
               FROM OrderDetail;""")
df1 = pd.DataFrame(cur.fetchall())
df1.columns = [x[0] for x in cur.description]
df1.head()
```
![](https://i.imgur.com/2ppFGQ3.png)

Next I took a look at how many different discount levels there are.
```
df1.Discount.value_counts()
```
![](https://i.imgur.com/juakf1S.png)

Looking at the discount levels, I removed discounts of 1-4% and 6% because of their low numbers.

**Hypothesis Testing**

Now I was able to run an ANOVA test on quantity with discount as a categorical variable.
```
formula = 'quantity ~ C(discount)'
lm = ols(formula, disc_df).fit()
table = sm.stats.anova_lm(lm, typ=2)
print(table)
```
![](https://i.imgur.com/khDaQj4.png)

With a p value much lower than 0.05, we can reject the null hypothesis and accept the alternative hypothesis that discount has a statistically significant effect on the quantity of product ordered. Next to find out which discount levels are statistically significant, I ran a Tukey HSD test to find out exactly where the significant differences lie.

```
from statsmodels.stats.multicomp import pairwise_tukeyhsd

m_comp = pairwise_tukeyhsd(endog=disc_df['quantity'], groups=disc_df['discount'], alpha=0.05)
print(m_comp)
```
![](https://i.imgur.com/jS0bI9K.png)

While the Tukey test compares all groups to each other, the only ones we are interested in are the discount levels compared to no discount. With this we can see that a discount level of 10% is the only one that does not have a significant difference when compared to no discount, so we can reject the null hypothesis on all discount levels except for the 10% discount level. However, the most effective discount levels are 5%, 10%, and 15% each with a p value of 0.001.

![](https://i.imgur.com/OiyN6AS.png)

After finding out how discount affected the quantity, I wanted to see if it had any effect on revenue, so the second question I answered was:
> **Does discount amount have a statistically significant effect on the average revenue of a product in an order?**

**Define our hypotheses:**

Null Hypothesis: There is no difference in average revenue between discounts and no discount

Alternative Hypothesis: There is a difference in average revenue between discounts and no discount

Alpha is set at 0.05

**Obtaining the data**

The data we need for this question is almost the same as the first question but we just need to add UnitPrice from the OrderDetail table.

![](https://i.imgur.com/9ok8hmz.png)

Then in order to calculate revenue, I created a new column called 'Total' in the dataframe multiplying unit price and quantity while taking into account the discount level.

```
revenue_df['Total'] = round(revenue_df.UnitPrice * (1 - revenue_df.Discount) * revenue_df.Quantity, 2)
```
![](https://i.imgur.com/hJkBQb7.png)

**Hypothesis Testing**

With this I can now run an ANOVA test on revenue using discount as a categorical variable.
```
formula = 'Total ~ C(discount)'
lm = ols(formula, revenue_df).fit()
table = sm.stats.anova_lm(lm, typ=2)
print(table)
```
![](https://i.imgur.com/2kRM3Kt.png)

With a p-value greater than 0.05, I fail to reject the null hypothesis and conclude that discount levels do not have a statistically significant effect on average revenue per order. However, since the p value is so close to 0.05 I ran a Tukey test to see if any discount levels were statistically significant when compared to no discount.
```
m_comp = pairwise_tukeyhsd(endog=revenue_df['Total'], groups=revenue_df['discount'], alpha=0.05)
print(m_comp)
```
![](https://i.imgur.com/L098bES.png)

Once again, we're only interested in the comparisons of each discount level to no discount and we can see that the discount level of 5% has a statistically significant effect on average revenue per order with a p value of 0.03. We can reject the null hypothesis for the discount level at 5%, but we fail to reject the null hypothesis for every other discount level.

![](https://i.imgur.com/5FMZ7EQ.png)

**Conclusions**

Based on my findings, if Northwind was trying to utilize discount to get rid of inventory quickly they should use a discount level of 5%, 15%, or 25%. However, if Northwind wants to maximize revenue while also utilizing a discount, they should use a discount level of 5%
