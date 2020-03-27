---
layout: post
title:      "Predicting Customer Churn Using Machine Learning"
date:       2020-03-27 11:41:12 +0000
permalink:  predicting_customer_churn_using_machine_learning
---


For a business, the cost of acquiring new customers is typically high when compared to retaining existing customers. Because of this, managing customer churn is an essential requirement of a successful business. Customer churn is the percentage of customers that stopped using your company's product or service during a ceratin time frame. Of course retaining all your customers is an impossible task, however, being able to predict which customers might leave in the future gives businesses valuable insights into their customers and their business. 

We'll be looking at data from SyriaTel, a telecommunications company, to try and predict whether or not a customer will churn. The dataset can be found [here](https://www.kaggle.com/becksddf/churn-in-telecoms-dataset).

**The Dataset**

We'll begin by cleaning up the data, getting rid of any unnecessary columns, convert categorical variables into integers, and seperating the target variable(churn) from the features.

```
X = df.drop(labels=['state', 'area code', 'phone number', 'number vmail messages', 'churn'], axis=1)
y = df['churn'].astype(int)
X['international plan'] = (X['international plan'] == 'yes').astype(int)
X['voice mail plan'] = (X['voice mail plan'] == 'yes').astype(int)
X.head()
```
![](https://i.imgur.com/P819sVM.png?1)

Next we'll take a look at the percentage of churn amongst customers.
```
plt.figure(figsize=(10, 5))
plt.pie(y.value_counts(sort=True), labels=('No', 'Yes'), autopct='%1.1f%%', shadow=True, startangle=270)
plt.title('Percentage of Churn in Dataset')
plt.show()
```
![](https://i.imgur.com/e2K7RF1.png)

Looks like SyriaTel has a customer churn rate of 14%.

**Using Models to Predict Churn**

There are many good models to use for classification:
* Logistic Regression
* K-Nearest Neighbors
* Decision Trees
* Support Vector Machines
* XGBoost

For this case, I'll be comparing Random Forest, which uses decision trees, and XGBoost. We'll get started by splitting up the data into training and test sets.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

Running Random Forest first, we'll create a pipeline to scale our data and use gridsearch to optimize hyperparameters.

Using gridsearch only finds the optimal value that you input into the grid so the optimization is limited to the number of hyperparameters you input. However, adding hyperparameters to test in gridsearch increases the runtime quite a bit as you add extra so be careful not to add to many.
```
random_forest_pipeline = Pipeline([('ss', StandardScaler()),
                                   ('RF', RandomForestClassifier(random_state=42))])
																	 
rf_grid = [{'RF__criterion': ['gini', 'entropy'],
            'RF__n_estimators': [10, 30, 100],
            'RF__max_depth': [None, 2, 5, 9, 15],
            'RF__max_features': [None, 'auto'],
            'RF__min_samples_split': [2, 5, 10],
            'RF__min_samples_leaf': [1, 4, 9]}]

rf_gridsearch = GridSearchCV(estimator=random_forest_pipeline, 
                             param_grid=rf_grid,
                             scoring='accuracy',
                             cv=5)

rf_gridsearch.fit(X_train, y_train)
```

After running the data through random forest, these are the scores for our evaluation metrics:
* Precision: 96.6%
* Recall: 68.8%
* Accuracy: 95%
* F1 Score: 80.3%

The metrics that you want to focus on are unique to each problem. For this problem, I believe accuracy and recall are most important. Accuracy because you want to be fairly confident that a customer will churn if the model predicts it. Recall because minimizing false negatives(model fails to predict a customer will churn) is most beneficial to the company. A false positive(model predicts a customer will churn when they don't) doesn't hurt the company because the customer will ultimately stay, however a false negative means that the customer will leave without the company being able to attempt to retain them.

This model performed extremely well in all metrics aside from recall, which has a decent score of 69%.

Now let's check the scores of our XGBoost Classifier. We'll use this code to run the model:
```
xgboost_pipeline = Pipeline([('ss', StandardScaler()),
                             ('XG', xgb.XGBClassifier(random_state=42))])
														 
xg_grid = [{'XG__learning_rate': [0.01, 0.1],
            'XG__max_depth': [3, 6, 9],
            'XG__min_child_weight': [5, 10, 20],
            'XG__subsample': [0.3, 0.7],
            'XG__n_estimators': [5, 30, 100, 250]}]

xg_gridsearch = GridSearchCV(estimator=xgboost_pipeline, 
                             param_grid=xg_grid,
                             scoring='accuracy',
                             cv=5)
														 
xg_gridsearch.fit(X_train, y_train)
```

Here are the scores for the XGBoost Classifier:
Precision: 93.7%
Recall: 71.2%
Accuracy: 95%
F1 Score: 81%

The scores are pretty similar to the random forest model but precision is slightly lower and recall is slightly higher. Because recall is more important than precision for this problem, XGBoost is the preferred model to use to predict customer churn for SyriaTel
