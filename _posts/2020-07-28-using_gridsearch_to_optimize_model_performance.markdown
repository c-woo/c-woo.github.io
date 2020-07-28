---
layout: post
title:      "Using Gridsearch to Optimize Model Performance"
date:       2020-07-28 00:41:50 -0400
permalink:  using_gridsearch_to_optimize_model_performance
---


Hyperparameters are parameters that are not directly learnt within a model and must be set before the learning process begins. Gridsearch is used to find the optimal hyperparameters of a model to optimize model performance. Gridsearch will exhaustively consider all hyperparameter combinations within a grid of parameter values and return the optimal values for each hyperparameter. 

Using Gridsearch only finds the optimal value from the paramaters that you input into the grid so the optimization is limited to the number of hyperparameters you input. However, as you add parameters to the grid, the unique combinations that can be made from the grid will increase exponentially which will significantly increase runtime, so be careful not to add too many.

Let's take a look at Gridsearch by looking at data from SyriaTel, a telecommunications company, to try and predict whether or not a customer will churn. The dataset can be found [here](https://www.kaggle.com/becksddf/churn-in-telecoms-dataset).

## Data Preparation

Let's take a look at the first five rows of the dataset

```
df = pd.read_csv('bigml_59c28831336c6604c800002a.csv')
df.head()
```
![](https://i.imgur.com/JbbcsNB.jpg)

There is not much cleaning needed for this dataset other than to drop any unnecessary columns. We'll do that while separating our target variable from the dataset.

```
X = df.drop(labels=['state', 'area code', 'phone number', 'number vmail messages', 'churn'], axis=1)
y = df['churn'].astype(int)
```

Next we'll separate our data into training and testing sets.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```

Before we get started with building the models, lets take a quick look at the data

![](https://i.imgur.com/rEMgMcE.jpg)

Looking at the data, there is a lot of class imbalance, so we'll be using SMOTE to take care of the class imbalance.

```
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_sample(X_train, y_train)
```

Now that we have our data prepared for our models, lets take a look at how gridsearch can improve our models performance!

## Using Gridsearch
I'll be using Gridsearch on a logistic regression model and a random forest model to see how it improves model performance. 

### 1. Logistic Regression

Lets begin by creating a baseline logistic regression model.

```
logreg = LogisticRegression(solver='liblinear', random_state=42)
logreg.fit(X_train_resampled, y_train_resampled)
```

Now we'll check our accuracy score with the baseline model.

```
logreg.score(X_test, y_test)
```

With the baseline model the accuracy score we get is **78.18%**. Let us see if we can improve upon that using Gridsearch. To use Gridseach we need to create a grid of parameters the function can iterate through to find the best parameter values that are given in the grid. Then we'll pass in our logistic regression estimator and our grid, the scoring metric and cross validation folds are optional. 

```
from sklearn.model_selection import GridSearchCV

logistic_grid = [{'C': [1, 50, 2000, 1e15],
                  'penalty': ['l1', 'l2']}]

logistic_gridsearch = GridSearchCV(estimator=logreg,
                                   param_grid=logistic_grid,
                                   scoring='accuracy',
                                   cv=5)
																	 
logistic_gridsearch.fit(X_train_resampled, y_train_resampled)
```
![](https://i.imgur.com/zOG2N2e.jpg)

We can find the optimal parameters gridsearch has found by running this line of code.

```
logistic_gridsearch.best_params_
```
![](https://i.imgur.com/uxOIXi8.jpg)

The accuracy score we get with Gridsearch is **78.67%**. It improved our models performance very slightly. Let's see if using Gridsearch with random forest has better results with more parameters in the grid.

### 2. Random Forest

We'll do the same as before and create a baseline random forest model.

```
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

rf_clf.score(X_test, y_test)
```

The baseline model has an accuracy score of **93.41%**. Let us see if we can improve upon this using Gridsearch.

```
rf_grid = [{'criterion': ['gini', 'entropy'],
            'n_estimators': [10, 30, 100],
            'max_depth': [None, 2, 5, 9, 15],
            'max_features': [None, 'auto'],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 4, 9]}]

rf_gridsearch = GridSearchCV(estimator=rf_clf, 
                             param_grid=rf_grid,
                             scoring='accuracy',
                             cv=5)
														 
rf_gridsearch.fit(X_train, y_train)
```
![](https://i.imgur.com/VJ4iGG8.jpg)

```
rf_gridsearch.best_params_
```
![](https://i.imgur.com/v39on9v.jpg)

The accuracy score we get with Gridsearch is **94.96%**. The model performance improved by **1.5%** even though the baseline model already had an accuracy score greater than **90%**! 

Gridsearch can be used with more parameters and parameter values to optimize performance but be careful not to add too many parameters or else your runtime may take ages to run!







