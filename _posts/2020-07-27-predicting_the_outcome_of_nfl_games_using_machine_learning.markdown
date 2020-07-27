---
layout: post
title:      "Predicting the Outcome of NFL Games Using Machine Learning"
date:       2020-07-27 10:38:09 +0000
permalink:  predicting_the_outcome_of_nfl_games_using_machine_learning
---


![](https://i.imgur.com/98xMf7J.jpg)

The NFL is the most successful sports league in America generating $13 billion dollars annually and sports gambling has seen a meteoric rise in popularity with an estimated value of $150 billion annually for American sports. Is it possible to predict NFL games at a high percentage using machine learning to take advantage of sports gambling and consistently turn a profit?

I'll be making 3 separate models to see which model performs the best in predicting the outcome of NFL games.

1. Logistic Regression
2. Random Forest
3. XGBoost

## Data Preparation
I'll be using data from [the football database](http://footballdb.com) website for the 2017-2019 seasons. For this project, I only used the team stats data from the boxscore of each game. After acquiring the data from each game, I cleaned it up and converted it into a separate dataframe with the columns that I will be using. Here is an example of data for one game.

![](https://i.imgur.com/m66viBm.jpg)

The visiting team is always the first row and the home team is always the second row, and I needed to get all the data into one row for each game. I added a "V_" or an "H_" in front of the column names for the visiting and home teams respectively and combined both rows into one row.

![](https://i.imgur.com/NobXxul.jpg)

After checking to see if I cleaned the data properly for one game, I did the same for every regular season game in 2017-2019 and combined all the games into one dataframe.

![](https://i.imgur.com/BKSABbE.jpg)

In order to predict future games, I decided to use the averages of the past 3 games played by each team. I then created another dataframe that would have the past 3 game averages for each of the games played in 2017-2019. This is only possible starting from week 4, so this next dataframe does not include weeks 1-3. I did initially add weeks 2 and 3 by using the averages of the games played prior, but it lowered the models performance so I chose to remove those datapoints. 

![](https://i.imgur.com/FfYmjqd.jpg)

Now that we've gotten all of our data ready, lets move on to trying to predict the outcome of NFL games!

![](https://i.imgur.com/B3827An.png)

From the NFL regular season data from 2017-2019, the home team wins their games 56% of the time, so I'm looking for an accuracy rate greater than 56% to consider the models a success.

## 1. Logistic Regression
We'll start with a logistic regression model. I used gridsearch to find the optimal parameters for the model.
```
logreg = LogisticRegression(fit_intercept=False, solver='liblinear', random_state=27)

logistic_grid = [{'C': [1, 50, 2000, 1e15],
                  'penalty': ['l1', 'l2']}]

logistic_gridsearch = GridSearchCV(estimator=logreg,
                                   param_grid=logistic_grid,
                                   scoring='accuracy',
                                   cv=5)
																	 
logistic_gridsearch.fit(X_train, y_train)
```

For all the models, I got the best results by training the models on actual game data and using those models to predict the outcome of the past 3 averages game data.
We'll use this snippet of code to get the evaluation metrics for the model.

```
pred = logistic_gridsearch.predict(scaled_pred_data)
print(confusion_matrix(y_pred_data, pred))
print(classification_report(y_pred_data, pred))
```
![](https://i.imgur.com/MQk147K.jpg)

The accuracy score for the logistic regression model is 57%. It is barely above the 56% threshold to consider the model a success and the other evaluation metrics are very low with an F1-score of 57%. Lets see if we get better results with the other models.

## 2. Random Forest
We'll start off by creating and fitting a random forest model to game data.
```
rf_clf = RandomForestClassifier(random_state=27)

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

Then we'll figure out the evaluation metrics for the random forest model by using the code above but swapping out the logistic regression model for our random forest model. 

```
pred = rf_gridsearch.predict(scaled_pred_data)
print(confusion_matrix(y_pred_data, pred))
print(classification_report(y_pred_data, pred))
```
![](https://i.imgur.com/ZiovWXi.jpg)

The random forest model performs slightly better than logistic regression with an accuracy score of 58%. The other evaluation metrics for random forest also improved with an F1-score of 63%, however, this model is also too close to the 56% threshold to consider the model a success.

## 3. XGBoost

Next we'll take a look at an XGBoost model. We'll do the same as before and fit the model to our game data.

```
xgboost = xgb.XGBClassifier(random_state=27)

xg_grid = [{'learning_rate': [0.01, 0.1],
            'max_depth': [3, 6, 9],
            'min_child_weight': [5, 10, 20],
            'subsample': [0.3, 0.7],
            'n_estimators': [5, 30, 100, 250]}]

xg_gridsearch = GridSearchCV(estimator=xgboost, 
                             param_grid=xg_grid,
                             scoring='accuracy',
                             cv=5)
														 
xg_gridsearch.fit(X_train, y_train)
```
```
pred = xg_gridsearch.predict(scaled_pred_data)
print(confusion_matrix(y_pred_data, pred))
print(classification_report(y_pred_data, pred))
```
![](https://i.imgur.com/bpN4dxW.jpg)

The XGBoost model performs the best of all 3 models with an accuracy score of 60% and an F1-score of 64%. 

## Conclusion

The best performing model, XGBoost, is able to predict the outcome of NFL games with an accuracy rate of 60%. With the data that I used, while the models can predict the outcome of NFL games better than just choosing the home team to win everytime, it is not at a high enough rate to be able to correctly predict the outcome of NFL games consistently. While the model alone might not prove too useful, the model could be combined with other information such as spreads to hopefully have a better shot at predicting the outcome of an NFL game.
