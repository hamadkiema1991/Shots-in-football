# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 22:01:56 2023

@author: hamad
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:42:38 2023

@author: hamad
"""
import numpy as np
import pandas as pd                                   #For dataframe operations
from sklearn.model_selection import train_test_split  #For generating training and test sets
import xgboost as xgb                                    #For calculating the probs
from matplotlib import pyplot as plt                  #For visualizing the results

df = pd.read_csv('../data/freeKicks.csv')

columns = ['distance', 'distance_squared', 'distance_cube', 'adj_distance',
                    'adj_distance_squared', 'adj_distance_cube','angle', 'arc_length']


freeKickShot=df[df['free_kick_type']=='free_kick_shot']
#construct the feature matrix and the target variable
X = freeKickShot[columns]
y = freeKickShot['goal']


# freeKickCross=df[df['free_kick_type']=='free_kick_cross']
# #construct the feature matrix and the target variable
# X = freeKickCross[columns]
# y = freeKickCross['goal']

# freeKickPass=df[df['free_kick_type']=='other_free_kick_type']
# #construct the feature matrix and the target variable
# X = freeKickPass[columns]
# y = freeKickPass['goal']

#Get the train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost (different learning rate)
learning_rate_range = np.arange(0.01, 1, 0.05) 
test_XG = [] 
train_XG = [] 
for lr in learning_rate_range : 
    xgb_classifier = xgb.XGBClassifier(eta = lr) 
    xgb_classifier.fit(X_train, y_train) 
    train_XG.append(xgb_classifier.score(X_train, y_train)) 
    test_XG.append(xgb_classifier.score(X_test, y_test))

fig = plt.figure(figsize=(10, 7)) 
plt.plot(learning_rate_range, train_XG, c='orange', label='Train') 
plt.plot(learning_rate_range, test_XG, c='m', label= 'Test') 
plt.xlabel('Learning rate')
plt.xticks(learning_rate_range)
plt.ylabel('Accuracy score')
plt.ylim(0.6, 1) 
plt.legend(prop={'size' : 12}, loc=3) 
plt.title('Accuracy score vs. Learning rate of XGBoost', size=14)
plt.show()


# new learning rate range
learning_rate_range = np.arange(0.01, 1, 0.05)
fig = plt.figure(figsize=(19, 17))
idx = 1
# grid search for min_child_weight
for weight in np.arange(0, 4.5, 0.5):
    train = []
    test = []
    for lr in learning_rate_range:
        xgb_classifier = xgb.XGBClassifier(eta = lr, reg_lambda=1, min_child_weight=weight)
        xgb_classifier.fit(X_train, y_train)
        train.append(xgb_classifier.score(X_train, y_train))
        test.append(xgb_classifier.score(X_test, y_test))
    fig.add_subplot(3, 3, idx)
    idx += 1
    plt.plot(learning_rate_range, train, c='orange', label='Training')
    plt.plot(learning_rate_range, test, c='m', label='Testing')
    plt.xlabel('Learning rate')
    plt.xticks(learning_rate_range)
    plt.ylabel('Accuracy score')
    plt.ylim(0.5, 1)
    plt.legend(prop={'size': 12}, loc=3)
    title = "Min child weight:" + str(weight)
    plt.title(title, size=16)
plt.show()
