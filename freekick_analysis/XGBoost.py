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
X = freeKickShot
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
X_te=X_test[X_test['goal']==1]
X_train=X_train[columns]
y_train=y_train
X_teste= X_test[columns]
y_test= y_test


free=freeKickShot[columns]


model = xgb.XGBClassifier()
mod=model.fit(X_train, y_train)
test=mod.predict(free)
freeKickShot['goal_test2']=test
sc=freeKickShot[freeKickShot['goal_test2']==1]
import pitch

(fig,ax) = pitch.createGoalMouth()
ax.scatter(sc.Y, sc.X)
plt.show()

import pitch

# #plot pitch
# (fig,ax) = pitch.createGoalMouth()

# #plot probability

# ax.scatter(freeKickShot.Y, freeKickShot.X)
# ax.scatter(sca.Y, sca.X)
# plt.show()

(fig,ax) = pitch.createGoalMouth()
cool=freeKickShot[freeKickShot['goal']==1]
H_other=np.histogram2d(cool['X'], cool['Y'],bins=25,range=[[0, 25],[0, 25]])
pos = ax.imshow(H_other[0],extent=[-1,68,104,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=3, zorder = 1)
fig.colorbar(pos, ax=ax)
plt.show()

(fig,ax) = pitch.createGoalMouth()
ax.scatter(cool.Y, cool.X)
plt.show()
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




import numpy as np
#Create a 2D map of xG
pgoal=np.zeros((68,68))
for x in range(68):
    for y in range(68):
     
        # Compute probability of goal to the left of penalty area     
      if (y>14) & (y<=53)& (x>=16.5):
          value=[]
          xG=pd.DataFrame(columns=['distance', 'distance_squared', 'distance_cube', 'adj_distance',
                              'adj_distance_squared', 'adj_distance_cube','angle', 'arc_length'])
          #angle
          angle = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
          if angle<0:
              angle= np.pi + angle
          #distance
          distance=np.sqrt(x**2+y**2)
          #distance squared
          distance_squared=np.power(distance,2)
          #distance cube
          distance_cube=np.power(distance,3)
          #adjusted distance
          adj_distance=abs(distance-16.5)
          #adjusted distance squared
          adj_distance_squared=np.power(adj_distance,2)
          # adjusted distance cube
          adj_distance_cube=np.power(adj_distance,3)
          #arc length
          arc_length=distance*angle
          #
          xG['distance']=[distance]
            #arc length
          xG['arc_length']=[arc_length]

            #distance squared
          xG['distance_squared']=[distance_squared]

            #distance cube
          xG['distance_cube']=[np.power(distance,3)]
           # adjusted distance
          xG['adj_distance']=[adj_distance]

           #adjusted distance squared
          xG['adj_distance_squared']=[adj_distance_squared]

          #adjusted distance cube
          xG['adj_distance_cube']=[adj_distance_cube]

          #angle
          xG['angle']=[angle]
          #make predictions
          xg= model.predict(xG)*100
          pgoal[x,y]=xg
        # Compute probability of goal to the left of penalty area     
      if (y>53):
          value=[]
          xG=pd.DataFrame(columns=['distance', 'distance_squared', 'distance_cube', 'adj_distance',
                              'adj_distance_squared', 'adj_distance_cube','angle', 'arc_length'])
          #angle
          angle = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
          if angle<0:
              angle= np.pi + angle
          #distance
          distance=np.sqrt(x**2+y**2)
          #distance squared
          distance_squared=np.power(distance,2)
          #distance cube
          distance_cube=np.power(distance,3)
          #adjusted distance
          adj_distance=abs(distance-16.5)
          #adjusted distance squared
          adj_distance_squared=np.power(adj_distance,2)
          # adjusted distance cube
          adj_distance_cube=np.power(adj_distance,3)
          #arc length
          arc_length=distance*angle
          #
          xG['distance']=[distance]
            #arc length
          xG['arc_length']=[arc_length]

            #distance squared
          xG['distance_squared']=[distance_squared]

            #distance cube
          xG['distance_cube']=[np.power(distance,3)]
           # adjusted distance
          xG['adj_distance']=[adj_distance]

           #adjusted distance squared
          xG['adj_distance_squared']=[adj_distance_squared]

          #adjusted distance cube
          xG['adj_distance_cube']=[adj_distance_cube]

          #angle
          xG['angle']=[angle]
          #make predictions
          xg= model.predict(xG)*100
          pgoal[x,y]=xg
      if (y<=14):
          value=[]
          xG=pd.DataFrame(columns=['distance', 'distance_squared', 'distance_cube', 'adj_distance',
                              'adj_distance_squared', 'adj_distance_cube','angle', 'arc_length'])
          #angle
          angle = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
          if angle<0:
              angle= np.pi + angle
          #distance
          distance=np.sqrt(x**2+y**2)
          #distance squared
          distance_squared=np.power(distance,2)
          #distance cube
          distance_cube=np.power(distance,3)
          #adjusted distance
          adj_distance=abs(distance-16.5)
          #adjusted distance squared
          adj_distance_squared=np.power(adj_distance,2)
          # adjusted distance cube
          adj_distance_cube=np.power(adj_distance,3)
          #arc length
          arc_length=distance*angle
          #
          xG['distance']=[distance]
            #arc length
          xG['arc_length']=[arc_length]

            #distance squared
          xG['distance_squared']=[distance_squared]

            #distance cube
          xG['distance_cube']=[np.power(distance,3)]
           # adjusted distance
          xG['adj_distance']=[adj_distance]

           #adjusted distance squared
          xG['adj_distance_squared']=[adj_distance_squared]

          #adjusted distance cube
          xG['adj_distance_cube']=[adj_distance_cube]

          #angle
          xG['angle']=[angle]
          #make predictions
          xg= model.predict(xG)*100
          pgoal[x,y]=xg


import pitch

#plot pitch
(fig,ax) = pitch.createGoalMouth()

#plot probability
pos = ax.imshow(pgoal, extent=[-1,68,68,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=100, zorder = 1)
fig.colorbar(pos, ax=ax)




pgoal1=np.zeros((100,100))
for x in range(100):
    for y in range(100):
     
        # Compute probability of goal to the left of penalty area     
      # if (y>14) & (y<=53)& (x>=16.5):
          xG=pd.DataFrame(columns=['distance', 'distance_squared', 'distance_cube', 'adj_distance',
                              'adj_distance_squared', 'adj_distance_cube','angle', 'arc_length'])
          #angle
          angle = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
          if angle<0:
              angle= np.pi + angle
          #distance
          distance=np.sqrt(x**2+y**2)
          #distance squared
          distance_squared=np.power(distance,2)
          #distance cube
          distance_cube=np.power(distance,3)
          #adjusted distance
          adj_distance=abs(distance-16.5)
          #adjusted distance squared
          adj_distance_squared=np.power(adj_distance,2)
          # adjusted distance cube
          adj_distance_cube=np.power(adj_distance,3)
          #arc length
          arc_length=distance*angle
          #
          xG['distance']=[distance]
            #arc length
          xG['arc_length']=[arc_length]

            #distance squared
          xG['distance_squared']=[distance_squared]

            #distance cube
          xG['distance_cube']=[np.power(distance,3)]
           # adjusted distance
          xG['adj_distance']=[adj_distance]

           #adjusted distance squared
          xG['adj_distance_squared']=[adj_distance_squared]

          #adjusted distance cube
          xG['adj_distance_cube']=[adj_distance_cube]

          #angle
          xG['angle']=[angle]
          #make predictions
          xg= model.predict(xG)
          pgoal1[x,y]=xg
import pitch

#plot pitch
(fig,ax) = pitch.createGoalMouth()

#plot probability
pos = ax.imshow(pgoal1, extent=[-1,100,100,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=1, zorder = 1)
fig.colorbar(pos, ax=ax)

