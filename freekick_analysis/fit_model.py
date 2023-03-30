# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:48:34 2023

@author: hamad
"""
#Statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf

# importing necessary libraries
import pandas as pd

import plotly.io as pio 
pio.renderers.default="browser"
plt_idx = 0
#-------------------------------------------------------------------------------------------------#
#we fit the free kick from shot and from cross
# we use the features following:
# distance of shooting
#the distance squared
# the distance cube
# adjusted distance that is la distance between the penalty area and the distance of shooting
# adjusted distance squared
#ajusted distance cube
# angle of shooting
# arc length that is the distance of shooting(meters) times by angle(radian)

#we use the AIC method and backward stepwise to found the best model
#---------------------------------------------------------------------------------------------------#



#Loading the data
df=pd.read_csv('../data/freeKicks.csv')

#loading the data from free kicks of shooting
freeKickShot=df[df['free_kick_type']=='free_kick_shot']

# list the model variables you want here
model_variables = ['distance', 'distance_squared', 'distance_cube', 'adj_distance',
                   'adj_distance_squared', 'adj_distance_cube','angle', 'arc_length']
model=''
for v in model_variables[:-1]:
    model = model  + v + ' + '
model = model + model_variables[-1]

#fit the free kick from shot model
freeKick_shot_test_model= smf.glm(formula="goal ~ " + model, data=freeKickShot, 
                           family=sm.families.Binomial()).fit()

#print summary
print(freeKick_shot_test_model.summary())

#print the AIC's value of model
print('the AIC value of the all model is',freeKick_shot_test_model.aic)


# after the backward stepwise and compared the different AIC's value the best model is 
# the following model 

#best model
b_freeKick_shot_test_model= smf.glm(formula="goal ~angle", data=freeKickShot, 
                           family=sm.families.Binomial()).fit()
#print summary
print(b_freeKick_shot_test_model.summary())

#print the AIC's value of model
print('the AIC value of the best model is', b_freeKick_shot_test_model.aic)

#get number of goal from free kick of shooting
goal=list(freeKickShot['goal'])

# 1 means goal 
count_FreeKickShotgoal=goal.count(1)

#print the number of goals from free kicks of shooting
print('The number of goals from free kicks of shooting is',count_FreeKickShotgoal)

#loading the data from free kicks of crossing
freeKickCross=df[df['free_kick_type']=='free_kick_cross']

# list the model variables you want here
model_variables = ['distance', 'distance_squared', 'distance_cube', 'adj_distance',
                   'adj_distance_squared', 'adj_distance_cube','angle', 'arc_length']
model=''
for v in model_variables[:-1]:
    model = model  + v + ' + '
model = model + model_variables[-1]

#fit the model
freeKick_cross_test_model= smf.glm(formula="goal ~ " + model, data=freeKickCross, 
                           family=sm.families.Binomial()).fit()

#print summary
print(freeKick_cross_test_model.summary())

#print the AIC's value of model
print('the AIC value of the all model is',freeKick_cross_test_model.aic)


# after the backward stepwise and compared the different AIC's value the best model is 
# the following model 

#best model
b_freeKick_cross_test_model= smf.glm(formula="goal ~adj_distance_cube", data=freeKickCross, 
                           family=sm.families.Binomial()).fit()
#print summary
print(b_freeKick_cross_test_model.summary())


#print the AIC's value of model
print('the AIC value of the best model is', b_freeKick_cross_test_model.aic)


#get number of goal by head shot
goal=list(freeKickCross['goal'])

# 1 means goal 
count_FreeKickCross_goal=goal.count(1)

#print the number from free kick of crossing
print('The number of goals from free kicks of crossing is',count_FreeKickCross_goal)
