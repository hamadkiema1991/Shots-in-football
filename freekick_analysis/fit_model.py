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
import numpy as np
import pitch 
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import matplotlib.colors as colors
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

#------------------free kick from shoot model------------------------------------------#

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
alpha=b_freeKick_shot_test_model.params
#print the AIC's value of model
print('the AIC value of the best model is', b_freeKick_shot_test_model.aic)

#get number of goal from free kick of shooting
goal=list(freeKickShot['goal'])

# 1 means goal 
count_FreeKickShotgoal=goal.count(1)

#print the number of goals from free kicks of shooting
print('The number of goals from free kicks of shooting is',count_FreeKickShotgoal)


#Create a 2D map of xG
pgoal_2d_shot=np.zeros((68,68))
for x in range(68):
    for y in range(68):
       # Compute probability of goal to the left of penalty area     
     if (y<=14):
          a = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
          if a<0:
              a = np.pi + a
            #probability of scoring by free kick from shot  is given pgoal
          pgoal_shot=(1/(1+np.exp(-alpha[0]-alpha[1]*a)))*100
          pgoal_2d_shot[x,y] =pgoal_shot
     # Compute probability of goal to the right of penalty area
     if (y>53):        
          a = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
          if a<0:
              a = np.pi + a
              #probability of scoring by free kick from shot  is given pgoal
          pgoal_shot=(1/(1+np.exp(-alpha[0]-alpha[1]*a)))*100
          pgoal_2d_shot[x,y] =pgoal_shot
          # Compute probability of goal to above of penalty area
     if (y<=53)& (y>14) & (x>=16.5):
        a = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
        if a<0:
            a = np.pi + a
            #probability of scoring by free kick from shot  is given pgoal
        pgoal_shot=(1/(1+np.exp(-alpha[0]-alpha[1]*a)))*100
        pgoal_2d_shot[x,y] =pgoal_shot
        
#plot pitch
(fig,ax) = pitch.createGoalMouth()

#plot probability
pos = ax.imshow(pgoal_2d_shot, extent=[-1,68,68,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=8, zorder = 1)
fig.colorbar(pos, ax=ax)

#make legend
ax.set_title('Free kicks from shot Probability of goal')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/contour_plot_freeKicks_shot.png')       

#----------------------------free kick from cross model-----------------------------------------#

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

beta=b_freeKick_cross_test_model.params

#get number of goal by head shot
goal=list(freeKickCross['goal'])

# 1 means goal 
count_FreeKickCross_goal=goal.count(1)

#print the number from free kick of crossing
print('The number of goals from free kicks of crossing is',count_FreeKickCross_goal)



# Contour plot of free kick from cross
pgoal_2d_cross=np.zeros((68,68))
for y in range(68):
    for x in range(68):
    # Compute probability of goal to the left of penalty area
     if (y<=14):
           
           #distance adjusted
           d= abs(np.sqrt(x**2 + abs(y-68/2)**2)-16.5)
           #distance adjusted cube
           d3= np.power(d,3)
           #probability of scoring by free kick from cross is given pgoal
           pgoal_cross=(1/(1+np.exp(-beta[0]-beta[1]*d3)))*100
           pgoal_2d_cross[x,y] =  pgoal_cross
     # Compute probability of goal to the right of penalty area
     if (y>53):
           #distance adjusted
           d= abs(np.sqrt(x**2 + abs(y-68/2)**2)-16.5)
           #distance adjusted cube
           d3= np.power(d,3)
           #probability of scoring by free kick from cross  is given pgoal
           pgoal_cross=(1/(1+np.exp(-beta[0]-beta[1]*d3)))*100
           pgoal_2d_cross[x,y] =  pgoal_cross
         # Compute probability of goal to above of penalty area
     if (y<=53)& (y>14) & (x>=16.5):
        #distance adjusted
        d= abs(np.sqrt(x**2 + abs(y-65/2)**2)-16.5)
        #distance adjusted cube
        d3= np.power(d,3)
        #probability of scoring by free kick from cross is given pgoal
        pgoal_cross=(1/(1+np.exp(-beta[0]-beta[1]*d3)))*100
        pgoal_2d_cross[x,y] = pgoal_cross
        

#plot pitch
(fig,ax) = pitch.createGoalMouth()

#plot probability
pos = ax.imshow(pgoal_2d_cross, extent=[-1,68,68,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=8, zorder = 1)
fig.colorbar(pos, ax=ax)

#make legend
ax.set_title('free kicks from cross probability of goal')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/contour_plot_freeKick_cross.png')


# --------------------Open play free kick model-------------------------------------#

#loading the data from free kicks of crossing
freeKick_Others=df[df['free_kick_type']=='other_free_kick_type']

# list the model variables you want here
model_variables = ['distance', 'distance_squared', 'distance_cube', 'adj_distance',
               'adj_distance_squared', 'adj_distance_cube','angle','arc_length']

model=''
for v in model_variables[:-1]:
    model = model  + v + ' + '
model = model + model_variables[-1]

#fit the model
freeKick_Others_test_model= smf.glm(formula="goal ~ " + model, data=freeKick_Others, 
                           family=sm.families.Binomial()).fit()

#print summary
print(freeKick_Others_test_model.summary())

#print the AIC's value of model
print('the AIC value of the all model is',freeKick_Others_test_model.aic)


# after the backward stepwise and compared the different AIC's value the best model is 
# the following model 

#best model
b_freeKick_Others_test_model= smf.glm(formula="goal ~angle", data=freeKick_Others, 
                           family=sm.families.Binomial()).fit()
#print summary
print(b_freeKick_Others_test_model.summary())

gamma=b_freeKick_Others_test_model.params
#print the AIC's value of model
print('the AIC value of the best model is', b_freeKick_Others_test_model.aic)


#get number of goal by head shot
goal=list(freeKick_Others['goal'])

# 1 means goal 
count_FreeKick_Others_goal=goal.count(1)

#print the number from free kick of crossing
print('The number of goals from free kicks of crossing is',count_FreeKick_Others_goal)


# Contour plot of free kick from cross
pgoal_2d=np.zeros((68,68))
for y in range(68):
    for x in range(68):
    # Compute probability of goal to the left of penalty area
     if (y<=14):
           a = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
           if a<0:
               a = np.pi + a
           #distance adjusted
           d= abs(np.sqrt(x**2 + abs(y-68/2)**2)-16.5)
           #distance adjusted cube
           d3= np.power(d,3)
           #probability of scoring by free kick from cross is given pgoal
           pgoal=(1/(1+np.exp(-gamma[0]-gamma[1]*a)))*100
           pgoal_2d[x,y] =  pgoal
     # Compute probability of goal to the right of penalty area
     if (y>53):
           a = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
           if a<0:
               a = np.pi + a
           #distance adjusted
           d= abs(np.sqrt(x**2 + abs(y-68/2)**2)-16.5)
           #distance adjusted cube
           d3= np.power(d,3)
           #probability of scoring by free kick from cross is given pgoal
           pgoal=(1/(1+np.exp(-gamma[0]-gamma[1]*a)))*100
           pgoal_2d[x,y] =  pgoal
         # Compute probability of goal to above of penalty area
     if (y<=53)& (y>14) & (x>=16.5):
        a = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
        if a<0:
            a = np.pi + a
        #distance adjusted
        d= abs(np.sqrt(x**2 + abs(y-68/2)**2)-16.5)
        #distance adjusted cube
        d3= np.power(d,3)
        #probability of scoring by free kick from cross is given pgoal
        pgoal=(1/(1+np.exp(-gamma[0]-gamma[1]*a)))*100
        pgoal_2d[x,y] =  pgoal

#plot pitch
(fig,ax) = pitch.createGoalMouth()

#plot probability
pos = ax.imshow(pgoal_2d, extent=[-1,68,68,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=8, zorder = 1)
fig.colorbar(pos, ax=ax)

#make legend
ax.set_title('Probability of goal open play free kicks')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/contour_plot_freeKicks_Open_play.png')

#-------------------------Free kicks from shot vs free kicks from cross---------------------------------------#



#plot pitch
(fig,ax) = pitch.createGoalMouth()
cmap = plt.cm.bwr.copy()
cmap.set_bad('grey')
# pitch = VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
# fig, ax = pitch.draw()
#plot probability
pos = ax.imshow(pgoal_2d_shot-pgoal_2d_cross, extent=[-1,68,68,-1], aspect='auto',cmap=cmap,vmin=-2, vmax=2, zorder = 1)
fig.colorbar(pos, ax=ax)
#make legend
ax.set_title('Probability of goal from shot minus from cross')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/contour_plot_freeKicks_Shot_Cross.png')


#-----------------------other free kicks type vs free kicks from cross----------------------------------#


#plot pitch
(fig,ax) = pitch.createGoalMouth()
cmap = plt.cm.bwr.copy()
cmap.set_bad('grey')
# pitch = VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
# fig, ax = pitch.draw()
#plot probability
pos = ax.imshow(pgoal_2d_cross-pgoal_2d, extent=[-1,68,68,-1], aspect='auto',cmap=cmap,vmin=-4, vmax=4, zorder = 1)
fig.colorbar(pos, ax=ax)
#make legend
ax.set_title('Probability of goal from open play free kicks minus from cross')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/contour_plot_Open_play_freeKicks_and_Cross.png')

#---------------------------------------------other free kicks type vs free kicks from shot------------------#


#plot pitch
(fig,ax) = pitch.createGoalMouth()
cmap = plt.cm.bwr.copy()
cmap.set_bad('grey')
# pitch = VerticalPitch(line_color='black', half = True, pitch_type='custom', pitch_length=105, pitch_width=68, line_zorder = 2)
# fig, ax = pitch.draw()
#plot probability
pos = ax.imshow(pgoal_2d_shot-pgoal_2d, extent=[-1,68,68,-1], aspect='auto',cmap=cmap,vmin=-4, vmax=4, zorder = 1)
fig.colorbar(pos, ax=ax)
#make legend
ax.set_title('Probability of goal from open play free kicks minus from shot')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/contour_plot_Open_play_freeKicks_and_shot.png')

#-------------------------------------------------------------------------------#

#best model
d_freeKick_cross_test_model= smf.glm(formula="goal ~distance_squared", data=freeKickCross, 
                           family=sm.families.Binomial()).fit()
#print summary
print(d_freeKick_cross_test_model.summary())


#print the AIC's value of model
print('the AIC value of the best model is', d_freeKick_cross_test_model.aic)

sigma=d_freeKick_cross_test_model.params




c0=(4*alpha[1])/7.32
c1=(alpha[0]-sigma[0])/beta[1]+(c0/(2*sigma[1]))**2
y=np.sqrt(c1+(x+c0/2*sigma[1])**2)
plt.plot(x,y)


x= np.arange(0, 40)
y=np.arange(-20,20)

X, Y = np.meshgrid(x, y)
# Make a function tha compute the different between probability to shot and to cross from a free kick
def f(x, y):
    pg=y**2-c1+(x+c0/2*sigma[1])**2
    return pg
#plotting
Z= f(X,Y)
fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z,[0], colors='black')
ax.clabel(CS, CS.levels,inline=True,  fontsize=10)

