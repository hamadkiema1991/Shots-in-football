# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:57:20 2023

@author: hamad
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import FCPython 
import matplotlib.colors as colors


# Plot the number of shots from free kick 
# loading the data 
df=pd.read_csv('../data/freeKicks.csv')

# get free kicks from shot 
freeKickShot=df[df['free_kick_type']=='free_kick_shot']

# remove the bad co-ordinates from the data
# freeKickShot.drop([1154,1247,1433, 485, 484,1328, 1433,725,1412], axis=0, inplace=True)

#Two dimensional histogram
H_Shots=np.histogram2d(freeKickShot['X'], freeKickShot['Y'],bins=50,range=[[0, 100],[0, 100]])

#Plot the number of shots from different points
(fig,ax) = FCPython.createGoalMouth()
pos=ax.imshow(H_Shots[0], extent=[-1,66,104,-1], aspect='auto',label=True,cmap=plt.cm.Reds,vmin=0, vmax=50)
fig.colorbar(pos, ax=ax)
#make legend
ax.set_title('Number of shots from free kick')
plt.xlim((0,65))
plt.ylim((0,35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/Number_of_shoots_from_free_kick.png')
# Plot the number of crosses from free kick 

# get free kicks from cross
freeKickCross=df[df['free_kick_type']=='free_kick_cross']


#Two dimensional histogram
H_Cross=np.histogram2d(freeKickCross['X'], freeKickCross['Y'],bins=50,range=[[0, 100],[0, 100]])

#Plot the number of crosses from different points
(fig,ax) = FCPython.createGoalMouth()
pos=ax.imshow(H_Cross[0], extent=[-1,66,104,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=50)
fig.colorbar(pos, ax=ax)

#make legend
ax.set_title('Number of crosses from free kick')
plt.xlim((0,65))
plt.ylim((0,35))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/Number_of_shoots_from_free_kick.png')

# plot the number of shots and crosses
#Plot the number of shots and crosses from different points
(fig,ax) = FCPython.createGoalMouth()
cmap = plt.cm.bwr.copy()
cmap.set_bad('grey')
#Number of shots minus number of crosses from different points
pos = ax.imshow(H_Shots[0]-H_Cross[0],extent=[-1,66,104,-1], aspect='auto',cmap=cmap,vmin=-20, vmax=20, zorder = 1)
fig.colorbar(pos, ax=ax)
#make legend
ax.set_title('Number of shots minus of crosses from free kick')
plt.xlim((0,65))
plt.ylim((0,35))
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/Number_of_shots_minus_of_crosses_from free kick.png')

#Create a 2D map of xG
pgoal_2d_shot=np.zeros((68,68))
for x in range(68):
    for y in range(68):
       # Compute probability of goal to the left of penalty area     
     if (y<=13):
          a = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
          if a<0:
              a = np.pi + a
            #probability of scoring by free kick from shot  is given pgoal
          pgoal_shot=(1/(1+np.exp(3.985-4.879*a)))*100
          pgoal_2d_shot[x,y] =pgoal_shot
     # Compute probability of goal to the right of penalty area
     if (y>52):        
          a = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
          if a<0:
              a = np.pi + a
              #probability of scoring by free kick from shot  is given pgoal
          pgoal_shot=(1/(1+np.exp(3.985-4.879*a)))*100
          pgoal_2d_shot[x,y] =pgoal_shot
          # Compute probability of goal to above of penalty area
     if (y<=52)& (y>13) & (x>=16.5):
        a = np.arctan(7.32 *x /(x**2 + abs(y-68/2)**2 - (7.32/2)**2))
        if a<0:
            a = np.pi + a
            #probability of scoring by free kick from shot  is given pgoal
        pgoal_shot=(1/(1+np.exp(3.985-4.879*a)))*100
        pgoal_2d_shot[x,y] =pgoal_shot
        
(fig,ax) = FCPython.createGoalMouth()
pos=ax.imshow(pgoal_2d_shot, extent=[-1,68,68,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=8)
fig.colorbar(pos, ax=ax)
ax.set_title('Probability of goal from shot(percent)')
plt.xlim((0,65))
plt.ylim((0,35))
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/contour_plot_freeKick_shot.png')

# Contour plot of free kick from cross
pgoal_2d_cross=np.zeros((68,68))
for y in range(68):
    for x in range(68):
    # Compute probability of goal to the left of penalty area
     if (y<=13):
           #distance adjusted
           d= abs(np.sqrt(x**2 + abs(y-68/2)**2)-16.5)
           #distance adjusted cube
           d3= np.power(d,3)
           #probability of scoring by free kick from cross is given pgoal
           pgoal_cross=(1/(1+np.exp(3.0936 +0.0000587*d3)))*100
           pgoal_2d_cross[x,y] =  pgoal_cross
     # Compute probability of goal to the right of penalty area
     if (y>52):
           #distance adjusted
           d= abs(np.sqrt(x**2 + abs(y-68/2)**2)-16.5)
           #distance adjusted cube
           d3= np.power(d,3)
           #probability of scoring by free kick from cross  is given pgoal
           pgoal_cross=(1/(1+np.exp(3.0936 +0.0000587*d3)))*100
           pgoal_2d_cross[x,y] =  pgoal_cross
         # Compute probability of goal to above of penalty area
     if (y<=52)& (y>13) & (x>=16.5):
        #distance adjusted
        d= abs(np.sqrt(x**2 + abs(y-68/2)**2)-16.5)
        #distance adjusted cube
        d3= np.power(d,3)
        #probability of scoring by free kick from cross is given pgoal
        pgoal_cross=(1/(1+np.exp(3.0936 +0.0000587*d3)))*100
        pgoal_2d_cross[x,y] = pgoal_cross
        

(fig,ax) = FCPython.createGoalMouth()
pos=ax.imshow(pgoal_2d_cross, extent=[-1,68,68,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=8)
fig.colorbar(pos, ax=ax)
ax.set_title('Probability of goal from cross(percent)')
plt.xlim((0,65))
plt.ylim((0,35))
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig('../figures/contour_plot_freeKick_cross.png')

# Compare cross and shot probability
#plot probability
cmap = plt.cm.bwr.copy()
(fig,ax) = FCPython.createGoalMouth()
cmap = plt.cm.bwr
#cmap.set_bad('grey')
norm = colors.TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2)
pos=ax.imshow(pgoal_2d_shot-pgoal_2d_cross, extent=[-1,68,68,-1], aspect='auto',cmap=cmap, norm=norm)
fig.colorbar(pos, ax=ax)
ax.set_title('Probability of goal from shot minus from cross(%)')
plt.xlim((0,65))
plt.ylim((0,35))
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig('../figures/contour_plot_freeKick_shot_cross.png')