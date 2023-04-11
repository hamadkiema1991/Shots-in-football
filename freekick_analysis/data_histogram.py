# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 20:57:20 2023

@author: hamad
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pitch


#-----------------------------------------free kicks from shot -------------------------------------------#

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
(fig,ax) = pitch.createGoalMouth()
pos=ax.imshow(H_Shots[0], extent=[-1,68,104,-1], aspect='auto',label=True,cmap=plt.cm.Reds,vmin=0, vmax=50)
fig.colorbar(pos, ax=ax)
#make legend
ax.set_title('Number of shots from free kick')
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/Number_of_shoots_from_free_kick.png')


# -----------------------------------free kicks from----------------------------------------------------------#

# Plotting the number of crosses from free kick 

# get free kicks from cross
freeKickCross=df[df['free_kick_type']=='free_kick_cross']


#Two dimensional histogram
H_Cross=np.histogram2d(freeKickCross['X'], freeKickCross['Y'],bins=50,range=[[0, 100],[0, 100]])

#Plot the number of crosses from different points
(fig,ax) = pitch.createGoalMouth()
pos=ax.imshow(H_Cross[0], extent=[-1,68,104,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=50)
fig.colorbar(pos, ax=ax)

#make legend
ax.set_title('Number of crosses from free kick')
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/Number_of_crosses_from_free_kick.png')


# ----------------------------------free kicks from shot vs from cross------------------------------#
#Plot the number of shots and crosses from different points
(fig,ax) = pitch.createGoalMouth()
cmap = plt.cm.bwr.copy()
cmap.set_bad('grey')
#Number of shots minus number of crosses from different points
pos = ax.imshow(H_Shots[0]-H_Cross[0],extent=[-1,68,104,-1], aspect='auto',cmap=cmap,vmin=-20, vmax=20, zorder = 1)
fig.colorbar(pos, ax=ax)
#make legend
ax.set_title('Number of shots minus of crosses from free kick')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/Number_of_shots_minus_of_crosses_from_free_kick.png')

#-------------------------------Open play free kicks type-----------------------------------------------------------#


# Plotting the number of other  free kick type

# get other free kicks type
freeKick_other=df[df['free_kick_type']=='other_free_kick_type']


#Two dimensional histogram
H_other=np.histogram2d(freeKick_other['X'], freeKick_other['Y'],bins=50,range=[[0, 100],[0, 100]])

#Plot the number of crosses from different points
(fig,ax) = pitch.createGoalMouth()
pos=ax.imshow(H_other[0], extent=[-1,68,104,-1], aspect='auto',cmap=plt.cm.Reds,vmin=0, vmax=100)
fig.colorbar(pos, ax=ax)

#make legend
ax.set_title('Number of open play free kicks')
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/Number_of_open_play_free_kicks.png')

# -----------------------------open play free kicks and free kicks from cross------------------------------------#

#Plot the number of crosses  and other type from different points
(fig,ax) = pitch.createGoalMouth()
cmap = plt.cm.bwr.copy()
cmap.set_bad('grey')
#Number of shots minus number of crosses from different points
pos = ax.imshow(H_other[0]-H_Cross[0],extent=[-1,68,104,-1], aspect='auto',cmap=cmap,vmin=-30, vmax=30, zorder = 1)
fig.colorbar(pos, ax=ax)
#make legend
ax.set_title('Number  of open play free kicks minus from crosses')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('../figures/Number_of_open_play_minus_of_crosses_from_free_kicks.png')