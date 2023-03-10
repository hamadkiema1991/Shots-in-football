# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:00:57 2023

@author: hamad
"""
import pandas as pd
import numpy as np
from mplsoccer import Pitch, Sbopen
import FCPython 
import ast
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

# k=0           
# freeKick_model=pd.read_csv('freeKick_model.csv')
# for i, select in freeKick_model.iterrows():
#               test=ast.literal_eval(select['possession'])
#               dse=test['types']
#               for j in range(len(dse)):
#                if(dse[j]=='direct_free_kick'):
#                  k=k+1
 
# Plot the number of shots from free kick 
# loading the data 
df_freeKick=pd.read_csv('../Data/df_freeKick.csv')
# remove the bad co-ordinates from the data
df_freeKick.drop([1154,1247,1433, 485, 484,1328, 1433,725,1412], axis=0, inplace=True)



#Two dimensional histogram
H_Shots=np.histogram2d(df_freeKick['X'], df_freeKick['Y'],bins=50,range=[[0, 100],[0, 100]])

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
plt.show()

# Plot the number of crosses from free kick 
# loading the data 
df_freeKickCross=pd.read_csv('../Data/df_freeKickCross.csv')


#Two dimensional histogram
H_Cross=np.histogram2d(df_freeKickCross['X'], df_freeKickCross['Y'],bins=50,range=[[0, 100],[0, 100]])
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
plt.show()

# plot the number of shots and crosses
#Plot the number of shots and crosses from different points
(fig,ax) = FCPython.createGoalMouth()
cmap = plt.cm.bwr
cmap.set_bad('grey')
#Number of shots minus number of crosses from different points
pos = ax.imshow(H_Shots[0]-H_Cross[0],extent=[-1,66,104,-1], aspect='auto',cmap=cmap,vmin=-20, vmax=20, zorder = 1)
fig.colorbar(pos, ax=ax)
#make legend
ax.set_title('Number of shots minus of crosses from free kick')
plt.xlim((0,65))
plt.ylim((0,35))
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

