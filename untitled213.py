# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:26:39 2023

@author: hamad
"""


"""
Created on Mon Jan  2 10:28:04 2023

@author: hamad
"""

#Statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf

# importing necessary libraries
import pandas as pd
import numpy as np
import ast
import plotly.io as pio 
pio.renderers.default="browser"
plt_idx = 0
#------------------------------------------------------------#
# In this file, we fitted models such that:
#free kick from shot
# free kick from cross

#------------------------------------------------------------#
# Opening data
# data from wyscout 
#load data - store it in train dataframe
#  we combined five leagues(ligue1, premier league, Bundesliga, Serie A, LaLiga)

train=pd.read_csv('data.csv')


#---------------------Direct Free Kick model-------------------------#

freeKick=pd.DataFrame(columns=['matchId','Type','SecondaryType','location', 'team','opponentTeam', 'Player', 'goal'])
for i, select in train.iterrows():
    #convert Components of the columns: type, team,player, opponent team and location from str to dictionnary
    type_dic=ast.literal_eval(select['type'])
    team_dic=ast.literal_eval(select['team'])
    players_dic=ast.literal_eval(select['player'])
    opponentTeam_dic=ast.literal_eval(select['opponentTeam'])

    # get free kicks from  direct shot
    if (type_dic['primary']=='free_kick'):
            #get details from free kicks
            freeKick.at[i,'Type']=type_dic['primary']
            freeKick.at[i,'SecondaryType']=type_dic['secondary']
              # get match Id
            freeKick.at[i,'matchId']=select['matchId']
            #add events location
            freeKick.at[i,'location']=select['location']
              #add team name in the Dataframe, opponent team and players
            freeKick.at[i,'team']=team_dic['name']
            freeKick.at[i,'opponentTeam']=opponentTeam_dic['name']
            freeKick.at[i,'Player']= players_dic['name']
              #get goal from free kicks
            poss_dic=ast.literal_eval(select['possession'])
            attack=poss_dic['attack']
            freeKick.at[i,'goal']=0
            if attack==None:
              # 0 for no goal from free kicks direct shot
              freeKick.at[i,'goal']=0
              continue
            if(attack['withGoal']==True):
              freeKick.at[i,'goal']=1
             
#Go through the dataframe and calculate X, Y co-ordinates.
#Distance from a line in the centre, distance squared, distance cube, 
# adjusted distance, adjusted distance squared, adjusted distance cube
# angle of shooting 
# arc length of shooting            
for i, loc in freeKick.iterrows():
            Location=ast.literal_eval(loc['location'])
            freeKick.at[i,'X']=100-Location['x']
            freeKick.at[i,'Y']=Location['y']
            freeKick.at[i,'C']=abs(Location['y']-50)
            #Distance in metres and shot angle in radians.
            x=freeKick.at[i,'X']*105/100
            y=freeKick.at[i,'C']*68/100
            freeKick.at[i,'distance']=np.sqrt(x**2 + y**2)
            a = np.arctan(7.32 *x /(x**2 + y**2 - (7.32/2)**2))
            if a<0:
                a=np.pi+a
            freeKick.at[i,'angle'] =a
            #arc length
            freeKick.at[i,'arc_length']=freeKick.at[i,'angle']*freeKick.at[i,'distance']
            #distance squared
            freeKick.at[i,'distance_squared']=np.power(freeKick.at[i,'distance'],2)
            #distance cube
            freeKick.at[i,'distance_cube']=np.power(freeKick.at[i,'distance'],3)
            # adjusted distance
            freeKick.at[i,'adj_distance']=(freeKick.at[i,'distance']-16.5)
            #adjusted distance squared
            freeKick.at[i,'adj_distance_squared']=np.power(freeKick.at[i,'adj_distance'],2)
            #adjusted distance cube
            freeKick.at[i,'adj_distance_cube']=np.power(freeKick.at[i,'adj_distance'],3)
freeKick.to_csv(' freeKick.csv')
freeKick=pd.read_csv(' freeKick.csv')

freeKicks_shoot=pd.DataFrame()
for i, select in freeKick.iterrows():
    type_second=select['SecondaryType']
    for secon in type_second:
        if secon=='free_kick_shot':
          freeKicks_shoot.at[i,]
            