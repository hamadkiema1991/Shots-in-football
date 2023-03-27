
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

# train=pd.read_csv('data.csv')


# freeKicks=pd.DataFrame(columns=['matchId','Type','SecondaryType','location', 'team','opponentTeam', 'Player', 'goal'])
# for i, select in train.iterrows():
#     #convert Components of the columns: type, team,player, opponent team and location from str to dictionnary
#     type_dic=ast.literal_eval(select['type'])
#     team_dic=ast.literal_eval(select['team'])
#     players_dic=ast.literal_eval(select['player'])
#     opponentTeam_dic=ast.literal_eval(select['opponentTeam'])

#     # get free kicks from  
#     if (type_dic['primary']=='free_kick'):
#             #get details from free kicks
#             freeKicks.at[i,'Type']=type_dic['primary']
#             freeKicks.at[i,'SecondaryType']=type_dic['secondary']
#               # get match Id
#             freeKicks.at[i,'matchId']=select['matchId']
#             #add events location
#             freeKicks.at[i,'location']=select['location']
#               #add team name in the Dataframe, opponent team and players
#             freeKicks.at[i,'team']=team_dic['name']
#             freeKicks.at[i,'opponentTeam']=opponentTeam_dic['name']
#             freeKicks.at[i,'Player']= players_dic['name']
#               #get goal from free kicks
#             poss_dic=ast.literal_eval(select['possession'])
#             attack=poss_dic['attack']
#             freeKicks.at[i,'goal']=0
#             if attack==None:
#               # 0 for no goal from free kicks 
#               freeKicks.at[i,'goal']=0
#               continue
#             if(attack['withGoal']==True):
#               freeKicks.at[i,'goal']=1
# freeKicks.to_csv('freeKicks.csv')
freeKicks=pd.read_csv('freeKicks.csv')
#---------------------Direct Free Kick model-------------------------#

# freeKick_shot=pd.DataFrame(columns=['matchId','Type','SecondaryType','location', 'team','opponentTeam', 'Player', 'goal'])
# for i, select in train.iterrows():
#     #convert Components of the columns: type, team,player, opponent team and location from str to dictionnary
#     type_dic=ast.literal_eval(select['type'])
#     team_dic=ast.literal_eval(select['team'])
#     players_dic=ast.literal_eval(select['player'])
#     opponentTeam_dic=ast.literal_eval(select['opponentTeam'])

#     # get free kicks from  direct shot
#     if (type_dic['primary']=='free_kick'):
#       type_second=type_dic['secondary']
#       for secon in type_second:
#           if secon=='free_kick_shot':
#             #get details from free kicks
#             freeKick_shot.at[i,'Type']=type_dic['primary']
#             freeKick_shot.at[i,'SecondaryType']=type_dic['secondary']
#               # get match Id
#             freeKick_shot.at[i,'matchId']=select['matchId']
#             #add events location
#             freeKick_shot.at[i,'location']=select['location']
#               #add team name in the Dataframe, opponent team and players
#             freeKick_shot.at[i,'team']=team_dic['name']
#             freeKick_shot.at[i,'opponentTeam']=opponentTeam_dic['name']
#             freeKick_shot.at[i,'Player']= players_dic['name']
#               #get goal from free kicks
#             poss_dic=ast.literal_eval(select['possession'])
#             attack=poss_dic['attack']
#             freeKick_shot.at[i,'goal']=0
#             if attack==None:
#               # 0 for no goal from free kicks direct shot
#               freeKick_shot.at[i,'goal']=0
#               continue
#             if(attack['withGoal']==True):
#               freeKick_shot.at[i,'goal']=1
             
#Go through the dataframe and calculate X, Y co-ordinates.
#Distance from a line in the centre, distance squared, distance cube, 
# adjusted distance, adjusted distance squared, adjusted distance cube
# angle of shooting 
# arc length of shooting            
# for i, loc in freeKick_shot.iterrows():
#             Location=ast.literal_eval(loc['location'])
#             freeKick_shot.at[i,'X']=100-Location['x']
#             freeKick_shot.at[i,'Y']=Location['y']
#             freeKick_shot.at[i,'C']=abs(Location['y']-50)
#             #Distance in metres and shot angle in radians.
#             x=freeKick_shot.at[i,'X']*105/100
#             y=freeKick_shot.at[i,'C']*68/100
#             freeKick_shot.at[i,'distance']=np.sqrt(x**2 + y**2)
#             a = np.arctan(7.32 *x /(x**2 + y**2 - (7.32/2)**2))
#             if a<0:
#                 a=np.pi+a
#             freeKick_shot.at[i,'angle'] =a
#             #arc length
#             freeKick_shot.at[i,'arc_length']=freeKick_shot.at[i,'angle']*freeKick_shot.at[i,'distance']
#             #distance squared
#             freeKick_shot.at[i,'distance_squared']=np.power(freeKick_shot.at[i,'distance'],2)
#             #distance cube
#             freeKick_shot.at[i,'distance_cube']=np.power(freeKick_shot.at[i,'distance'],3)
#             # adjusted distance
#             freeKick_shot.at[i,'adj_distance']=(freeKick_shot.at[i,'distance']-16.5)
#             #adjusted distance squared
#             freeKick_shot.at[i,'adj_distance_squared']=np.power(freeKick_shot.at[i,'adj_distance'],2)
#             #adjusted distance cube
#             freeKick_shot.at[i,'adj_distance_cube']=np.power(freeKick_shot.at[i,'adj_distance'],3)
# freeKick_shot.to_csv(' freeKick_shot.csv')
# freeKick_shot=pd.read_csv(os.path.join('..', 'data', 'freeKick_shot.csv'))
freeKick_shot=pd.read_csv('freeKick_shot.csv')
# A Data frame to fit free kick model
model_variables=['angle','arc_length','distance', 'distance_squared','distance_cube',
         'adj_distance','adj_distance_squared','adj_distance_cube','goal']
#          'adj_distance','adj_distance_squared','adj_distance_cube','goal']
fit_freeKick_shot=freeKick_shot[model_variables]

fit_freeKick_shot.to_csv('fit_freeKick_shot.csv')

fit_freeKick_shot=pd.read_csv('fit_freeKick_shot.csv')

#fitting the model

#fit the model
freeKick_shot_test_model= smf.glm(formula="goal ~angle", data=fit_freeKick_shot, 
                           family=sm.families.Binomial()).fit()
print(freeKick_shot_test_model.summary())
print(freeKick_shot_test_model.aic)

#get number of goal by free kick from shot
goal=list(fit_freeKick_shot['goal'])

# 1 means goal 
count_FreeKickShotgoal=goal.count(1)
print(count_FreeKickShotgoal)


#---------------------Free Kick from cross model-------------------------#

# freeKick_cross=pd.DataFrame(columns=['matchId','Type','SecondaryType','location', 'team','opponentTeam', 'Player', 'goal'])
# for i, select in train.iterrows():
#     #convert Components of the columns: type, team,player, opponent team and location from str to dictionnary
#     type_dic=ast.literal_eval(select['type'])
#     team_dic=ast.literal_eval(select['team'])
#     players_dic=ast.literal_eval(select['player'])
#     opponentTeam_dic=ast.literal_eval(select['opponentTeam'])

#     # get free kicks from  direct shot
#     if (type_dic['primary']=='free_kick'):
#       type_second=type_dic['secondary']
#       for secon in type_second:
#           if secon=='free_kick_cross':
#             #get details from free kicks
#             freeKick_cross.at[i,'Type']=type_dic['primary']
#             freeKick_cross.at[i,'SecondaryType']=type_dic['secondary']
#               # get match Id
#             freeKick_cross.at[i,'matchId']=select['matchId']
#             #add events location
#             freeKick_cross.at[i,'location']=select['location']
#               #add team name in the Dataframe, opponent team and players
#             freeKick_cross.at[i,'team']=team_dic['name']
#             freeKick_cross.at[i,'opponentTeam']=opponentTeam_dic['name']
#             freeKick_cross.at[i,'Player']= players_dic['name']
#               #get goal from free kicks
#             poss_dic=ast.literal_eval(select['possession'])
#             attack=poss_dic['attack']
#             freeKick_cross.at[i,'goal']=0
#             if attack==None:
#               # 0 for no goal from free kicks direct shot
#               freeKick_cross.at[i,'goal']=0
#               continue
#             if(attack['withGoal']==True):
#               freeKick_cross.at[i,'goal']=1
             
# Go through the dataframe and calculate X, Y co-ordinates.
# Distance from a line in the centre, distance squared, distance cube, 
# adjusted distance, adjusted distance squared, adjusted distance cube
# angle of shooting 
# arc length of shooting            
# for i, loc in freeKick_cross.iterrows():
#             Location=ast.literal_eval(loc['location'])
#             freeKick_cross.at[i,'X']=100-Location['x']
#             freeKick_cross.at[i,'Y']=Location['y']
#             freeKick_cross.at[i,'C']=abs(Location['y']-50)
#             #Distance in metres and shot angle in radians.
#             x=freeKick_cross.at[i,'X']*105/100
#             y=freeKick_cross.at[i,'C']*68/100
#             freeKick_cross.at[i,'distance']=np.sqrt(x**2 + y**2)
#             a = np.arctan(7.32 *x /(x**2 + y**2 - (7.32/2)**2))
#             if a<0:
#                 a=np.pi+a
#             freeKick_cross.at[i,'angle'] =a
#             #arc length
#             freeKick_cross.at[i,'arc_length']=freeKick_cross.at[i,'angle']*freeKick_cross.at[i,'distance']
#             #distance squared
#             freeKick_cross.at[i,'distance_squared']=np.power(freeKick_cross.at[i,'distance'],2)
#             #distance cube
#             freeKick_cross.at[i,'distance_cube']=np.power(freeKick_cross.at[i,'distance'],3)
#             # adjusted distance
#             freeKick_cross.at[i,'adj_distance']=(freeKick_cross.at[i,'distance']-16.5)
#             #adjusted distance squared
#             freeKick_cross.at[i,'adj_distance_squared']=np.power(freeKick_cross.at[i,'adj_distance'],2)
#             #adjusted distance cube
#             freeKick_cross.at[i,'adj_distance_cube']=np.power(freeKick_cross.at[i,'adj_distance'],3)
# freeKick_cross.to_csv(' freeKick_cross.csv')

freeKick_cross=pd.read_csv('freeKick_cross.csv')

# A Data frame to fit free kick model
model_variables=['angle','arc_length','distance', 'distance_squared','distance_cube',
          'adj_distance','adj_distance_squared','adj_distance_cube','goal']
fit_freeKick_cross=freeKick_cross[model_variables]

fit_freeKick_cross.to_csv('fit_freeKick_cross.csv')

fit_freeKick_cross=pd.read_csv('fit_freeKick_cross.csv')

#fitting the model

#fit the model
freeKick_cross_test_model= smf.glm(formula="goal ~distance_squared", data=fit_freeKick_cross, 
                           family=sm.families.Binomial()).fit()
print(freeKick_cross_test_model.summary())
print(freeKick_cross_test_model.aic)


#get number of goal by head shot
goal=list(fit_freeKick_cross['goal'])

# 1 means goal 
count_FreeKickCross_goal=goal.count(1)
print(count_FreeKickCross_goal)
