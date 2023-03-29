#---------------------Free Kick from cross model-------------------------#


# importing necessary libraries
import pandas as pd
import numpy as np
import ast


# Opening data data from wyscout 
train=pd.read_csv('data.csv')


freeKick_cross=pd.DataFrame(columns=['matchId','Type','SecondaryType','location', 'team','opponentTeam', 'Player', 'goal'])
for i, select in train.iterrows():
    #convert Components of the columns: type, team,player, opponent team and location from str to dictionnary
    type_dic=ast.literal_eval(select['type'])
    team_dic=ast.literal_eval(select['team'])
    players_dic=ast.literal_eval(select['player'])
    opponentTeam_dic=ast.literal_eval(select['opponentTeam'])

    # get free kicks from  direct shot
    if (type_dic['primary']=='free_kick'):
      type_second=type_dic['secondary']
      for secon in type_second:
          if secon=='free_kick_cross':
            #get details from free kicks
            freeKick_cross.at[i,'Type']=type_dic['primary']
            freeKick_cross.at[i,'SecondaryType']=type_dic['secondary']
              # get match Id
            freeKick_cross.at[i,'matchId']=select['matchId']
            #add events location
            freeKick_cross.at[i,'location']=select['location']
              #add team name in the Dataframe, opponent team and players
            freeKick_cross.at[i,'team']=team_dic['name']
            freeKick_cross.at[i,'opponentTeam']=opponentTeam_dic['name']
            freeKick_cross.at[i,'Player']= players_dic['name']
              #get goal from free kicks
            poss_dic=ast.literal_eval(select['possession'])
            attack=poss_dic['attack']
            freeKick_cross.at[i,'goal']=0
            if attack==None:
              # 0 for no goal from free kicks direct shot
              freeKick_cross.at[i,'goal']=0
              continue
            if(attack['withGoal']==True):
              freeKick_cross.at[i,'goal']=1
             
# Go through the dataframe and calculate X, Y co-ordinates.
# Distance from a line in the centre, distance squared, distance cube, 
# adjusted distance, adjusted distance squared, adjusted distance cube
# angle of shooting 
# arc length of shooting   
         
for i, loc in freeKick_cross.iterrows():
            Location=ast.literal_eval(loc['location'])
            freeKick_cross.at[i,'X']=100-Location['x']
            freeKick_cross.at[i,'Y']=Location['y']
            freeKick_cross.at[i,'C']=abs(Location['y']-50)
            #Distance in metres and shot angle in radians.
            x=freeKick_cross.at[i,'X']*105/100
            y=freeKick_cross.at[i,'C']*68/100
            freeKick_cross.at[i,'distance']=np.sqrt(x**2 + y**2)
            a = np.arctan(7.32 *x /(x**2 + y**2 - (7.32/2)**2))
            if a<0:
                a=np.pi+a
            freeKick_cross.at[i,'angle'] =a
            #arc length
            freeKick_cross.at[i,'arc_length']=freeKick_cross.at[i,'angle']*freeKick_cross.at[i,'distance']
            #distance squared
            freeKick_cross.at[i,'distance_squared']=np.power(freeKick_cross.at[i,'distance'],2)
            #distance cube
            freeKick_cross.at[i,'distance_cube']=np.power(freeKick_cross.at[i,'distance'],3)
            # adjusted distance
            freeKick_cross.at[i,'adj_distance']=(freeKick_cross.at[i,'distance']-16.5)
            #adjusted distance squared
            freeKick_cross.at[i,'adj_distance_squared']=np.power(freeKick_cross.at[i,'adj_distance'],2)
            #adjusted distance cube
            freeKick_cross.at[i,'adj_distance_cube']=np.power(freeKick_cross.at[i,'adj_distance'],3)
freeKick_cross.to_csv(' freeKick_cross.csv')

