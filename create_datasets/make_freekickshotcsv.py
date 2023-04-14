#--------------------Free Kick shot model-------------------------#


# importing necessary libraries
import pandas as pd
import numpy as np
import ast


# Opening data data from wyscout 
train=pd.read_csv('data.csv')

freeKick_shot=pd.DataFrame(columns=['matchId','Type','SecondaryType','location', 'team','opponentTeam', 'Player', 'goal'])
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
          if secon=='free_kick_shot':
            #get details from free kicks
            freeKick_shot.at[i,'Type']=type_dic['primary']
            freeKick_shot.at[i,'SecondaryType']=type_dic['secondary']
              # get match Id
            freeKick_shot.at[i,'matchId']=select['matchId']
            #add events location
            freeKick_shot.at[i,'location']=select['location']
              #add team name in the Dataframe, opponent team and players
            freeKick_shot.at[i,'team']=team_dic['name']
            freeKick_shot.at[i,'opponentTeam']=opponentTeam_dic['name']
            freeKick_shot.at[i,'Player']= players_dic['name']
              #get goal from free kicks
            poss_dic=ast.literal_eval(select['possession'])
            attack=poss_dic['attack']
            freeKick_shot.at[i,'goal']=0
            if attack==None:
              # 0 for no goal from free kicks direct shot
              freeKick_shot.at[i,'goal']=0
              continue
            if(attack['withGoal']==True):
              freeKick_shot.at[i,'goal']=1
             
# Go through the dataframe and calculate X, Y co-ordinates.
# Distance from a line in the centre, distance squared, distance cube, 
# adjusted distance, adjusted distance squared, adjusted distance cube
# angle of shooting 
# arc length of shooting       
     
for i, loc in freeKick_shot.iterrows():
            Location=ast.literal_eval(loc['location'])
            freeKick_shot.at[i,'X']=100-Location['x']
            freeKick_shot.at[i,'Y']=Location['y']
            freeKick_shot.at[i,'C']=abs(Location['y']-50)
            #Distance in metres and shot angle in radians.
            x=freeKick_shot.at[i,'X']*105/100
            y=freeKick_shot.at[i,'C']*68/100
            freeKick_shot.at[i,'distance']=np.sqrt(x**2 + y**2)
            a = np.arctan(7.32 *x /(x**2 + y**2 - (7.32/2)**2))
            if a<0:
                a=np.pi+a
            freeKick_shot.at[i,'angle'] =a
            #arc length
            freeKick_shot.at[i,'arc_length']=freeKick_shot.at[i,'angle']*freeKick_shot.at[i,'distance']
            #distance squared
            freeKick_shot.at[i,'distance_squared']=np.power(freeKick_shot.at[i,'distance'],2)
            #distance cube
            freeKick_shot.at[i,'distance_cube']=np.power(freeKick_shot.at[i,'distance'],3)
            # adjusted distance
            freeKick_shot.at[i,'adj_distance']=(freeKick_shot.at[i,'distance']-16.5)
            #adjusted distance squared
            freeKick_shot.at[i,'adj_distance_squared']=np.power(freeKick_shot.at[i,'adj_distance'],2)
            #adjusted distance cube
            freeKick_shot.at[i,'adj_distance_cube']=np.power(freeKick_shot.at[i,'adj_distance'],3)
            
freeKick_shot.to_csv(' freeKick_shot.csv')
freeKick_shot=pd.read_csv(os.path.join('..', 'data', 'freeKick_shot.csv'))
