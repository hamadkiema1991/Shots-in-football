#------------------------------------------------------------#
# This file creates the freekicks.csv file which is a dataframe 
# with
#
# We combine five leagues(ligue1, premier league, Bundesliga, Serie A, LaLiga)


# importing necessary libraries
import pandas as pd
import numpy as np
import ast

# Opening data data from wyscout 
train=pd.read_csv('../data/data.csv')

freeKicks=pd.DataFrame(columns=['matchId','Type','SecondaryType','location', 'team','opponentTeam', 'Player', 'goal'])

for i, freekick in train.iterrows():
    #convert Components of the columns: type, team,player, opponent team and location from str to dictionnary
    type_dic=ast.literal_eval(freekick['type'])
    team_dic=ast.literal_eval(freekick['team'])
    players_dic=ast.literal_eval(freekick['player'])
    opponentTeam_dic=ast.literal_eval(freekick['opponentTeam'])

    # get free kicks from  
    if (type_dic['primary']=='free_kick'):
            #get details from free kicks
            freeKicks.at[i,'Type']=type_dic['primary']
            
            #Cross or shot or pass
            freeKicks.at[i,'SecondaryType']=type_dic['secondary']
            
            # get match Id
            freeKicks.at[i,'matchId']=freekick['matchId']
            
            # add events location
            freeKicks.at[i,'location']=freekick['location']
            
            # add team name in the Dataframe, opponent team and players
            freeKicks.at[i,'team']=team_dic['name']
            freeKicks.at[i,'opponentTeam']=opponentTeam_dic['name']
            freeKicks.at[i,'Player']= players_dic['name']
            
            # determine whether goal from free kick
            #
            poss_dic=ast.literal_eval(freekick['possession'])
            attack=poss_dic['attack']
            
            freeKicks.at[i,'goal']=0
            # handles the case where there is no attack
            if not(attack==None):
              # there is an attack
              if (attack['withGoal']==True):
                  # it ends in a goal
                  freeKicks.at[i,'goal']=1
                  
#create a column for free kick type
for i , freekick in freeKicks.iterrows():
    secon_type=freekick['SecondaryType']
    
    #get free kicks from others type that cross and shot
    freeKicks.at[i,'free_kick_type']='other_free_kick_type'
    
    # secon_type is a list that contains the free kicks details
    # in the data, secon_type is often empty
    if len(secon_type)!=0:
     for j in range(len(secon_type)):
      # get free kicks from shot
      if( secon_type[j]=='free_kick_shot') :
         freeKicks.at[i,'free_kick_type']='free_kick_shot'
         
      #get free kicks from cross
      if( secon_type[j]=='free_kick_cross') :
        freeKicks.at[i,'free_kick_type']='free_kick_cross'
        
    # Go through the dataframe and calculate X, Y co-ordinates.
    # Distance from a line in the centre, distance squared, distance cube, 
    # adjusted distance, adjusted distance squared, adjusted distance cube
    # angle of shooting 
    # arc length of shooting    
    Location=ast.literal_eval(freekick['location'])
    freeKicks.at[i,'X']=100-Location['x']
    freeKicks.at[i,'Y']=Location['y']
    freeKicks.at[i,'C']=abs(Location['y']-50)
    
    #Distance in metres and shot angle in radians.
    x=freeKicks.at[i,'X']*105/100
    y=freeKicks.at[i,'C']*68/100
    freeKicks.at[i,'distance']=np.sqrt(x**2 + y**2)
    a = np.arctan(7.32 *x /(x**2 + y**2 - (7.32/2)**2))
    if a<0:
      a=np.pi+a
    freeKicks.at[i,'angle'] =a
    
    #arc length
    freeKicks.at[i,'arc_length']=freeKicks.at[i,'angle']*freeKicks.at[i,'distance']
    
    #distance squared
    freeKicks.at[i,'distance_squared']=np.power(freeKicks.at[i,'distance'],2)
    
    #distance cube
    freeKicks.at[i,'distance_cube']=np.power(freeKicks.at[i,'distance'],3)
    
    # adjusted distance
    freeKicks.at[i,'adj_distance']=(freeKicks.at[i,'distance']-16.5)
    
    #adjusted distance squared
    freeKicks.at[i,'adj_distance_squared']=np.power(freeKicks.at[i,'adj_distance'],2)
    
    #adjusted distance cube
    freeKicks.at[i,'adj_distance_cube']=np.power(freeKicks.at[i,'adj_distance'],3)
    


# Save data              
freeKicks.to_csv('../data/freeKicks.csv')
