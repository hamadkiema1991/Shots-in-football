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
train=pd.read_csv('data.csv')

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
     
# Save data              
freeKicks.to_csv('freeKicks.csv')