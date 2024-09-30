# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:33:44 2024

@author: devam
"""
#importing libraries
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import  nba_api
from nba_api.stats.endpoints import HustleStatsBoxScore
import scipy.optimize as opt
from sklearn import preprocessing
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#%%
# Initialize an empty dataframe for storing results
df_hustle = pd.DataFrame()

# Iterate through unique GAME_IDs
for team_id in df_teams_log['GAME_ID'].unique():
    try:
        # Fetch hustle stats and concatenate to the dataframe
        df_hustle = pd.concat([df_hustle, HustleStatsBoxScore(str(team_id)).team_stats.get_data_frame()], axis=0)
        
        # Introduce a delay of 2 seconds between each request
        time.sleep(1)
    
    except Exception as e:
        print(f"Error fetching data for team {team_id}: {e}")
        
#%%
#normalizing data
import copy
df_c = copy.deepcopy(df_hustle)

columns = ['CONTESTED_SHOTS', 'CONTESTED_SHOTS_2PT',
       'CONTESTED_SHOTS_3PT', 'DEFLECTIONS', 'CHARGES_DRAWN', 'SCREEN_ASSISTS',
       'SCREEN_AST_PTS', 'OFF_LOOSE_BALLS_RECOVERED',
       'DEF_LOOSE_BALLS_RECOVERED', 'LOOSE_BALLS_RECOVERED', 'OFF_BOXOUTS',
       'DEF_BOXOUTS', 'BOX_OUT_PLAYER_TEAM_REBS', 'BOX_OUT_PLAYER_REBS',
       'BOX_OUTS']
for column in columns:
    df_c[column] = (df_c[column]-df_c[column].min())/(df_c[column].max()-df_c[column].min())
    
#%%
#resetting index
df_c = df_c.reset_index()
df_c = df_c.drop('index', axis=1)

#%%
#Creating a win/loss column
df_c['Win/Loss'] = np.nan

#%%
#Filling win/loss column

for i in range(0, len(df_c), 2):
    if df_c.loc[i, 'PTS'] > df_c.loc[i+1, 'PTS']:
        df_c.loc[i, 'Win/Loss'] = 'Win'
        df_c.loc[i+1, 'Win/Loss'] = 'Loss'
    else:
        df_c.loc[i+1, 'Win/Loss'] = 'Win'
        df_c.loc[i, 'Win/Loss'] = 'Loss'
#%%
#creating total hustle column
df_c['Hustle_total'] = df_c['CONTESTED_SHOTS'] + df_c['DEFLECTIONS'] + df_c['CHARGES_DRAWN'] + df_c['SCREEN_ASSISTS'] + df_c['LOOSE_BALLS_RECOVERED'] + df_c['BOX_OUTS']
    
#%%
#testing if hustle avg is higher in wins
df_plot = pd.DataFrame(df_c.groupby('Win/Loss')['Hustle_total'].mean())
#%%
#Making data table
data = [['Win', 1.937415], ['Loss', 1.816253]]

# Convert to DataFrame for easy plotting
df = pd.DataFrame(data, columns=['Result', 'Average Hustle Index'])

# Plot the table
fig, ax = plt.subplots()

# Hide axes
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Create table
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

# Display the plot
plt.show()

#%%
# Creating dataframe for logistic regression
df_log = copy.deepcopy(df_hustle)
df_log = df_log.reset_index()
#%%
#Populating
df_log.drop('index', axis=1)
df_log = df_log[['PTS', 'CONTESTED_SHOTS', 'DEFLECTIONS', 'CHARGES_DRAWN', 'SCREEN_ASSISTS', 'LOOSE_BALLS_RECOVERED', 'BOX_OUTS']]
df_log['Win/Loss'] = np.nan

for i in range(0, len(df_log), 2):
    if df_log.loc[i, 'PTS'] > df_log.loc[i+1, 'PTS']:
        df_log.loc[i, 'Win/Loss'] = 'Win'
        df_log.loc[i+1, 'Win/Loss'] = 'Loss'
    else:
        df_log.loc[i+1, 'Win/Loss'] = 'Win'
        df_log.loc[i, 'Win/Loss'] = 'Loss'
#%%
# Populating Win/Loss with 0 or 1
for i in range(0, len(df_log)):
    if df_log.loc[i, 'Win/Loss'] == 'Loss':
        df_log.loc[i, 'Win/Loss'] = 0
    else:
        df_log.loc[i, 'Win/Loss'] = 1
#%%
#Changing dtype to int for sklearn
df_log['Win/Loss'].astype('int')
#%%
#Creating arrays for model
X = np.asarray(df_log[['CONTESTED_SHOTS', 'DEFLECTIONS', 'CHARGES_DRAWN', 'SCREEN_ASSISTS', 'LOOSE_BALLS_RECOVERED', 'BOX_OUTS']])
y = np.asarray(df_log['Win/Loss'])

print(X[0:5], y[0:5])

#%%
#Normalize data
X = preprocessing.StandardScaler().fit(X).transform(X)
#%%
#Making sure y is discrete
y = y.astype('int')
#%%
#Fitting logistic regression model
LR = LogisticRegression(C=0.1, solver='liblinear').fit(X,y)
#%%
coef = LR.coef_
print(coef)
#%%
#%%
#Making data table
data = [['Contested Shots', 0.12824338], ['Deflections', 0.1224204], ['Charges Drawn', 0.01873273], ['Screen Assists', 0.0819328], ['Loose Balls Recovered', 0.13469319], ['Box Outs', 0.20981812]]

# Convert to DataFrame for easy plotting
df = pd.DataFrame(data, columns=['Statistic Category', 'Coefficient Value'])

# Plot the table
fig, ax = plt.subplots()

# Hide axes
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False)
ax.set_frame_on(False)

# Create table
table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')

# Display the plot
plt.show()
#%%
#Creating a leaderboard
df_leaderboard = df_c.groupby('TEAM_NAME')['Hustle_total'].mean()
#%%
#Creating leaderboard visual

data = [[1, 'Warriors', 2.23472],[2, 'Thunder', 2.06218],[3, 'Kings', 1.98904],[4, 'Mavericks', 1.98901],[5, 'Knicks', 1.96664]]

rankings = [item[0] for item in data]
teams = [item[1] for item in data]
scores = [item[2] for item in data]

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.barh(teams, scores, color=colors)

for bar in bars:
    ax.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2, 
            f'{bar.get_width():.5f}', 
            va='center', ha='right', color='white', fontweight='bold')

ax.set_title('NBA Hustle Team Leaderboard', fontsize=16)
ax.set_xlabel('Average Hustle Index')
ax.set_ylabel('Top 5 Teams in the NBA')


ax.set_facecolor('#F0F0F0')  
plt.grid(axis='x', color='gray', linestyle='--', linewidth=0.5)

ax.invert_yaxis()

plt.show()

