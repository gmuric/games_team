# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:00:16 2020

@author: liu
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from matplotlib import pyplot as plt

"""
find team performance vs the the total number of change player
use the average score in every match represents team performance
more total player number means team change players more often
"""

df = pd.read_csv('F:\grid\data\\matches.csv')
df = df[['Team 1 ID', 'Team 2 ID','Team 1 Score', 'Team 2 Score']]
df1 = df[['Team 1 ID', 'Team 1 Score']].groupby('Team 1 ID').mean().reset_index()
df2 = df[['Team 2 ID', 'Team 2 Score']].groupby('Team 2 ID').mean().reset_index()
df2 = df2.rename(columns={'Team 2 ID':'Team 1 ID', 'Team 2 Score':'Team 1 Score'})
dfscore = pd.concat([df1,df2])
dfscore = dfscore.groupby('Team 1 ID').mean().reset_index()
dfscore = dfscore.rename(columns={'Team 1 ID':'team_id', 'Team 1 Score':'average score'})
dfplayer = pd.read_csv('F:\grid\data\player_team_match.csv')
ulist = dfplayer.groupby('team_id')['player_id'].nunique()
ulist = ulist[(np.abs(stats.zscore(ulist)) < 3)].rename('player_num').reset_index()
df=pd.merge(dfscore, ulist, on='team_id',how='inner').sort_values(by='player_num')
average = df[['player_num', 'average score']].groupby('player_num').mean()
fig, ax = plt.subplots()
ax.plot(average)
ax.set_xlabel('number of players')
ax.set_ylabel('average score')
plt.savefig('player-score.png')
plt.show()


"""
find team performance vs the frequence of change player
use the average score in every match represents team performance
the change player frequency is derived by the total player number / how many year the team exists
"""

df = pd.read_csv('F:\grid\data\\matches.csv')
df = df[['Team 1 ID','Team 2 ID','Team 1 Score', 'Team 2 Score','Date']]
pd.to_datetime(df['Date'])
df['year'] = pd.DatetimeIndex(df['Date']).year
df1 = df[['Team 1 ID', 'Team 1 Score', 'year']]
df1 = df1.rename(columns={'Team 1 ID':'team_id', 'Team 1 Score':'Team Score'})
df2 = df[['Team 2 ID', 'Team 2 Score', 'year']]
df2 = df2.rename(columns={'Team 2 ID':'team_id', 'Team 2 Score':'Team Score'})
df = pd.concat([df1,df2])
duration = df.groupby('team_id')['year'].nunique().reset_index()
score = df.groupby(['team_id'])['Team Score'].mean().reset_index()



dfplayer = pd.read_csv('F:\grid\data\player_team_match.csv')
ulist = dfplayer.groupby(['team_id'])['player_id'].nunique().reset_index()
ulist = ulist.rename(columns={'player_id':'player_num'})
ulist = ulist[(np.abs(stats.zscore(ulist['player_num'])) < 2)]
df=pd.merge(score, ulist, on=['team_id'],how='inner').sort_values(by='team_id')
df= pd.merge(df, duration, on='team_id', how='inner')
average = pd.DataFrame(df['Team Score'])
average['player num'] = df['player_num']/df['year']
average = average.groupby(['player num']).mean().reset_index()
import seaborn as sns
sns.lmplot(x ='player num', y ='Team Score', data = average)
from scipy import stats
X = average['player num']
Y = average['Team Score']
X = sm.add_constant(X) # adding a constant
model = sm.OLS(Y, X).fit()
print_model = model.summary()

#




#
