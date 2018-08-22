import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
import numpy as np

teams = pd.read_csv(r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\learning3\lahman-csv_2014-02-14\Teams.csv')
players = pd.read_csv(
    r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\learning3\lahman-csv_2014-02-14\Batting.csv')
salaries = pd.read_csv(
    r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\learning3\lahman-csv_2014-02-14\Salaries.csv')
fielding = pd.read_csv(
    r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\learning3\lahman-csv_2014-02-14\Fielding.csv')
master = pd.read_csv(r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\learning3\lahman-csv_2014-02-14\Master.csv')

sa = pd.DataFrame(salaries.groupby([salaries.playerID])['salary'].median())
nam = master.loc[:, ['playerID', 'nameLast', 'nameGiven']].copy()
median_salary = pd.merge(nam, sa, left_on='playerID', right_index=True)
print(median_salary.head())

subTeams = teams[(teams.G == 162) & (teams.yearID > 1947)].copy()
subTeams["1B"] = subTeams.H - subTeams["2B"] - subTeams["3B"] - subTeams["HR"]
subTeams["PA"] = subTeams.BB + subTeams.AB
sta = ["1B", "2B", "3B", "HR", "BB"]
for col in sta:
    subTeams[col] = subTeams[col] / subTeams.PA
stats = subTeams[["teamID", "yearID", "W", "1B", "2B", "3B", "HR", "BB"]].copy()
print(stats.head())

te = stats.groupby([stats.yearID])['2B'].mean()
plt.plot(te.index, te.values)

for col in sta:
    stats[col] = preprocessing.scale(stats[col])
    stats[col] = preprocessing.maxabs_scale(stats[col])

train = stats[stats.yearID < 2002]
test = stats[stats.yearID >= 2002]
x_train = train.loc[:, sta].values
y_train = train.loc[:, 'W'].values
x_test = test.loc[:, sta].values
y_test = test.loc[:, 'W'].values

liner = linear_model.LinearRegression()
liner.fit(x_train, y_train)
print(np.mean((y_test - liner.predict(x_test)) ** 2))
