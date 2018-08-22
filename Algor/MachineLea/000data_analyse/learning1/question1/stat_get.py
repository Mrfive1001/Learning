import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read csv
Salary = pd.read_csv(
    r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\learning1\question1\lahman-csv_2014-02-14\Salaries.csv')
Team = pd.read_csv(
    r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\learning1\question1\lahman-csv_2014-02-14\Teams.csv')

# head of the table
print(Salary.columns)
print(Team.columns)

# 分组计算
salary = Salary.groupby([Salary.yearID, Salary.teamID], as_index=False).sum()
print(salary)

# 合并
mer = pd.merge(pd.DataFrame(salary), Team[['yearID', 'teamID', 'W']], how='inner', on=['yearID', 'teamID'])
print(mer.head(2))

# 画图

# fig = plt.figure()
# plt.scatter(mer.salary, mer.winRate)
# plt.show()
