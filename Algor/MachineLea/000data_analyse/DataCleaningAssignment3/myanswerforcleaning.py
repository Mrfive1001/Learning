import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

energy = pd.read_excel(r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\DataCleaning\Energy Indicators.xls',
                       skiprows=range(16), skip_footer=1,
                       names=['h', 'hh', 'Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable'])
energy = energy.iloc[1:228, 2:]
energy[energy == '...'] = np.nan
energy['Energy Supply'] *= 1000000
ScimEn = pd.read_excel(r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\DataCleaning\scimagojr-3.xlsx')
GDP = pd.read_csv(r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\DataCleaning\world_bank.csv',
                  skiprows=range(4), index_col=0)
GDP = GDP.iloc[:, -10:].copy()
GDP['Country'] = GDP.index
# 上面是读入数据

new_energy = energy['Country'].values.copy()
r1 = re.compile('\s*\([^\(\)]*\)')
r2 = re.compile('\d+')
for i in range(len(new_energy)):
    new_energy[i] = r1.sub('', new_energy[i])
    new_energy[i] = r2.sub('', new_energy[i])
energy['Country'] = new_energy

worddic = {"Republic of Korea": "South Korea",
           "United States of America": "United States",
           "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
           "China, Hong Kong Special Administrative Region": "Hong Kong"}
for ke, val in worddic.items():
    s = (energy['Country'] == ke)
    energy['Country'][energy['Country'] == ke] = val

# 更改国家名称
worddic = {"Korea, Rep.": "South Korea",
           "Iran, Islamic Rep.": "Iran",
           "Hong Kong SAR, China": "Hong Kong"}
for ke, val in worddic.items():
    GDP['Country'][GDP['Country'] == ke] = val
# 更改国家名称


temp1 = pd.merge(ScimEn, energy, on='Country')
result1 = pd.merge(temp1, GDP, on='Country')
result1.index = result1['Country'].tolist()
result = result1.drop('Country', axis=1)
result1 = result.iloc[:15, :]
print(result1)

result2 = len(ScimEn.index) - len(result.index)
print(result2)

result3 = result1.iloc[:, -10:].mean(axis=1, skipna=True)
result3 = result3.sort_values(ascending=False)
print(result3)

old, new = result1.loc[result3.index[6 - 1], '2006'], result1.loc[result3.index[6 - 1], '2015']
result4 = (new - old) / old
print(result4)

result5 = result1['Energy Supply per Capita'].mean()
print(result5)

temp = result1['% Renewable']
temp = temp.sort_values()
result6 = (temp.index[-1], temp.values[-1])
print(result6)

result8 = result1.copy()
result8['Population'] = result8['Energy Supply'] // result8['Energy Supply per Capita']
result8_tem = result8.Population.sort_values().index[-1]
print(result8_tem)

result9 = result8.copy()
result9['Citable documents per capita'] = (result9['Citable documents'] / result9['Population']).astype(np.float64)
result9['Energy Supply per Capita'] = result9['Energy Supply per Capita'].astype(np.float64)
result9_tem = result9['Citable documents per capita'].corr(result9['Energy Supply per Capita'])
# 计算协方差出错，不知道为啥, 转化数据类型
Top15 = result9.copy()
Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
plt.scatter(Top15['Citable docs per Capita'].values, Top15['Energy Supply per Capita'].values)
plt.xlim(0, 0.0006)
plt.xlabel('Citable docs per Capita')
plt.ylabel('Energy Supply per Capita')
plt.show()

result10 = result1.copy()
result10['high'] = 0
result10['high'][result10['% Renewable'] > result1['% Renewable'].median()] = 1
ob = result10['% Renewable'].rank()
ob.name = 'HighRenew'

ContinentDict = {'China': 'Asia',
                 'United States': 'North America',
                 'Japan': 'Asia',
                 'United Kingdom': 'Europe',
                 'Russian Federation': 'Europe',
                 'Canada': 'North America',
                 'Germany': 'Europe',
                 'India': 'Asia',
                 'France': 'Europe',
                 'South Korea': 'Asia',
                 'Italy': 'Europe',
                 'Spain': 'Europe',
                 'Iran': 'Asia',
                 'Australia': 'Australia',
                 'Brazil': 'South America'}
result11 = result9.copy()
result11['Continent'] = ContinentDict.values()
result11['Population'] = result11['Population'].astype(np.float64)
group = result11['Population'].groupby(result11.Continent)
dict_temp = {'size': group.size(), 'sum': group.sum(), 'mean': group.mean(), 'std': group.std()}
conti = pd.DataFrame(dict_temp)
print(conti)

result12 = result11[['% Renewable', 'Continent']].copy()
result12['temp'] = 1
result12_temp = result12['temp'].groupby([result12['Continent'], pd.qcut(result12['% Renewable'], 5)]).size()
print(result12_temp)

result13 = result9['Population'].copy()


def solve(x):  # x:int
    a = (str(x))[::-1]
    for i in range(1, 4):
        temp = i * 4 - 1
        if temp < len(a):
            a = a[:temp] + ',' + a[temp:]
        else:
            break
    return a[::-1]


result13 = result13.apply(solve)
print(result13)

# 14题没做
