import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

countries = pd.read_csv(r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\learning1\question2\countries.csv')
income = pd.read_excel(
    r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\learning1\question2\GDPpercapitaconstant2000US.xlsx',
    index_col=0)
income_drop = (income.T).dropna(how='all', axis=1)
income_drop.sort_values(by='2011', axis=1, ascending=False, inplace=True)
print(income_drop.columns)

fig1 = plt.figure()
for colo in income_drop.columns[:10]:
    plt.plot(income_drop.index, income_drop[colo], label=colo)
plt.legend()
plt.show()

income = income.T


def fun(yea):
    left = pd.DataFrame(income.loc[str(yea)])
    result = pd.merge(pd.DataFrame(left), countries, left_index=True, right_index=True)
    result.columns = ['Income', 'Region']
    return result
