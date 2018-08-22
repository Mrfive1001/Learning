import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimSun', 'SimHei', 'SimKai', 'PingFang SC']
plt.rcParams['font.serif'] = ['SimSun', 'SimHei', 'SimKai', 'PingFang SC']
plt.rcParams['font.family'] = 'sans-serif'

data_train = pd.read_csv(r"C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\001Titanic\train.csv")
fig = plt.figure(figsize=(9, 6))
fig.set(alpha=0.2)

plt.subplot2grid((1, 2), (0, 0))
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel('年龄')
plt.ylabel('密度')
plt.title('年龄和舱位与存活的关系')
plt.legend(('头等舱', '二等舱', '三等舱'))
plt.show()

# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()  # value_counts返回一个Series对象，值为计数值，索引为值
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u"各乘客等级的获救情况")
# plt.xlabel(u"乘客等级")
# plt.ylabel(u"人数")
#
# Survived_s0 = data_train.Sex[data_train.Survived == 0].value_counts()  # value_counts返回一个Series对象，值为计数值，索引为值
# Survived_s1 = data_train.Sex[data_train.Survived == 1].value_counts()
# df1 = pd.DataFrame({u'获救': Survived_s1, u'未获救': Survived_s0})
# df1.plot(kind='bar', stacked=True)
# plt.title(u"不同性别获救情况")
# plt.xlabel(u"乘客性别")
# plt.ylabel(u"人数")
#
# Survived_e0 = data_train.Embarked[data_train.Survived == 0].value_counts()  # value_counts返回一个Series对象，值为计数值，索引为值
# Survived_e1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df1 = pd.DataFrame({u'获救': Survived_e1, u'未获救': Survived_e0})
# df1.plot(kind='bar', stacked=True)
# plt.title(u"不同上船地区获救情况")
# plt.xlabel(u"上船地点")
# plt.ylabel(u"人数")
# plt.show()
