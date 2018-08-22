import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import svm
from sklearn import model_selection

data_train = pd.read_csv(r"C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\001Titanic\train.csv")

data_test = pd.read_csv(r'C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\001Titanic\test.csv')


# 处理缺失数据age和cabin
def solve(data):
    new_data = (data[['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin']]).copy()
    new_data.Cabin[pd.notnull(data.Cabin)] = 1
    new_data.Cabin[pd.isnull(data.Cabin)] = 0
    new_data.Sex[new_data.Sex == 'male'] = 1
    new_data.Sex[new_data.Sex == 'female'] = 0
    return new_data


x_train = solve(data_train).values
x_test = solve(data_test).values
y_train = data_train['Survived'].values
lg = linear_model.LogisticRegression()
lg.fit(x_train, y_train)


def piline(clf):
    print(np.mean(model_selection.cross_val_score(clf, x_train, y_train, cv=10)))
    clf.fit(x_train, y_train)
    print(clf.score(x_train, y_train))


piline(lg)

result = pd.DataFrame(
    {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': lg.predict(x_test).astype(np.int32)})
result.to_csv(
    r"C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\001Titanic\simple_predict.csv",
    index=False)

svc = svm.SVC()
svc.fit(x_train, y_train)
y_test = svc.predict(x_test)
result = pd.DataFrame(
    {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': y_test.astype(np.int32)})
result.to_csv(
    r"C:\Users\MrFive1001\Documents\PycharmProjects\Kaggle\001Titanic\svm_predict.csv",
    index=False)
piline(svc)

# from sklearn.ensemble import RandomForestClassifier
#
# random_forest = RandomForestClassifier(n_estimators=500)
# piline(random_forest)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
piline(knn)