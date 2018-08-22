from sklearn import datasets
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.decomposition import TruncatedSVD
from sklearn import neighbors
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
X = preprocessing.maxabs_scale(preprocessing.scale(digits.data))
Y = digits.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.33)

svd = TruncatedSVD()
dat = svd.fit_transform(X)

fig = plt.figure()
for i in range(10):
    temp = dat[Y == i]
    plt.scatter(temp[:, 0], temp[:, 1])
# plt.show()

knn = neighbors.KNeighborsClassifier()
par = {'n_neighbors': range(2, 20)}
score = model_selection.GridSearchCV(knn, param_grid=par, cv=10)

best_par = []
accura = []
num = 20
for i in range(1, num):
    sv = TruncatedSVD(n_components=i)
    data = sv.fit_transform(x_train)
    score.fit(data, y_train)
    best_par.append(score.best_params_['n_neighbors'])
    accura.append(metrics.accuracy_score(y_test, score.predict(sv.transform(x_test))))
    # print(metrics.accuracy_score(y_test, score.predict(sv.transform(x_test))))

fig2 = plt.figure()
ax = fig2.add_subplot(1, 1, 1)
plt.bar(range(1, num), accura)
ax.set_xticks(range(1, num))
plt.plot(range(1, num), preprocessing.minmax_scale(best_par), 'r')
ax.set_xlabel('dimensions')
ax.set_ylabel('accuracy')
plt.title('Inflect of dimensions')
plt.show()
