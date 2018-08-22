from sklearn import datasets
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn import neighbors
import pandas as pd
from sklearn import metrics

X, Y = datasets.load_iris().data, datasets.load_iris().target
print(X.shape, Y.shape)

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3)

svd = TruncatedSVD()
dat = svd.fit_transform(X)

fig = plt.figure()
for i, color in zip((0, 1, 2), ('r', 'g', 'y')):
    dat_temp = dat[Y == i]
    plt.scatter(dat_temp[:, 0], dat_temp[:, 1], color=color)

par = {'n_neighbors': range(1, 30)}
knn = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), param_grid=par)
knn.fit(x_train, y_train)
print(knn.best_estimator_)
print(knn.cv_results_['rank_test_score'])
re = pd.DataFrame(knn.cv_results_).loc[:, ['param_n_neighbors', 'std_test_score', 'std_train_score']]
fig2 = plt.figure()
plt.scatter(re['param_n_neighbors'], re['std_test_score'])
plt.scatter(re['param_n_neighbors'], re['std_train_score'], color='r')
print(metrics.accuracy_score(y_test, knn.predict(x_test)))
plt.show()
