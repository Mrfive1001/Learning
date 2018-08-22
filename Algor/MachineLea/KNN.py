import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def create_data():
    angles = np.linspace(0, 2 * math.pi, 50).reshape(-1, 1)
    R1 = 1
    R2 = 2
    X1 = np.hstack((R1 * np.cos(angles), R1 * np.sin(angles))) + \
         R1 / 10.0 * np.random.rand(angles.shape[0], 2)
    X2 = np.hstack((R2 * np.cos(angles), R2 * np.sin(angles))) + \
         R2 / 10.0 * np.random.rand(angles.shape[0], 2)
    Y1 = np.ones(angles.shape[0])
    Y2 = np.zeros(angles.shape[0])
    X = np.vstack((X1, X2))
    Y = np.hstack((Y1, Y2))
    return X, Y


class KNN:
    def __init__(self, k=5):
        self.X = None
        self.Y = None
        self.k = k

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, X):
        # verify the effectiveness of input data
        try:
            if X.shape[1] != self.X.shape[1]:
                raise ValueError
        except Exception:
            print('You\'ve input wrong data shape!')

        result_Y = []
        for index in range(len(X)):
            X_temp = X[index:index + 1, :]
            X_temp = np.dot(np.ones((len(self.X), 1)), X_temp)
            distance = np.linalg.norm(X_temp - self.X, ord=2, axis=1)
            result = self.Y[np.argsort(distance)[:self.k]]
            result = np.argmax(np.bincount(result.astype('int64')))
            result_Y.append(result)
        return np.array(result_Y)


def main():
    X, Y = create_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    # 定义KNN实例
    knn = KNN()
    knn.fit(X, Y)
    Y_pre = knn.predict(X)
    print(classification_report(Y, Y_pre))

    # 画图
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.scatter(X[Y == 0, 0], X[Y == 0, 1])
    ax1.scatter(X[Y == 1, 0], X[Y == 1, 1])
    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.scatter(X[Y_pre == 0, 0], X[Y_pre == 0, 1])
    ax2.scatter(X[Y_pre == 1, 0], X[Y_pre == 1, 1])

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(2, 1, 1)
    ax1.scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1])
    ax1.scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1])
    ax2 = fig2.add_subplot(2, 1, 2)
    ax2.scatter(X_test[Y_test == 0, 0], X_test[Y_test == 0, 1])
    ax2.scatter(X_test[Y_test == 1, 0], X_test[Y_test == 1, 1])

    plt.show()


if __name__ == '__main__':
    main()
