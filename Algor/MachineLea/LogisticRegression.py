import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class LogisticReg:
    def __init__(self, method='linearReg'):
        self.X = None
        self.Y = None
        self.W = None
        self.method = method

    def binary(self, Y):
        threshold = 0.5
        Y[Y <= threshold] = 0
        Y[Y > threshold] = 1
        return Y

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        if self.method == 'linearReg':
            from LinearRegression import LinearReg
            lr = LinearReg('iter')
            lr.fit(X, Y)
            self.W = lr.W
        elif self.method == 'logisticReg':
            iter_max = 10000
            alpha = 0.00002
            tol = 0.0001
            X = np.hstack((self.X, np.ones((len(self.X), 1))))
            X_T = np.transpose(X)
            self.W = np.zeros(self.X.shape[1] + 1)
            for i in range(iter_max):
                Y_new = 1 / (1 + np.exp(-np.dot(X, self.W)))
                delta = alpha * np.dot(X_T, self.Y - Y_new)
                if np.linalg.norm(delta) < tol:
                    print('Train breaks early in %d iterations.' % (i + 1))
                    print('W = ', self.W)
                    break
                self.W += delta

    def predict(self, X):
        if self.method == 'linearReg':
            Y_pre = np.dot(np.hstack((X, np.ones((len(X), 1)))), self.W)
            Y_pre = self.binary(Y_pre)
        elif self.method == 'logisticReg':
            Y_pre = 1 / (1 + np.exp(-np.dot(np.hstack((X, np.ones((len(X), 1)))), self.W)))
            Y_pre = self.binary(Y_pre)
        return Y_pre

    def plot2d(self):
        plot_X = np.linspace(2, 5)
        plot_Y = (0.5 - self.W[-1] - self.W[0] * plot_X) / self.W[1]
        return plot_X, plot_Y


def create_data():
    angles = np.linspace(0, 2 * math.pi, 200).reshape(-1, 1)
    random_scale = 1
    centers = [2, 4]
    X1 = np.hstack((np.cos(angles) + centers[0], np.sin(angles) + centers[1])) + \
         random_scale * np.random.rand(angles.shape[0], 2)
    X2 = np.hstack((np.cos(angles) + centers[1], np.sin(angles) + centers[0])) + \
         random_scale * np.random.rand(angles.shape[0], 2)
    Y1 = np.ones(angles.shape[0])
    Y2 = np.zeros(angles.shape[0])
    X = np.vstack((X1, X2))
    Y = np.hstack((Y1, Y2))
    return X, Y


def main():
    X, Y = create_data()
    lgr = LogisticReg(method='logisticReg')
    lgr.fit(X, Y)
    Y_pre = lgr.predict(X)
    plot_X, plot_Y = lgr.plot2d()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(2, 1, 1)
    ax1.scatter(X[Y == 0, 0], X[Y == 0, 1])
    ax1.scatter(X[Y == 1, 0], X[Y == 1, 1])
    ax2 = fig1.add_subplot(2, 1, 2)
    ax2.scatter(X[Y_pre == 0, 0], X[Y_pre == 0, 1])
    ax2.scatter(X[Y_pre == 1, 0], X[Y_pre == 1, 1])
    ax2.plot(plot_X, plot_Y)
    print(classification_report(Y, Y_pre))
    plt.show()


if __name__ == '__main__':
    main()
