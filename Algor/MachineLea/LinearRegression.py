import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def create_data():
    X1 = np.linspace(0, 10, 200)
    X2 = X1.copy() + 5
    Y = 10 * X1 + 2 * X2 + np.random.rand(len(X1)) + 10
    X = np.transpose(np.vstack((X1, X2)))
    return X, Y


class LinearReg:
    def __init__(self, method='ols'):
        self.X = None
        self.Y = None
        self.W = None
        self.method = method

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.W = np.zeros(self.X.shape[1] + 1)
        X = np.hstack((self.X, np.ones((len(self.X), 1))))
        X_T = np.transpose(X)
        if self.method == 'ols':
            # 最小二乘法
            self.W = np.dot(np.dot(np.linalg.inv(np.dot(X_T, X)), X_T), self.Y)
        elif self.method == 'iter':
            # 迭代法
            iter_max = 10000
            alpha = 0.00002
            tol = 0.0001
            for i in range(iter_max):
                Y_new = np.dot(X, self.W)
                delta = alpha * np.dot(X_T, self.Y - Y_new)
                if np.linalg.norm(delta) < tol:
                    print('Train breaks early in %d iterations.' % (i + 1))
                    print('W = ', self.W)
                    break
                self.W += delta

    def predict(self, X):
        return np.dot(np.hstack((X, np.ones((len(X), 1)))), self.W)


def main():
    X, Y = create_data()
    # X = np.arange(-4, 4, 0.1)
    # Y = np.arange(-4, 4, 0.1)
    # X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X**2 + Y**2)
    # Z = np.sin(R)
    # ax.plot_surface(X,Y,Z,cmap='rainbow')
    lr = LinearReg('iter')
    lr.fit(X, Y)
    Y_pre = lr.predict(X)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X[:, 0], X[:, 1], Y)
    ax.scatter(X[:, 0], X[:, 1], Y_pre)
    plt.show()


if __name__ == '__main__':
    main()
