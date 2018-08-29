import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class TSP:
    '''
    定义TSP问题
    输入数据种类: small middle big
    '''

    def __init__(self, data_type='small'):
        self.dir = sys.path[0]
        self.cities = np.load(os.path.join(
            self.dir, 'Data\%s_tsp.npy' % data_type))
        print(self.cities)

    def plot_map(self, path=None):
        sns.set(style='dark')
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(self.cities[:, 0], self.cities[:, 1])


def main():
    my_tsp = TSP('middle')
    my_tsp.plot_map()
    plt.show()


if __name__ == '__main__':
    main()
