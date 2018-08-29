import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class TSP:
    '''
    定义TSP问题
    输入数据种类: small(52) middle(194) big(734)
    '''
    def __init__(self, data_type='small'):
        self.dir = sys.path[0]
        self.cities = np.load(os.path.join(
            self.dir, 'Data\%s_tsp.npy' % data_type))

    def plot_map(self, path=None):
        '''
        画出所有点，以及图像
        '''
        sns.set()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(self.cities[:, 0], self.cities[:, 1])
        if path is not None:
            path = np.array(path)
            if path.ndim == 1:
                path = self.cities[path]
            ax.plot(path[:,0],path[:,1])



def main():
    my_tsp = TSP('middle')
    my_tsp.plot_map()
    plt.show()


if __name__ == '__main__':
    main()
