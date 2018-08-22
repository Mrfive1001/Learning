import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
import seaborn as sns
'''
知识点：
动态画图
散点图的设置
'''


def plot_path():
    '''
    将规划的地图进行动态显示
    '''
    # 读取地图
    number = 4650
    data_dir = os.path.join(sys.path[0], 'Data')
    map_data = np.load(os.path.join(data_dir, str(number) + 'm_small.npy'))
    # 读取规划出来的路线
    dir = sys.path[0]
    path = np.load(os.path.join(dir, 'result.npz'))
    final_path, explore_path = path['final'], path['explore']
    # 画图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(map_data)
    len_explore = len(explore_path)
    plot_numbers = 100000
    # 每隔一段时间画出一些点的散点图
    for i in range(int(len_explore / plot_numbers)+1):
        ax.scatter(explore_path[plot_numbers * i:plot_numbers * (i + 1), 1],
                   explore_path[plot_numbers * i:plot_numbers * (i + 1), 0], s=2, c='g', alpha=0.5)
        if i % 5 == 0 or i == len_explore:
            # 画出起点终点
            ax.scatter(final_path[0, 1], final_path[0, 0],
                       s=40, marker='*', c='r')
            ax.scatter(final_path[-1, 1],
                       final_path[-1, 0], s=40, marker='*', c='r')
        plt.pause(0.0000001)
    # 画出最终路径
    ax.plot(final_path[1:-1, 1], final_path[1:-1, 0], 'r')
    plt.show()


def main():
    sns.set()
    plot_path()


if __name__ == '__main__':
    main()
