import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
# 定义地图对象

class Map:
    '''
    定义地图对象
    输入地图数据，起始点坐标，结束点坐标
    点的表示:
    含有两个元素的列表cor:[x,y]
    表示索引的值index:inde
    '''

    def __init__(self, map_data, start_position, end_position):
        self.data = map_data
        self.shape = self.data.shape
        # 读取地图数据 2维array 255表示能够通过 0表示不能通过
        self.start = start_position
        self.end = end_position

    def is_valid(self, position):
        # 判断点是否有效，是否在边界内
        if position[0] < 0 or position[0] >= self.data.shape[0] or position[1] < 0 or position[1] >= self.data.shape[1]:
            return False
        if self.data[position[0], position[1]] == 0:
            return False
        return True

    def plot_map(self, path=None):
        # 画出地图，如果规划的路径不为空的话，将其画出来
        # 注意地图中x与y是反过来的，横坐标是y
        sns.set(style='dark')
        plt.figure()
        plt.imshow(self.data)
        plt.xticks([])
        plt.yticks([])
        if path is not None:
            plt.plot(path[:, 1], path[:, 0], 'r')
            plt.scatter([self.start[1], self.end[1]],
                        [self.start[0], self.end[0]], s=40, marker='*', c='r')

    def index2cor(self, index):
        # 输入index转化为坐标
        return index//self.data.shape[1], index % self.data.shape[1]

    def cor2index(self, x, y):
        # 输入坐标转化为index
        return x*self.data.shape[1]+y

    def get_dis(self, obj1, obj2, kind='index'):
        # 输入两个点的索引或者坐标得到距离
        # kind = 'index' or 'cor'
        if kind == 'index':
            x1, y1 = self.index2cor(obj1)
            x2, y2 = self.index2cor(obj2)
        else:
            x1, y1 = obj1
            x2, y2 = obj2
        return ((x1-x2)**2+(y1-y2)**2)**0.5

    def feasible_points(self, obj, kind='index'):
        # 输入当前点得到周围可行点
        # kind = ‘index'时输入索引值，kind='cor'时输入坐标值
        if kind == 'index':
            x, y = self.index2cor(obj)
        else:
            x, y = obj
        results = []
        for i in range(3):
            for j in range(3):
                x_new, y_new = x-1+i, y-1+j
                if self.is_valid([x_new, y_new]) == False:
                    continue
                if kind == 'index':
                    index_new = self.cor2index(x_new, y_new)
                    results.append(index_new)
                else:
                    results.append([x_new,y_new])
        return results

    def plot_precess(self,filename,plot_numbers = 5000):
        '''
        画出寻路过程
        输入包含寻路过程的文件,一次性画图的点数量
        '''
        # 画出原始图
        fig = plt.figure()
        sns.set(style='dark')
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self.data)
        plt.xticks([])
        plt.yticks([])
        # 读取规划出来的路线
        path = np.load(filename)
        final_path, explore_path = path['final'], path['explore']   
        len_explore = len(explore_path)
        plot_numbers = 1000
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
    # 选择number m深度的海域进行测试
    number = 4650
    data_dir = os.path.join(sys.path[0], 'Data')
    map_data = np.load(os.path.join(data_dir, str(number) + 'm_small.npy'))
    map = Map(map_data, None, None)
    map.plot_map()
    plt.show()


if __name__ == '__main__':
    main()
