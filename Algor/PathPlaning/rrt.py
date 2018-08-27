import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from get_map import Map
from aco import get_data
sns.set(style='dark')
np.random.seed(200)

class RRT:
    '''
    定义RRT算法来解决路径规划问题
    输入图，包含起点和终点
    '''    
    def __init__(self, my_map):
        self.map = my_map
        self.expandDis = 1.0
        self.goalSampleRate = 0.05  # 选择终点的概率是0.05
        self.nodeList = [Node(*self.map.start)]  # 定义节点的列表
        self.nodeCorList = np.array([self.map.start])   # 保存所有节点的坐标

    def find_path(self):
        while True:
            if random.random() > self.goalSampleRate:
                rnd = self.random_node()
            else:
                rnd = self.map.end

class Node:
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

def main():
    # 1 生成数据
    map_data = get_data(40, 40, 0.1)
    # 2 定义起始点和目标点生成图
    start_point = [0, 0]
    end_point = [38, 34]
    my_map = Map(map_data, start_point, end_point)

    '''
    # # 读取大文件数据
    # # 1 读取数据
    # number = 4650
    # data_dir = os.path.join(sys.path[0], 'Data')
    # map_data = np.load(os.path.join(data_dir, str(number) + 'm_small.npy'))

    # # 2 定义起点终点，然后生成图
    # read_position = [[500, 500, 200, 600], [1100, 460, 1150, 360], [500, 500, 500, 2500],
    #                  [2355, 2430, 2000, 4000], [1140, 1870, 820, 3200], [1500, 20, 2355, 2430]]
    # # 起点终点备选
    # read = 0  # 规划数据，选择对那一组测试
    # start_position = read_position[read][: 2]
    # end_position = read_position[read][2:]
    # my_map = Map(map_data, start_position, end_position)
    '''

    # 3 定义算法
    rrt = RRT(my_map)
    # 4 运行和显示结果

    plt.show()


if __name__ == '__main__':
    main()
