import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from aco import get_data
from get_map import Map
from rrt import RRT

class RrtGa:
    '''
    结合RRT和GA算法来解决路径规划问题
    输入图，遗传算法的一些参数
    '''
    def __init__(self,my_map):
        self.map = my_map
        self.pop_num = 20   # 种群数量
        self.rrt = RRT(self.map)
        self.mutate_rate = 0.0001 # 突变概率
        self.iterations = 100
        self.cross_rate = 0.9


    def init_pop(self):
        paths = []
        for _ in range(self.pop_num):
            path = self.rrt.find_path()
            paths.append(path)
        return paths

    def find_path(self):
        self.pop = self.init_pop() # 生成初始种群
        self.fits = [self.map.get_fitness(path) for path in self.pop] # 计算初始种群的fitness值
        for iteration in range(self.iterations):
            print(self.fits)
        

def main():
    # 1 生成数据
    map_data = get_data(1000, 1000, 0.01)
    # 2 定义起始点和目标点生成图
    start_point = [86, 870]
    end_point = [849, 324]
    my_map = Map(map_data, start_point, end_point)

    # 读取大文件数据
    # 1 读取数据
    number = 4650
    data_dir = os.path.join(sys.path[0], 'Data')
    map_data = np.load(os.path.join(data_dir, str(number) + 'm_small.npy'))

    # 2 定义起点终点，然后生成图
    read_position = [[500, 500, 200, 600], [1100, 460, 1150, 360], [500, 500, 500, 2500],
                     [2355, 2430, 2000, 4000], [1140, 1870, 820, 3200], [1500, 20, 2355, 2430]]
    # 起点终点备选
    read = 0 # 规划数据，选择对那一组测试
    start_position = read_position[read][: 2]
    end_position = read_position[read][2:]
    my_map = Map(map_data, start_position, end_position)

    # 3 定义算法
    rrtga = RrtGa(my_map)
    # 4 运行和显示结果
    rrtga.find_path()


if __name__ == '__main__':
    main()
