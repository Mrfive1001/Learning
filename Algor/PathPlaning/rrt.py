import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from get_map import Map
from aco import get_data
import math
import time
import copy
sns.set(style='dark')
np.random.seed(200)


class RRT:
    '''
    定义RRT算法来解决路径规划问题
    输入图，包含起点和终点
    '''

    def __init__(self, my_map):
        self.map = my_map
        self.expandDis = 10
        self.goalSampleRate = 0.2  # 选择终点的概率是0.05
        self.nodeList = [Node(*self.map.start)]  # 定义节点的列表
        self.nodeCorList = np.array([self.map.start])   # 保存所有节点的坐标

    def find_path(self):
        while True:
            # 产生随机顶点
            if random.random() > self.goalSampleRate:
                rnd = self.random_node()
            else:
                rnd = self.map.end
            # 找到最近节点
            min_index = self.get_nearest_index(rnd)

            # 扩展随机树
            nearest_node = self.nodeList[min_index]

            # 返回弧度制
            theta = math.atan2(rnd[1] - nearest_node.y,
                               rnd[0] - nearest_node.x)

            new_node = copy.deepcopy(nearest_node)
            new_node.x += round(self.expandDis * math.cos(theta))
            new_node.y += round(self.expandDis * math.sin(theta))
            new_node.parent = min_index

            # 判断新节点是否有效
            if self.cross_obstacle(nearest_node, theta):
                continue

            # 添加节点
            self.nodeList.append(new_node)
            self.nodeCorList = np.vstack((self.nodeCorList,np.array([[new_node.x, new_node.y]])))

            # 检查是否到达目标
            dx = new_node.x - self.map.end[0]
            dy = new_node.y - self.map.end[1]
            d = math.sqrt(dx**2+dy**2)
            if d <= self.expandDis:
                print("Goal!!")
                break
    
    def plot_final(self):
        '''
        将最终得到的路径进行显示
        返回最终长度
        '''
        curnode = self.nodeList[-1]
        paths = []
        length = 0
        while curnode.parent is not None:
            length += self.expandDis
            paths.append([curnode.x,curnode.y])
            curnode = self.nodeList[int(curnode.parent)]
        self.map.plot_map(paths)
        return length


    def random_node(self):
        '''
        在地图范围内生成一个随机点
        '''
        m, n = self.map.shape
        x = random.randint(0, m)
        y = random.randint(0, n)
        return [x, y]

    def get_nearest_index(self, rnd):
        '''
        返回离当前点最近的节点
        '''
        dis = np.linalg.norm(self.nodeCorList-rnd, axis=1)
        return np.argmin(dis)

    def cross_obstacle(self, node, theta):
        '''
        判断两节点和角度中间的点是否经过障碍物
        '''
        x, y = node.x, node.y
        for length in range(1,self.expandDis+1):
            newx = x+round(length * math.cos(theta))
            newy = y+round(length * math.sin(theta))
            if self.map.is_valid([newx,newy]) is False:
                return True
        return False


class Node:
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


def main():
    # # 1 生成数据
    # map_data = get_data(400, 400, 0.05)
    # # 2 定义起始点和目标点生成图
    # start_point = [0, 0]
    # end_point = [307, 324]
    # my_map = Map(map_data, start_point, end_point)


    # 读取大文件数据
    # 1 读取数据
    number = 4650
    data_dir = os.path.join(sys.path[0], 'Data')
    map_data = np.load(os.path.join(data_dir, str(number) + 'm_small.npy'))

    # 2 定义起点终点，然后生成图
    read_position = [[500, 500, 200, 600], [1100, 460, 1150, 360], [500, 500, 500, 2500],
                     [2355, 2430, 2000, 4000], [1140, 1870, 820, 3200], [1500, 20, 2355, 2430]]
    # 起点终点备选
    read = 5  # 规划数据，选择对那一组测试
    start_position = read_position[read][: 2]
    end_position = read_position[read][2:]
    my_map = Map(map_data, start_position, end_position)



    # 3 定义算法
    time0 = time.time()
    rrt = RRT(my_map)
    # 4 运行和显示结果
    rrt.find_path()
    print(time.time()-time0)
    print(rrt.plot_final())
    plt.show()


if __name__ == '__main__':
    main()
