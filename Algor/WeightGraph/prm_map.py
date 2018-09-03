import math
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from get_map import Map

np.random.seed(100)


class PRM:
    '''
    利用PRM算法将二维数据图像数据进行离散网格化
    输入：二维图像，起始点，目标点， 离散点的数量，半径大小
    最终一个图像可以用nodelist来表示，将二维图像变成了无向有权图
    '''

    def __init__(self, map_data, start_position, end_position, num_points=200,radius = 200):
        self.map = Map(map_data, start_position, end_position)
        self.num_points = num_points
        self.node_lists = []   # 最终有用的就是这个列表
        self.cor_lists = []
        self.get_nodes()
        self.get_neighbors(radius)

    def get_neighbors(self, radius):
        '''
        得到点周围一定范围的neighbors
        neighbors用dict来表示
        '''
        for index in range(self.num_points):
            dis = np.linalg.norm(
                self.cor_lists-self.cor_lists[index], axis=1)  # 找出距离最近的点
            indexs = list(np.where(dis <= radius)[0])
            result = []
            for index_temp in indexs:
                if self.cross_obstacle(self.node_lists[index], self.node_lists[index_temp]):
                    continue
                if index_temp == index:
                    continue
                result.append(index_temp)
            if result:
                self.node_lists[index].neighbors = {}
                for res in result:
                    self.node_lists[index].neighbors[res] = dis[res]

    def cross_obstacle(self, node1, node2):
        '''
        判断两节点中间的点是否经过障碍物
        输入两个点的对象
        '''
        x1, y1 = node1.x, node1.y
        x2, y2 = node2.x, node2.y
        theta = math.atan2(y2 - y1, x2 - x1)
        total = ((x1-x2)**2+(y1-y2)**2)**0.5
        for length in range(1, int(total)+1):
            newx = x1+round(length * math.cos(theta))
            newy = y1+round(length * math.sin(theta))
            if self.map.is_valid([newx, newy]) is False:
                return True
        return False

    def get_nodes(self):
        '''
        生成一定数量的有效点
        '''
        # 加入起点
        node = Node(*self.map.start)
        self.node_lists.append(node)
        self.cor_lists.append(self.map.start)
        for _ in range(self.num_points-2):
            while True:
                position = self.random_node()
                if self.map.is_valid(position):
                    break
            node = Node(*position)
            self.node_lists.append(node)
            self.cor_lists.append(position)
        # 加入终点
        node = Node(*self.map.end)
        self.node_lists.append(node)
        self.cor_lists.append(self.map.end)
        self.cor_lists = np.array(self.cor_lists)

    def random_node(self):
        '''
        在地图范围内生成一个随机点
        '''
        m, n = self.map.shape
        x = random.randint(0, m)
        y = random.randint(0, n)
        return [x, y]

    def plot(self, lines_plot=False):
        '''
        画出散点图
        是否画出点的连线
        '''
        self.map.plot_map()
        sns.set(style='dark')
        plt.scatter(self.cor_lists[:, 1], self.cor_lists[:, 0], s=10)
        plt.scatter([self.map.start[1], self.map.end[1]],
                        [self.map.start[0], self.map.end[0]], s=20, marker='*', c='r')
        if lines_plot:
            for node in self.node_lists:
                paths = []
                if node.neighbors:
                    for key in node.neighbors.keys():
                        neigh_node = self.node_lists[key]
                        paths.append([node.x, node.y])
                        paths.append([neigh_node.x, neigh_node.y])
                    paths = np.array(paths)
                    plt.plot(paths[:, 1], paths[:, 0], c='g')
                    plt.pause(0.0000001)

    def save(self, name):
        '''
        将nodelists保存到Data下的name文件里面
        '''
        dir = os.path.join(sys.path[0], 'Data')
        with open(os.path.join(dir,name),'wb') as f:
            pickle.dump(self.node_lists,f)


class Node:
    '''
    定义PRM所需要的点
    '''

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.neighbors = None


def main():
    # 1 读取数据
    number = 4650
    data_dir = os.path.join(sys.path[0], 'Data')
    map_data = np.load(os.path.join(data_dir, str(number) + 'm_small.npy'))

    # 2 定义起点终点，然后生成图
    read_position = [[500, 500, 200, 600], [1100, 460, 1150, 360], [500, 500, 500, 2500],
                     [2355, 2430, 2000, 4000], [1140, 1870, 820, 3200], [1500, 20, 2355, 2430]]
    # 起点终点备选
    read =  5 # 规划数据，选择对那一组测试
    start_point = read_position[read][: 2]
    end_point = read_position[read][2:]

    
    prm = PRM(map_data, start_point, end_point, num_points=500,radius=240)
    prm.save('graph1.pk')
    # print(prm.node_lists)
    # prm.plot(lines_plot=True)
    # plt.show()


if __name__ == '__main__':
    main()
