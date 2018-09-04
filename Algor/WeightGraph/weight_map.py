import math
import os
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from get_map import Map
from prm_map import Node


class WeightGraph:
    '''
    输入数据
    数据格式: 
    总体是list类型,每个元素是一个Node类型
    Node 包含以下因素:
    x,y,neighbors(None 或者 dict 表示下个节点的索引和其对应的距离)
    '''

    def __init__(self, node_lists=None,data_resoure = 1):
        if node_lists:
            self.node_lists= node_lists
        else:
            self.node_lists = self.load_data('graph%d.pk'%data_resoure)

    def load_data(self, name):
        '''
        读取图的文件名称
        '''
        number = 4650
        data_dir = os.path.join(sys.path[0], 'Data')
        map_data = np.load(os.path.join(data_dir, str(number) + 'm_small.npy'))
        self.map = Map(map_data, None, None)

        with open(os.path.join(data_dir, name), 'rb') as f:
            nodes = pickle.load(f)
        return nodes 
    
    def plot_nodes(self,lines_plot=False,paths = None,dynamic = True):
        '''
        画出散点图
        是否画出点的连线
        是否动态显示
        是否输入路径
        '''
        self.map.plot_map(paths)
        for node in self.node_lists:
            plt.scatter(node.y,node.x,c= 'w',s = 10)
            paths = []
            if node.neighbors:
                for key in node.neighbors.keys():
                    neigh_node = self.node_lists[key]
                    paths.append([node.x, node.y])
                    paths.append([neigh_node.x, neigh_node.y])
                if lines_plot:
                    paths = np.array(paths)
                    plt.plot(paths[:, 1], paths[:, 0], c='g')
                if dynamic:
                    plt.pause(0.0000001)



def main():
    wg = WeightGraph()
    wg.plot_nodes(True)
    plt.show()

if __name__ == '__main__':
    main()