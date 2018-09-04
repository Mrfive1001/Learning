import matplotlib.pyplot as plt
import numpy as np

from weight_map import Node, WeightGraph


class GSearch:
    '''
    实现图搜索算法的三种形式
    输入：地图对象，算法名称
    算法名称：
    Dijstra算法       慢   'D'
    A*算法            中   'A'
    BFS最佳搜索       快   'B'
    '''

    def __init__(self, my_map, alg):
        self.map = my_map
        self.alg = alg
        self.open_lists = []
        self.close_lists = []
        self.explore_path = []
        self.result_path = []
        self.node_lists = self.init_nodes()

    def init_nodes(self):
        '''
        初始化点对象
        '''
        results = []
        for node in self.map.node_lists:
            results.append(GNode(node))
        return results

    def find_path(self):
        '''
        按照相应算法找到路径
        '''
        self.start_node = self.node_lists[0]
        self.end_node = self.node_lists[-1]
        self.end_index = len(self.node_lists) - 1
        cur_index = 0
        cur_node = self.start_node
        self.open_lists.append(cur_index)
        while True:
            next_node = None
            next_f = 0
            self.close_lists.append(cur_index)
            self.open_lists.remove(cur_index)
            neighs = cur_node.neighbors.keys()
            self.explore_path.append([cur_node.x, cur_node.y])
            for neigh_index in neighs:
                node = self.node_lists[neigh_index]
                if neigh_index in self.close_lists:
                    continue
                else:
                    dis = cur_node.neighbors[neigh_index]
                    if neigh_index in self.open_lists:
                        if (dis + cur_node.g_score) < node.g_score:
                            node.g_score = dis + cur_node.g_score
                            node.father = cur_index
                            node.f_score = self.get_f(node.g_score, node.h_score)
                    else:
                        node.g_score = dis + cur_node.g_score
                        node.father = cur_index
                        node.h_score = self.get_h(node)
                        node.f_score = self.get_f(node.g_score, node.h_score)
                        self.open_lists.append(neigh_index)
                # 选出下一个要探索的点
                if next_node is None:
                    next_node = node
                    next_index = neigh_index
                    next_f = node.f_score
                else:
                    if next_f > node.f_score:
                        next_node = node
                        next_index = neigh_index
                        next_f = node.f_score
            # 选出下一个要探索的点
            if next_node is None:
                for node_index in self.open_lists:
                    node = self.node_lists[node_index]
                    if next_node:
                        if node.f_score < next_node.f_score:
                            next_index = node_index
                            next_node = node
                            next_f = node.f_score
                    else:
                        next_node = node
                        next_index = node_index
                        next_f = next_node.f_score
            if self.end_index in self.close_lists:
                break
            cur_node = next_node
            cur_index = next_index

        cur_node = self.end_node
        while True:
            self.result_path.append([cur_node.x, cur_node.y])
            cur_node = self.node_lists[cur_node.father]
            if cur_node.father is None:
                break
        self.result_path = np.array(self.result_path)
        self.explore_path = np.array(self.explore_path)

    def plot(self):
        self.map.plot_nodes(paths=self.result_path, dynamic=False)

    def get_f(self, g, h):
        # 设置Node的f值
        # 传入Node对象
        if self.alg == 'A':
            alpha = 0.5
        elif self.alg == 'D':
            alpha = 0
        else:
            alpha = 1
        return alpha * h + (1 - alpha) * g

    def get_h(self, node):
        return ((self.end_node.x - node.x) ** 2 + (self.end_node.y - node.y) ** 2) ** 0.5


class GNode:
    def __init__(self, Node):
        self.x = Node.x
        self.y = Node.y
        self.neighbors = Node.neighbors
        self.father = None
        self.g_score = 0  # 起始点到当前位置的距离
        self.h_score = 0  # 到目标点的距离
        self.f_score = None  # 每个点的价值


def main():
    wg_map = WeightGraph(data_resoure=6)
    gs_alg = GSearch(wg_map, 'D')
    gs_alg.find_path()
    gs_alg.plot()
    plt.show()


if __name__ == '__main__':
    main()
