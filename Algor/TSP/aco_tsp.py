import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from tsp import TSP


class AntTsp:
    def __init__(self, data_type='small'):
        self.tsp = TSP(data_type)
        self.numant = 100  # 蚂蚁个数
        self.numcity = len(self.tsp.cities)  # 点的数量
        self.alpha = 1  # 信息素重要程度因子
        self.beta = 5  # 启发函数重要程度因子
        self.rho = 0.1  # 信息素的挥发速度
        self.Q = 1  # 完成率
        self.itermax = 70  # 迭代总数
        self.pheromonetable = np.ones((self.numcity, self.numcity))  # 信息素矩阵
        self.distances = self.get_dis()
        self.lengthaver = []  # 迭代,存放每次迭代后，路径的平均长度
        self.lengthbest = []  # 迭代,存放每次迭代后，最佳路径长度
        self.pathbest = []
        self.pathbest_length = None

    def find_path(self, iterations=None):
        if iterations is None:
            iterations = self.itermax
        for iter in range(iterations):
            # 迭代itermax轮
            paths = []  # 本轮每个蚂蚁走过的路径
            lengths = []  # 每个蚂蚁的本轮走过的长度
            start_index = 0  # 从第0个开始寻找
            for ant in range(self.numant):
                # 每个蚂蚁开始找路
                visiting = start_index
                length = 0
                unvisited = list(range(self.numcity))
                unvisited.remove(visiting)  # 删除已经访问过的点
                paths.append([visiting])  # 添加刚经过的节点
                while True:
                    # 根据概率选择下个要探索的点
                    probtrans=[]  # 每次循环都初始化转移概率矩阵
                    for city in unvisited:
                        probtrans.append(np.power(self.pheromonetable[visiting, city], self.alpha)
                            * np.power(1.0/self.distances[visiting, city], self.alpha))
                    # 利用轮盘赌来找到需要探索的点
                    probtrans=np.array(probtrans)
                    cumsumprobtrans=(probtrans / sum(probtrans)).cumsum()
                    cumsumprobtrans -= np.random.rand()
                    k=list(cumsumprobtrans >= 0).index(True)  # 找到在list中的索引值
                    next_city=unvisited[k]
                    length += self.distances[visiting, next_city]
                    paths[ant].append(next_city)
                    # 选择下个点
                    visiting=next_city
                    unvisited.remove(visiting)  # 删除已经访问过的城市元素
                    if unvisited == []:
                        length += self.distances[visiting, start_index]
                        paths[ant].append(start_index)
                        break
                # 结束之后添加长度
                lengths.append(length)
            # 进行信息素更新
            changepheromonetable=np.zeros((self.numcity, self.numcity))
            lengths=np.array(lengths)
            min_index=lengths.argmin()
            self.lengthbest.append(lengths.min())
            self.lengthaver.append(lengths.mean())
            if self.pathbest == []:
                # 第一次
                self.pathbest=paths[min_index].copy()
                self.pathbest_length=lengths[min_index]
            else:
                # 如果当前不是最优，继续更改
                if self.pathbest_length > lengths[min_index]:
                    self.pathbest_length=lengths[min_index]
                    self.pathbest=paths[min_index].copy()
                for i in range(len(paths)):  # 更新所有的蚂蚁
                    path=paths[i]
                    for j in range(len(path)-1):
                        changepheromonetable[path[j], path[j+1]] +=  self.Q*10 / self.distances[path[j],path[j+1]]
            print('Iterations:%d,ave_length:%.2f,best_length:%.2f,global best:%.2f' % (iter+1, self.lengthaver[-1], self.lengthbest[-1], self.pathbest_length))
            self.pheromonetable=(1 - self.rho) * \
                self.pheromonetable + changepheromonetable

    def get_dis(self):
        '''
        得到城市之间的距离矩阵
        '''
        distances=np.zeros((self.numcity,self.numcity))
        for i in range(self.numcity):
            for j in range(self.numcity):
                if i == j:
                    continue
                elif i > j:
                    # 计算过就不用算了
                    distances[i, j]=distances[j, i]
                else:
                    distances[i, j]=np.linalg.norm(self.tsp.cities[i]-self.tsp.cities[j])+1
        return distances

    def plot_result(self):
        self.tsp.plot_map(self.pathbest)


def main():
    ants = AntTsp('middle')
    ants.find_path()
    ants.plot_result()
    plt.show()

if __name__ == '__main__':
    main()
