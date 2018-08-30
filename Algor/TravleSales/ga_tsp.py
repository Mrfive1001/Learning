import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from tsp import TSP
np.random.seed(10)

class GaTsp:
    def __init__(self, data_type='small'):
        self.tsp = TSP(data_type)
        # self.tsp.cities = np.random.rand(52,2)
        self.data_type = data_type
        self.cross_rate = 0.5  # 交叉概率，如果不交叉的话直接继承父代的特性
        self.mutate_rate = 0.02  # 变异概率
        self.numcity = len(self.tsp.cities)  # 点的数量
        self.pop_size = 500  # 种群数量
        self.itermax = 1000
        self.pop = np.vstack([np.random.permutation(self.numcity)
                              for _ in range(self.pop_size)])  # pop_size*numcity
        self.distances = np.array([self.tsp.get_fitness(pop) for pop in self.pop])
        self.fit = np.exp(self.numcity*5000/(self.distances))
        self.lengthaver = []  # 迭代,存放每次迭代后，路径的平均长度
        self.lengthbest = []  # 迭代,存放每次迭代后，最佳路径长度
        self.pathbest = []
        self.pathbest_length = None
        self.result_dir = os.path.join(sys.path[0], 'Results')

    def find_path(self, iterations=None):
        '''
        寻找最短路径
        '''
        if iterations is None:
            iterations = self.itermax
        for iter in range(iterations):
            # 迭代itermax轮
            pop = self.select()
            pop_copy = pop.copy()
            for parent in pop:  # for every parent
                child = self.crossover(parent, pop_copy)
                child = self.mutate(child)
                parent[:] = child
            self.pop = pop
            self.distances = np.array([self.tsp.get_fitness(pop) for pop in self.pop])
            self.fit = np.exp(self.numcity * 5000 / (self.distances))
            # 保存最优值
            min_index = self.fit.argmax()
            self.lengthbest.append(self.distances.min())
            self.lengthaver.append(self.distances.mean())
            if self.pathbest == []:
                # 第一次
                self.pathbest = self.pop[min_index].copy()
                self.pathbest_length = self.distances[min_index]
            else:
                # 如果当前不是最优，继续更改
                if self.pathbest_length > self.distances[min_index]:
                    self.pathbest_length = self.distances[min_index]
                    self.pathbest = self.pop[min_index].copy()
            print('Iterations:%d,ave_length:%.2f,best_length:%.2f,global best:%.2f' % (
                iter + 1, self.lengthaver[-1], self.lengthbest[-1], self.pathbest_length))

    def select(self):
        '''
        按照fitness值选择配对父母,未选择到的自行淘汰
        '''
        index = np.random.choice(np.arange(
            self.pop_size), size=self.pop_size, replace=True, p=self.fit / self.fit.sum())
        return self.pop[index].copy()

    def crossover(self, parent, pop):
        '''
        从pop中选择一个父代与parent产生子代
        如果不交叉的话直接继承父代的特性
        '''
        if np.random.rand() < self.cross_rate:
            index = np.random.randint(0, self.pop_size, size=1)  # 选择另一个亲代
            cross_points = np.random.randint(
                0, 2, self.numcity).astype(np.bool)
            # 选择交叉点,numcity个0或者1的随机数
            keep_city = parent[~cross_points]  # 取出父代保留的城市
            swap_city = pop[index, np.isin(
                pop[index].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def mutate(self, child):
        '''
        对路径进行变异，两个点进行交换
        '''
        for point in range(self.numcity):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.numcity)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def plot_result(self):
        self.tsp.plot_map(self.pathbest)

    def save_result(self):
        self.tsp.plot_map(self.pathbest)
        plt.savefig(os.path.join(self.result_dir,
                                 'ACO_%s.png' % self.data_type))


def main():
    ga = GaTsp('small')
    ga.find_path()
    ga.plot_result()
    plt.show()


if __name__ == '__main__':
    main()
