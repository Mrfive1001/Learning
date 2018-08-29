import numpy as np
import math
import multiprocessing as mp
import time

class WW:
    # 水草算法
    def __init__(self,
                 num_ww=20,  # 水草种群数量
                 max_age=10,  # 每株水草的存活年限
                 num_seeds=20,  # 水草生产种子最大数量
                 change_rate=0.3,  # 变异概率
                 num_var=2,  # 变量个数
                 bound_var=None,  # 变量上下界 eg.np.array([0,0,0],[3,4,5]) 3个变量上下界
                 ):
        # 六个超参数
        self.num_ww = num_ww
        self.change_rate = change_rate
        self.max_age = max_age
        self.num_seeds = num_seeds
        self.num_var = num_var
        self.bound_var = np.array(bound_var)

        self.wws = np.array([np.random.rand(self.num_var) * (self.bound_var[1] - self.bound_var[0]) +
                             self.bound_var[0] for _ in range(self.num_ww)])  # 初始化种群
        self.ages = np.zeros(self.num_ww)  # 初始化族群年龄
        self.wws_fitness = np.array([self.fitness(ww) for ww in self.wws])  # 得到初始化种群的适应值
        self.wws_rank = np.argsort(-self.wws_fitness)  # 对适应值由大到小排序得到第一名所在位置，第二名等所在位置
        self.best = {'ww': self.wws[self.wws_rank[0]],
                     'fitness': self.wws_fitness[self.wws_rank[0]]}  # 将最好的结果保存下来

    def step(self, pool):
        # 进行一次迭代
        for i in range(len(self.wws_rank)):
            # 由最大到最小开始循环
            num = int(self.wws_rank[i])  # 在种群中的序号，i+1 表示第几名
            ww = self.wws[num]  # 水草
            fitness = self.wws_fitness[num]
            seeds = self.make_kid(i + 1, num)  # 产生一群种子
            if pool is None:
                wws_fitness = np.array([self.fitness(ww) for ww in seeds])  # 直接计算
            else:
                jobs = [pool.apply_async(self.fitness, (ww,)) for ww in seeds]  # 并行运算
                wws_fitness = np.array([j.get() for j in jobs])

            max_num = np.argmax(wws_fitness)
            next_ww, next_fitness = seeds[max_num], wws_fitness[max_num]
            if self.ages[num] < self.max_age and fitness >= next_fitness:
                self.ages[num] += 1
                # 仍然保持母代
            else:
                # 母代死亡,子代取代
                self.wws[num] = next_ww.copy()
                self.wws_fitness[num] = next_fitness
                self.ages[num] = 0
        self.wws_rank = np.argsort(-self.wws_fitness)  # 对适应值由大到小排序得到第一名所在位置，第二名等所在位置
        self.best = {'ww': self.wws[self.wws_rank[0]],
                     'fitness': self.wws_fitness[self.wws_rank[0]]}  # 将最好的结果保存下来
        print('价值函数平均值为%f' % (np.mean(self.wws_fitness)))
        print(self.best)

    def make_kid(self, rank, num):
        # 根据排序来决定子代种子的数量
        seeds = []
        mother = self.wws[num].copy()
        seeds_num = max(math.ceil(self.num_seeds - (rank - 1) * (self.num_seeds - 1) / (self.num_ww)), 1)  # 种子数量
        for _ in range(seeds_num):
            # 对于每一个种子
            while 1:
                father_num = np.random.randint(0, self.num_ww)
                if father_num != num:
                    break
            # 选出父代
            ww = mother
            change = 0 + (np.random.rand(len(mother)) < self.change_rate)  # 是否变异，不变异就跟母亲一样
            change_scale = np.random.rand(len(mother))
            ww = ww + change * 2 * (change_scale - 0.5) * (ww - self.wws[father_num])
            ww = np.clip(ww, self.bound_var[0], self.bound_var[1])
            seeds.append(ww)
        return seeds

    def fitness(self, ww, ):
        # 评估参数适应值的函数
        # 输入一组参数，返回适应值实数
        f = 0.5 + ((np.sin((ww[0] ** 2 + ww[1] ** 2) ** 0.5)) ** 2 - 0.5) / (1 + 0.001 * (ww[0] ** 2 + ww[1] ** 2)) ** 2
        return -f


if __name__ == '__main__':
    ww = WW(num_ww=20,  # 水草种群数量
            max_age=2,  # 每株水草的存活年限
            num_seeds=20,  # 水草生产种子最大数量
            change_rate=0.5,  # 变异概率
            num_var=2,  # 变量个数
            bound_var=[[-100, -100], [100, 100]]  # 变量上下界 eg.np.array([0,0,0],[3,4,5]) 3个变量上下界
            )
    num_core = mp.cpu_count() - 1  # 多核操作
    pool = mp.Pool(processes=num_core)
    for _ in range(100):
        ww.step(pool)
