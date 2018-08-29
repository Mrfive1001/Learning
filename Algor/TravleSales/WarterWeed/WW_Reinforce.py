import numpy as np
import math
import multiprocessing as mp
import time
import gym
import copy


class Para:
    # 强化学习环境和神经网络参数
    def __init__(self,
                 env=None,  # 环境
                 s_dim=10,  # 状态维度
                 a_dim=2,  # 动作维度
                 abound=None,  # 动作上下界[[1,1,2],[3,3,4]]
                 ep_max_step=500,  # 每回合最大步数
                 units=30,  # 神经网络单元数
                 continuous_a=[True],  # 是否离散
                 stop_reward=None   # 截止条件
                 ):
        self.env = env
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.abound = abound
        self.ep_max_step = ep_max_step
        self.units = units
        self.continuous_a = continuous_a
        self.stop_reward = stop_reward
        self.net_shapes, self.example_ww = self.build_net()  # 神经网络shape

    def build_net(self):
        # 建立神经网络结构，得到shape和神经网络参数(一维)
        def linear(n_in, n_out):  # network linear layer
            w = np.random.randn(n_in * n_out).astype(np.float32) * .1
            b = np.random.randn(n_out).astype(np.float32) * .1
            return (n_in, n_out), np.concatenate((w, b))

        s0, p0 = linear(self.s_dim, self.units)
        s1, p1 = linear(self.units, self.units)
        s2, p2 = linear(self.units, self.a_dim)
        return [s0, s1, s2], np.concatenate((p0, p1, p2))

    def choose_action(self, params, state):
        # 根据当前网络参数和状态输出动作
        p, start = [], 0
        for i, shape in enumerate(self.net_shapes):  # flat params to matrix
            n_w, n_b = shape[0] * shape[1], shape[1]
            p = p + [params[start: start + n_w].reshape(shape),
                     params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
            start += n_w + n_b
        params = p.copy()
        state = state[np.newaxis, :]
        state = np.tanh(state.dot(params[0]) + params[1])
        state = np.tanh(state.dot(params[2]) + params[3])
        state = state.dot(params[4]) + params[5]
        if not self.continuous_a[0]:  # 离散动作
            return np.argmax(state, axis=1)[0]  # for discrete action
        else:
            return (self.abound[1] - self.abound[0]) / 2 * np.tanh(state)[0] + np.mean(self.abound)


class WW:
    # 水草算法
    def __init__(self,
                 para=None,  # 强化学习参数
                 num_ww=20,  # 水草种群数量
                 max_age=10,  # 每株水草的存活年限
                 num_seeds=20,  # 水草生产种子最大数量
                 change_rate=0.3,  # 变异概率
                 num_var=2,  # 变量个数,强化学习中不起作用
                 bound_var=None,  # 变量上下界 eg.[[0,0,0],[3,4,5]] 3个变量上下界,强化学习中不起作用
                 ):
        # 六个超参数
        self.num_ww = num_ww
        self.change_rate = change_rate
        self.max_age = max_age
        self.num_seeds = num_seeds
        self.para = para
        self.num_var = len(self.para.example_ww)
        self.bound_var = np.array([[-1] * self.num_var, [1] * self.num_var])

        self.wws = np.array([np.random.rand(self.num_var) * (self.bound_var[1] - self.bound_var[0]) +
                             self.bound_var[0] for _ in range(self.num_ww)])  # 初始化种群
        self.ages = np.zeros(self.num_ww)  # 初始化族群年龄
        self.wws_fitness = np.array([self.fitness(ww) for ww in self.wws])  # 得到初始化种群的适应值
        self.wws_rank = np.argsort(-self.wws_fitness)  # 对适应值由大到小排序得到第一名所在位置，第二名等所在位置
        self.best = {'ww': self.wws[self.wws_rank[0]],
                     'fitness': self.wws_fitness[self.wws_rank[0]]}  # 将最好的结果保存下来

    def run(self, pool):
        # 主函数
        for epsi in range(1000):
            # 每一代
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
            if self.wws_fitness[self.wws_rank[0]] > self.best['fitness']:
                self.best = {'ww': self.wws[self.wws_rank[0]],
                             'fitness': self.wws_fitness[self.wws_rank[0]]}  # 将最好的结果保存下来
            print('第%d轮价值函数平均值为%f,历史最好状态为%f' % (epsi, np.mean(self.wws_fitness), self.best['fitness']))
            if self.best['fitness'] >= self.para.stop_reward:
                np.save("WW_Net/model.npy", self.best['ww'])
                break

    def make_kid(self, rank, num):
        # 根据排序来决定子代种子的数量
        seeds = []
        mother = self.wws[num].copy()
        seeds_num = int(max(math.ceil(self.num_seeds - (rank - 1) * (self.num_seeds - 1) / (self.num_ww)), 1))  # 种子数量
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

    def fitness(self, ww):
        # 评估参数适应值的函数
        # 输入一组参数，返回适应值实数
        env1 = copy.deepcopy(self.para.env)
        ep_r = 0
        num = 2
        for _ in range(num):
            s = env1.reset()
            print(s)
            for step in range(self.para.ep_max_step):
                a = self.para.choose_action(ww, s)
                s, r, done, _ = env1.step(a)
                ep_r += r
                if done: break
        return ep_r/num

    def display(self, CONFIG):
        # 显示
        ww = np.load("WW_Net/model.npy")
        while True:
            s = env.reset()
            for _ in range(CONFIG['ep_max_step']):
                env.render()
                a = self.para.choose_action(ww, s)
                s, r, done, _ = env.step(a)
                if done:
                    break

if __name__ == '__main__':
    CONFIG = [
        dict(game="CartPole-v0",
             n_feature=4, n_action=2, continuous_a=[False], a_bound=None, ep_max_step=700, eval_threshold=500),
        dict(game="MountainCar-v0",
             n_feature=2, n_action=3, continuous_a=[False], a_bound=None, ep_max_step=200, eval_threshold=-120),
        dict(game="Pendulum-v0",
             n_feature=3, n_action=1, continuous_a=[True, 2.], a_bound=[-2., 2.], ep_max_step=200, eval_threshold=-120)
    ][2]  # choose your game
    env = gym.make(CONFIG['game']).unwrapped

    para = Para(env=env,
                s_dim=CONFIG['n_feature'],
                a_dim=CONFIG['n_action'],
                abound=CONFIG['a_bound'],
                continuous_a=CONFIG['continuous_a'],
                stop_reward=CONFIG['eval_threshold'])
    ww = WW(para=para,
            num_ww=20,  # 水草种群数量
            max_age=10,  # 每株水草的存活年限
            num_seeds=20,  # 水草生产种子最大数量
            change_rate=0.4,  # 变异概率
            )
    num_core = mp.cpu_count() - 1  # 多核操作
    pool = mp.Pool(processes=num_core)
    train = True
    # train = False
    if train:
        ww.run(pool)
    else:
        ww.display(CONFIG)
