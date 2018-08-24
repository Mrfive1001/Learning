import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from get_map import Map
sns.set(style='dark')
np.random.seed(200)


def get_data(m=10, n=10, rate=0.1):
    # 输入行列和障碍物所占比例，返回图像数据值
    data = np.ones((m, n))*255
    obstacle_num = int(rate*m*n)
    for _ in range(obstacle_num):
        data[np.random.randint(m), np.random.randint(n)] = 0
    return data


class Ant:
    '''
    输入地图数据
    '''

    def __init__(self, my_map):
        self.map = my_map
        self.numant = 60  # 蚂蚁个数
        self.numcity = self.map.shape[0]*self.map.shape[1]  # 点的数量
        self.alpha = 1  # 信息素重要程度因子
        self.beta = 5  # 启发函数重要程度因子
        self.rho = 0.05  # 信息素的挥发速度
        self.Q = 1  # 完成率
        self.itermax = 150  # 迭代总数
        self.pheromonetable = np.ones((self.numcity, self.numcity))  # 信息素矩阵
        self.lengthaver = []  # 迭代,存放每次迭代后，路径的平均长度
        self.lengthbest = []  # 迭代,存放每次迭代后，最佳路径长度
        self.pathbest = []
        self.pathbest_length = None

    def find_path(self):
        for iter in range(self.itermax):
            # 迭代itermax轮
            paths = []  # 本轮每个蚂蚁走过的路径
            path_valid = []  # 蚂蚁路径是否到达终点
            lengths = []
            start_index = self.map.cor2index(*self.map.start)  # 开始索引
            end_index = self.map.cor2index(*self.map.end)   # 结束索引
            for ant in range(self.numant):
                # 每个蚂蚁开始找路
                visiting = start_index
                length = 0
                unvisited = set(range(self.numcity))
                unvisited.remove(visiting)  # 删除已经访问过的点
                paths.append([visiting])  # 添加刚经过的节点
                while True:
                    # 从当前点到下个点
                    # 判断退出条件，找到终点或者死胡同
                    if visiting == end_index:
                        path_valid.append(True)
                        break
                    indexs = self.map.feasible_points(visiting)
                    next_indexs = []
                    distances = []
                    for index in indexs:
                        if index not in unvisited:
                            continue
                        next_indexs.append(index)
                        distances.append(self.map.get_dis(visiting, index))
                    next_len = len(next_indexs)
                    if next_len == 0:
                        path_valid.append(False)
                        break
                    # 根据概率选择下个要探索的点
                    probtrans = np.zeros(next_len)  # 每次循环都初始化转移概率矩阵
                    for k in range(next_len):
                        probtrans[k] = np.power(self.pheromonetable[visiting][next_indexs[k]], self.alpha) \
                            * np.power(1.0/(distances[k]), self.alpha)
                    # 利用轮盘赌来找到需要探索的点
                    cumsumprobtrans = (probtrans / sum(probtrans)).cumsum()
                    cumsumprobtrans -= np.random.rand()
                    k = list(cumsumprobtrans >= 0).index(
                        True)  # 找到在next_indexs中的索引值
                    next_index = next_indexs[k]
                    length += distances[k]
                    paths[ant].append(next_index)
                    # 选择下个点
                    visiting = next_index
                    unvisited.remove(visiting)  # 删除已经访问过的城市元素
                # 结束之后添加长度
                lengths.append(length)
            # 只看找到终点的
            lengths = np.array(lengths)
            paths = np.array(paths)
            path_valid = np.array(path_valid)
            lengths = lengths[path_valid == True]
            paths = paths[path_valid == True]
            changepheromonetable = np.zeros((self.numcity, self.numcity))
            # 需要改变的信息素矩阵
            if len(lengths) != 0:
                # 至少有一只蚂蚁找到终点
                min_index = lengths.argmin()
                self.lengthbest.append(lengths.min())
                self.lengthaver.append(lengths.mean())
                if self.pathbest == []:
                    self.pathbest = paths[min_index].copy()
                    self.pathbest_length = lengths[min_index]
                else:
                    # 如果当前不是最优，继续更改
                    if self.pathbest_length > self.lengthbest[-1]:
                        self.pathbest_length = self.lengthbest[-1]
                        self.pathbest = paths[min_index].copy()
                for i in range(len(paths)):  # 更新所有的蚂蚁
                    path = paths[i]
                    for j in range(len(path)-1):
                        changepheromonetable[path[j],
                                             path[j+1]] += self.Q*10 / lengths[i]
                print('Iterations:%d,feasible_path:%d,ave_length:%.2f,best_length:%.2f,global best:%.2f' %
                  (iter+1,len(lengths),self.lengthaver[-1],self.lengthbest[-1],self.pathbest_length))
            self.pheromonetable = (1 - self.rho) * \
                self.pheromonetable + changepheromonetable
    def plot_path(self):
        path = []
        for index in self.pathbest:
            path.append(self.map.index2cor(index))
        self.map.plot_map(np.array(path))


def main():
    # 生成数据
    map_data = get_data(40, 40, 0.1)
    # 定义起始点和目标点生成图
    start_point = [0, 0]
    end_point = [27, 23]
    my_map = Map(map_data, start_point, end_point)
    aco = Ant(my_map)
    aco.find_path()
    aco.plot_path()
    plt.show()

if __name__ == '__main__':
    main()
