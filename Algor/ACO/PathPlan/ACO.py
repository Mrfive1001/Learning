from generate_data import get_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='dark')
np.random.seed(200)


class Map:
    def __init__(self, map, start_position, end_position,):
        self.data = map
        self.start = start_position
        self.end = end_position

    def is_valid(self, position):
        # 判断点是否有效，是否在边界内
        if position[0] < 0 or position[0] >= self.data.shape[0] or position[1] < 0 or position[1] >= self.data.shape[1]:
            return False
        if self.data[position[0], position[1]] == 0:
            return False
        return True

    def plot_map(self, path=None):
        # 画出地图，如果规划的路径不为空的话，将其画出来
        # 注意地图中x与y是反过来的，横坐标是y
        plt.imshow(self.data)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
        if path is not None:
            plt.plot(path[:, 1], path[:, 0], 'r')
            plt.scatter([self.start[1], self.end[1]],
                        [self.start[0], self.end[0]], s=40, marker='*', c='r')

    def index2cor(self, index):
        return index//self.data.shape[1], index % self.data.shape[1]

    def cor2index(self, x, y):
        return x*self.data.shape[1]+y

    def feasible_points(self, index, stored):
        # 输入当前点和没探索过的点，找出周围可行点
        x, y = self.index2cor(index)
        distances = []
        indexs = []
        for i in range(3):
            for j in range(3):
                x_new, y_new = x-1+i, y-1+j
                index_new = self.cor2index(x_new, y_new)
                if self.is_valid([x_new, y_new]) == False or index_new not in stored:
                    continue
                indexs.append(index_new)
                add_value = ((i - 1) ** 2 + (j - 1) ** 2) ** 0.5
                distances.append(add_value)
        return indexs, distances

    def get_dis(self, index1, index2):
        x1, y1 = self.index2cor(index1)
        x2, y2 = self.index2cor(index2)
        return ((x1-x2)**2+(y1-y2)**2)**0.5

    def get_path(self, path):
        if path[-1] == True or path[-1] == False:
            path = path[:-1]
        paths = []
        for i in range(len(path)):
            paths.append([*self.index2cor(path[i])])
        return paths


map_data = get_data(40, 40, 0.1)
start_point = [0, 0]
end_point = [27, 23]
map = Map(map_data, start_point, end_point)

# plt.show()

numant = 60  # 蚂蚁个数
numcity = map_data.shape[0]*map_data.shape[1]  # 点的数量
alpha = 1  # 信息素重要程度因子
beta = 5  # 启发函数重要程度因子
rho = 0.05  # 信息素的挥发速度
Q = 1  # 完成率
itermax = 150  # 迭代总数

pheromonetable = np.ones((numcity, numcity))  # 信息素矩阵
lengthaver = np.zeros(itermax)  # 迭代,存放每次迭代后，路径的平均长度
lengthbest = np.zeros(itermax)  # 迭代,存放每次迭代后，最佳路径长度
pathbest = []


for iter in range(itermax):
    # 迭代总数
    pathtable = []
    lengths = []
    start_index = map.cor2index(*map.start)
    end_index = map.cor2index(*map.end)
    for ant in range(numant):
        # 每个蚂蚁开始找路
        visiting = start_index
        length = 0
        unvisited = set(range(numcity))
        unvisited.remove(visiting)  # 删除已经访问过的城市元素
        pathtable.append([visiting])
        while True:
            if visiting == end_index:
                pathtable[ant].append(True)
                break
            next_indexs, distances = map.feasible_points(visiting, unvisited)
            next_len = len(next_indexs)
            if next_len == 0:
                pathtable[ant].append(False)
                break
            probtrans = np.zeros(next_len)  # 每次循环都初始化转移概率矩阵
            for k in range(next_len):
                probtrans[k] = np.power(pheromonetable[visiting][next_indexs[k]], alpha) \
                    * np.power(1.0/(distances[k]), alpha)
            cumsumprobtrans = (probtrans / sum(probtrans)).cumsum()
            cumsumprobtrans -= np.random.rand()
            k = list(cumsumprobtrans >= 0).index(True)
            next_index = next_indexs[k]
            length += distances[k]
            pathtable[ant].append(next_index)
            visiting = next_index
            unvisited.remove(visiting)  # 删除已经访问过的城市元素
        lengths.append(length)

    lengths = np.array(lengths)
    if iter == 0:
        lengthbest[iter] = lengths.min()
        pathbest.append(pathtable[lengths.argmin()].copy())
    # 如果是第一轮路径，则选择本轮最短的路径,并返回索引值下标，并将其记录
    else:
        # 后面几轮的情况，更新最佳路径
        if lengths.min() > lengthbest[iter - 1]:
            lengthbest[iter] = lengthbest[iter - 1]
            pathbest.append(pathbest[iter - 1].copy())
        # 如果是第一轮路径，则选择本轮最短的路径,并返回索引值下标，并将其记录
        else:
            lengthbest[iter] = lengths.min()
            pathbest.append(pathtable[lengths.argmin()].copy())
    # 更新信息素矩阵
    changepheromonetable = np.zeros((numcity, numcity))
    num = 0
    for i in range(numant):  # 更新所有的蚂蚁
        path = pathtable[i][:-1]
        MissionCom = pathtable[i][-1]
        if MissionCom == True:
            num += 1
            for j in range(len(path)-1):
                changepheromonetable[path[j], path[j+1]] += Q*10 / lengths[i]
    pheromonetable = (1 - rho) * pheromonetable + changepheromonetable
    print(num)

path = np.array(map.get_path(pathtable[-1]))
map.plot_map(path)
plt.show()
