import multiprocessing
import os
import pickle
import sys
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from aco import get_data
from get_map import Map

np.random.seed(200)
'''
图搜索的三种算法实现路径规划
'''


class GraphSearch:
    '''
    实现图搜索算法的三种形式
    输入：地图对象，算法名称，搜索模式
    算法名称：
    Dijstra算法       慢   'D'
    A*算法            中   'A'
    BFS最佳搜索       快   'B'
    搜索模式:
    mode = 0 正向搜索
    mode = 1 逆向搜索
    mode = 2 双向搜索加上多进程
    mode = 3 双向搜索加上多线程
    '''

    def __init__(self, my_map, alg, mode):
        self.map = my_map
        self.alg = alg
        self.mode = mode
        self.data_dir = os.path.join(sys.path[0], 'Data')  # 存放中间变量
        self.result_dir = os.path.join(sys.path[0], 'Results')  # 存放结果

    def get_f(self, node):
        # 设置Node的f值
        # 传入Node对象
        if self.alg == 'A':
            alpha = 0.5
        elif self.alg == 'D':
            alpha = 0
        else:
            alpha = 1
        return alpha * node.h_score + (1 - alpha) * node.g_score

    def find_path(self):
        '''
        按照相应算法找到路径
        返回：找到的最优路径，探索过的点，相关信息
        '''
        if self.mode >= 2:
            # 多任务搜索路径
            result_path, explore_path, info = self.multi_path()
        else:
            if self.mode == 1:
                # 起点终点交换
                temp = self.map.end.copy()
                self.map.end = self.map.start.copy()
                self.map.start = temp.copy()
            # 从开始点进行计算
            start_node = Node(None, self.map.start[0], self.map.start[1])
            end_node = Node(None, self.map.end[0], self.map.end[1])
            open_set = {start_node.hash(): start_node}  # 开启列表加入的是周围未经过完全探索的点
            close_set = {}  # 关闭列表是经过完全探索的点
            curNode = start_node
            explore_path = []  # 所有经过的点,与close_set是等价的
            # 往前探索
            while True:
                curNode, open_set, close_set, explore_path = self.step(
                    curNode, open_set, close_set, explore_path, False)
                if end_node.hash() in close_set:
                    # 如果终点被探索过就退出
                    break
            # 回溯
            result_path = []
            curNode = close_set[end_node.hash()]
            info = {}
            info['total_length'] = curNode.g_score
            while curNode:
                result_path.append([curNode.x, curNode.y])
                curNode = curNode.father
            result_path = np.array(result_path)
            result_path = result_path[range(len(result_path) - 1, 0, -1)]
            explore_path = np.array(explore_path)
        np.savez(os.path.join(self.result_dir, 'result.npz'),
                 final=result_path, explore=explore_path,length = info['total_length'])
        return result_path, explore_path, info

    def multi_path(self):
        # 双向任务搜索
        start_nodeP = Node(None, self.map.start[0], self.map.start[1])  # 正向起点
        start_nodeN = Node(None, self.map.end[0], self.map.end[1])  # 反向起点
        self.open_setP = {start_nodeP.hash(): start_nodeP}
        self.close_setP = {}
        self.open_setN = {start_nodeN.hash(): start_nodeN}
        self.close_setN = {}
        self.curNodeP = start_nodeP
        self.explore_pathP = []
        self.curNodeN = start_nodeN
        self.explore_pathN = []
        self.mid_node = None
        self.go_on = True
        # 从头尾两个方面进行探索
        if self.mode == 2:
            # 多进程处理
            end = multiprocessing.Value('i', 1)
            curN = multiprocessing.Array('i', [start_nodeN.x, start_nodeN.y])
            curP = multiprocessing.Array('i', [start_nodeP.x, start_nodeP.y])
            mid = multiprocessing.Queue()
            obj1 = (True, end, curN, curP, mid)
            obj2 = (False, end, curN, curP, mid)
            # 制定正反两个方向的任务，进行执行，得到的结果放到文件里面
            t1 = multiprocessing.Process(target=self.loop_mulpro, args=(obj1,))
            t2 = multiprocessing.Process(target=self.loop_mulpro, args=(obj2,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            # 读取文件结果
            res1 = np.load(os.path.join(self.data_dir, str(True) + '.npz'))
            res2 = np.load(os.path.join(self.data_dir, str(False) + '.npz'))

            explore_pathN = res1['explore']
            setN = res1['set']
            lengthN = res1['length']

            explore_pathP = res2['explore']
            setP = res2['set']
            lengthP = res2['length']

            info = {}
            info['total_length'] = lengthP + lengthN

            result = setP[range(len(setP) - 2, -1, -1)]
            result = np.vstack((result, setN))
            explore_path = np.vstack((explore_pathP, explore_pathN))
        elif self.mode == 3:
            # 多线程处理
            self.lock = threading.Lock()
            # 方式1
            # t1 = threading.Thread(target=self.loop_multhr, args=(True,))
            # t2 = threading.Thread(target=self.loop_multhr, args=(False,))
            # t1.start()
            # t2.start()
            # t1.join()
            # t2.join()
            # 方式2 带返回值的
            from multiprocessing.pool import ThreadPool
            pool = ThreadPool(2)
            res = pool.map(self.loop_multhr, [*(True,), *(False,)])
            # 回溯
            resultP = []
            resultN = []
            try:
                curNodeN = self.open_setN[self.mid_node.hash()]
            except Exception:
                curNodeN = self.close_setN[self.mid_node.hash()]
            try:
                curNodeP = self.open_setP[self.mid_node.hash()]
            except Exception:
                curNodeP = self.close_setP[self.mid_node.hash()]
            info = {}
            info['total_length'] = curNodeN.g_score + curNodeP.g_score
            while curNodeP:
                resultP.append([curNodeP.x, curNodeP.y])
                curNodeP = curNodeP.father
            while curNodeN:
                resultN.append([curNodeN.x, curNodeN.y])
                curNodeN = curNodeN.father
            explore_path = self.explore_pathP
            explore_path.extend(self.explore_pathN)
            result = resultP[:: -1]
            result.extend(resultN[1:])
        return np.array(result), np.array(explore_path), info

    def step(self, curNode, open_set, close_set, explore_path, inverse):
        '''
        实现当前点到下一个点规划的一步转换
        输入：当前点 开启列表 关闭列表 探索过的路径 是否起终点互换(多任务处理有用)
        输出：当前点 开启列表 关闭列表 探索过的路径
        '''
        nextNode = None
        nextF = 0
        close_set[curNode.hash()] = curNode
        del open_set[curNode.hash()]
        explore_path.append([curNode.x, curNode.y])
        for i in range(3):
            for j in range(3):
                # 周围的8个点分为3种情况
                node = Node(curNode, curNode.x - 1 + i, curNode.y - 1 + j)
                # 1 如果无效或着已关闭
                if self.map.is_valid([node.x, node.y]) == False or node.hash() in close_set:
                    continue
                else:
                    # 2 开启列表中已存在
                    add_value = ((i - 1) ** 2 + (j - 1) ** 2) ** 0.5
                    if node.hash() in open_set:
                        node = open_set[node.hash()]
                        if (add_value + curNode.g_score) < node.g_score:
                            node.set_g(add_value + curNode.g_score)
                            node.father = curNode
                            node.set_f(self.get_f(node))
                    # 3 开启列表中未存在
                    else:
                        node.set_g(add_value + curNode.g_score)
                        value = None
                        if inverse:
                            # 目标是起始点
                            value = self.map.get_dis(
                                [node.x, node.y], self.map.start, kind='cor')
                        else:
                            # 目标是结束点
                            value = self.map.get_dis(
                                [node.x, node.y], self.map.end, kind='cor')
                        node.set_h(value)
                        node.set_f(self.get_f(node))
                        open_set[node.hash()] = node
                    # 选择出下一个点
                    # 找到最小f值的点如果找不到将执行下一步
                    if nextNode is None:
                        nextNode = node
                        nextF = node.f_score
                    else:
                        if nextF > node.f_score:
                            nextNode = node
                            nextF = node.f_score
        # 选出下一个要探索的点
        if nextNode is None:
            for val in open_set.values():
                if nextNode:
                    if val.f_score < nextNode.f_score:
                        nextNode = val
                        nextF = val.f_score
                else:
                    nextNode = val
                    nextF = nextNode.f_score
        curNode = nextNode
        info = (curNode, open_set, close_set, explore_path)
        return info

    def loop_multhr(self, inverse):
        # 多线程求解
        while self.go_on:
            if inverse:
                # 对反向的进行迭代判断
                self.curNodeN, self.open_setN, self.close_setN, self.explore_pathN = self.step(
                    self.curNodeN, self.open_setN, self.close_setN, self.explore_pathN, True)
                # 判断是否需要结束
                if self.curNodeP.hash() in self.close_setN:
                    self.lock.acquire()
                    self.go_on = False
                    self.mid_node = self.curNodeP
                    self.lock.release()
                    break
            else:
                self.curNodeP, self.open_setP, self.close_setP, self.explore_pathP = self.step(
                    self.curNodeP, self.open_setP, self.close_setP, self.explore_pathP, False)
                if self.curNodeN.hash() in self.close_setP:
                    self.lock.acquire()
                    self.go_on = False
                    self.mid_node = self.curNodeN
                    self.lock.release()
                    break
        return None

    def loop_mulpro(self, obj):
        # 多进程求解
        # 进行循环求解
        inverse, end, curNq, curPq, mid = obj
        my_mid = None
        while end.value:
            if inverse:
                self.curNodeN, self.open_setN, self.close_setN, self.explore_pathN = self.step(
                    self.curNodeN, self.open_setN, self.close_setN, self.explore_pathN, True)
                curNq[0], curNq[1] = self.curNodeN.x, self.curNodeN.y
                value = [curPq[0], curPq[1]]
                has = get_hash(value[0], value[1])
                if has in self.close_setN:
                    end.value = 0
                    time.sleep(0.01)
                    if mid.empty():
                        my_mid = has
                        mid.put(my_mid)
                    break
            else:
                self.curNodeP, self.open_setP, self.close_setP, self.explore_pathP = self.step(
                    self.curNodeP, self.open_setP, self.close_setP, self.explore_pathP, False)
                curPq[0], curPq[1] = self.curNodeP.x, self.curNodeP.y
                value = [curNq[0], curNq[1]]
                has = get_hash(value[0], value[1])
                if has in self.close_setP:
                    end.value = 0
                    if mid.empty():
                        my_mid = has
                        mid.put(my_mid)
                    break
        # 通过节点寻找路径
        if my_mid is None:
            my_mid = mid.get()
        if inverse:
            close_set, open_set, explore_path = self.close_setN, self.open_setN, self.explore_pathN
        else:
            close_set, open_set, explore_path = self.close_setP, self.open_setP, self.explore_pathP
        # 回溯
        result = []
        curNode = close_set[my_mid]
        Length = np.array([curNode.g_score])
        while curNode:
            result.append([curNode.x, curNode.y])
            curNode = curNode.father
        np.savez(os.path.join(self.data_dir, str(inverse) + '.npz'), length=Length, set=np.array(result),
                 explore=np.array(explore_path))
        print('Write %s done!' % inverse)

    def plot_process(self):
        '''
        画出路径动态图
        '''
        filename = os.path.join(self.result_dir, 'result.npz')
        self.map.plot_precess(filename,10000)

    def report(self, time0, name):
        # 对规划的每一个路径进行汇总结果
        path = np.load(os.path.join(self.result_dir, 'result.npz'))
        final_path, explore_path, length = path['final'], path['explore'], path['length']
        filename = os.path.join(sys.path[0], 'Results\%s.txt' % name)
        text = '总时间：%ds,路径总长度：%d,探索点个数：%d\n' % (int(time.time() - time0),
                                                int(length), int(len(explore_path)))
        print(text)
        # 保存路径规划结果
        with open(filename, 'w+') as f:
            f.write(text)
        # 保存路径规划图像
        self.map.plot_map(final_path)
        plt.savefig(os.path.join(sys.path[0], 'Results\%s.png' % name))
        # 保存探索结果
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(map_data)
        # 画出探索路径
        ax.scatter(explore_path[:, 1], explore_path[:, 0], s=2, c='g', alpha=0.5)
        # 画出最终路径
        ax.plot(final_path[:, 1], final_path[:, 0], 'r')
        # 画出起始点和终点
        ax.scatter(final_path[0, 1], final_path[0, 0], s=40, marker='*', c='r')
        ax.scatter(final_path[-1, 1], final_path[-1, 0], s=40, marker='*', c='r')


class Node:
    '''
    每个点的定义
    输入父节点、输入横纵坐标
    保存到起始点的距离和启发函数
    '''

    def __init__(self, father, x, y):
        self.father = father  # 父节点
        self.x = x
        self.y = y
        self.g_score = 0  # 起始点到当前位置的距离
        self.h_score = 0  # 到目标点的距离
        self.f_score = None  # 每个点的价值

    def hash(self):
        # 计算每一个值对应的hash值
        return get_hash(self.x, self.y)

    def set_f(self, value):
        self.f_score = value

    def set_g(self, value):
        self.g_score = value

    def set_h(self, value):
        self.h_score = value


def get_hash(x, y):
    '''
    输入横纵坐标得到Hash值
    '''
    return str(x * 20000 + y)

if __name__ == '__main__':
    # 1 读取数据
    number = 4650
    data_dir = os.path.join(sys.path[0], 'Data')
    map_data = np.load(os.path.join(data_dir, str(number) + 'm_small.npy'))

    # 2 定义起点终点，然后生成图
    read_position = [[500, 500, 200, 600], [1100, 460, 1150, 360], [500, 500, 500, 2500],
                     [2355, 2430, 2000, 4000], [1140, 1870, 820, 3200], [1500, 20, 2355, 2430]]
    # 起点终点备选
    read =  4 # 规划数据，选择对那一组测试
    start_point = read_position[read][: 2]
    end_point = read_position[read][2:]
    my_map = Map(map_data, start_point, end_point)


    # 与蚁群算法对比
    # 1 生成数据
    map_data = get_data(40, 40, 0.1)
    # 2 定义起始点和目标点生成图
    start_point = [0, 0]
    end_point = [38, 34]
    my_map = Map(map_data, start_point, end_point)

    # 3 定义算法
    model = GraphSearch(my_map, alg='D', mode=1)

    # 4 运行算法，得到结果展示
    time0 = time.time()
    print('起始点(%d,%d)，目标点(%d,%d)，开始规划：' % (*start_point, *end_point))
    final_path, explore_path, info = model.find_path()
    # model.map.plot_map(final_path)
    model.plot_process()
    # model.report(time0, 'play')
    plt.show()
