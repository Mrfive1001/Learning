import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import sys
import threading
import multiprocessing
import pickle
from visualization import plot_path
import seaborn as sns


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
        self.f_score = self.cal_f()  # 价值

    def hash(self):
        # 计算每一个值对应的hash值
        return get_hash(self.x, self.y)

    def cal_f(self, alg='A'):
        # alpha = 0 是Dijstra算法         慢
        # alpha = 0.5 是A*算法            中
        # alpha = 1 是BFS最佳优先搜索      快
        if alg == 'A':
            alpha = 0.5
        elif alg == 'D':
            alpha = 0
        else:
            alpha = 1
        self.f_score = alpha * self.h_score + (1 - alpha) * self.g_score
        return self.f_score

    def set_g(self, value):
        self.g_score = value

    def set_h(self, value):
        self.h_score = value


def get_hash(x, y):
    '''
    输入横纵坐标得到Hash值
    '''
    return str(x * 20000 + y)


class Map:
    '''
    定义地图对象
    输入地图数据，起始点坐标，结束点坐标

    '''

    def __init__(self, map_data, start_position, end_position, alg='A'):
        self.data = map_data
        # 读取地图数据 2维array 255表示能够通过 0表示不能通过
        self.start = start_position
        self.end = end_position
        self.go_on = True
        self.alg = alg

    def is_valid(self, position):
        # 判断点是否有效，是否在边界内
        if position[0] < 0 or position[0] >= self.data.shape[0] or position[1] < 0 or position[1] >= self.data.shape[1]:
            return False
        if self.data[position[0], position[1]] == 0:
            return False
        return True

    def cal_h(self, x, y, inverse=False):
        # 计算x,y距离目标的距离
        # inverse 表示是否需要将起始点与终点互换
        if inverse:
            return ((x - self.start[0]) ** 2 + (y - self.start[1]) ** 2) ** 0.5
        else:
            return ((x - self.end[0]) ** 2 + (y - self.end[1]) ** 2) ** 0.5

    def plot_map(self, path=None):
        # 画出地图，如果规划的路径不为空的话，将其画出来
        # 注意地图中x与y是反过来的，横坐标是y
        plt.imshow(self.data)
        if path is not None:
            plt.plot(path[:, 1], path[:, 0], 'r')
            plt.scatter([self.start[1], self.end[1]],
                        [self.start[0], self.end[0]], s=40, marker='*', c='r')

    def step(self, curNode, open_set, close_set, explore_path, inverse):
        '''
        实现当前点到下一个点规划的一步转换
        输入：当前点 开启列表 关闭列表 探索过的路径 是否起终点互换
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
                if self.is_valid([node.x, node.y]) == False or node.hash() in close_set:
                    continue
                else:
                    # 2 开启列表中已存在
                    add_value = ((i - 1) ** 2 + (j - 1) ** 2) ** 0.5
                    if node.hash() in open_set:
                        node = open_set[node.hash()]
                        if (add_value + curNode.g_score) < node.g_score:
                            node.set_g(add_value + curNode.g_score)
                            node.father = curNode
                            node.cal_f(self.alg)
                    # 3 开启列表中未存在
                    else:
                        node.set_g(add_value + curNode.g_score)
                        node.set_h(self.cal_h(node.x, node.y, inverse))
                        node.cal_f(self.alg)
                        open_set[node.hash()] = node
                    # 选择出下一个点
                    # 找到最小f值的点如果找不到将执行下一步
                    if nextNode is None:
                        nextNode = node
                        nextF = node.cal_f(self.alg)
                    else:
                        if nextF > node.cal_f(self.alg):
                            nextNode = node
                            nextF = node.cal_f(self.alg)
        # 选出下一个要探索的点
        if nextNode is None:
            for val in open_set.values():
                if nextNode:
                    if val.cal_f(self.alg) < nextNode.cal_f(self.alg):
                        nextNode = val
                        nextF = val.cal_f(self.alg)
                else:
                    nextNode = val
                    nextF = nextNode.cal_f(self.alg)
        curNode = nextNode
        info = (curNode, open_set, close_set, explore_path)
        return info

    def loop_two_multhr(self, inverse):
        # 多线程求解
        while self.go_on:
            if inverse:
                # 对反向的进行迭代判断
                self.curNodeN, self.open_setN, self.close_setN, self.explore_pathN = self.step(
                    self.curNodeN, self.open_setN, self.close_setN, self.explore_pathN, True)
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

    def loop_two_mulpro(self, obj):
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
        dir = sys.path[0]
        np.savez(os.path.join(dir, str(inverse) + '.npz'), length=Length, set=np.array(result),
                 explore=np.array(explore_path))
        print('Write %s done!' % inverse)
        return inverse

    def two_path(self, mode):
        start_nodeP = Node(None, self.start[0], self.start[1])  # 正向起点
        start_nodeN = Node(None, self.end[0], self.end[1])  # 反向起点
        self.open_setP = {start_nodeP.hash(): start_nodeP}
        self.close_setP = {}
        self.open_setN = {start_nodeN.hash(): start_nodeN}
        self.close_setN = {}
        self.curNodeP = start_nodeP
        self.explore_pathP = []
        self.curNodeN = start_nodeN
        self.explore_pathN = []
        self.mid_node = None
        # 从头尾两个方面进行探索

        if mode == 2:
            # 多进程处理
            end = multiprocessing.Value('i', 1)
            curN = multiprocessing.Array('i', [start_nodeN.x, start_nodeN.y])
            curP = multiprocessing.Array('i', [start_nodeP.x, start_nodeP.y])
            mid = multiprocessing.Queue()
            obj1 = (True, end, curN, curP, mid)
            obj2 = (False, end, curN, curP, mid)
            # pool = multiprocessing.Pool(2)
            # res = pool.map(self.loop_two_mulpro, [obj1, obj2])

            t1 = multiprocessing.Process(target=self.loop_two_mulpro, args=(obj1,))
            t2 = multiprocessing.Process(target=self.loop_two_mulpro, args=(obj2,))
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            dir = sys.path[0]
            res1 = np.load(os.path.join(dir, str(True) + '.npz'))
            res2 = np.load(os.path.join(dir, str(False) + '.npz'))

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
        elif mode == 3:
            # 多进程
            self.lock = threading.Lock()
            # 方式1
            # t1 = threading.Thread(target=self.loop_two_multhr, args=(True,))
            # t2 = threading.Thread(target=self.loop_two_multhr, args=(False,))
            # t1.start()
            # t2.start()
            # t1.join()
            # t2.join()
            # 方式2 带返回值的
            from multiprocessing.pool import ThreadPool
            pool = ThreadPool(2)
            res = pool.map(self.loop_two_multhr, [*(True,), *(False,)])
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

    def find_path(self, mode=0):
        '''
        找到mode模式下的路径
        mode = 0 正向搜索
        mode = 1 逆向搜索
        mode = 2 双向搜索加上多进程
        mode = 3 双向搜索加上多线程
        输出 最终路径、走过的路径、路径总长度
        '''

        if mode >= 2:
            return self.two_path(mode)
        else:
            if mode == 1:
                # 起点终点交换
                temp = self.end.copy()
                self.end = self.start.copy()
                self.start = temp.copy()

            # 从开始点进行计算
            start_node = Node(None, self.start[0], self.start[1])
            end_node = Node(None, self.end[0], self.end[1])
            open_set = {start_node.hash(): start_node}  # 开启列表加入的是周围未经过完全探索的点
            close_set = {}  # 关闭列表是经过完全探索的点
            curNode = start_node
            explore_path = []  # 所有经过的点,与close_set是等价的
            # 往前探索
            while True:
                curNode, open_set, close_set, explore_path = self.step(
                    curNode, open_set, close_set, explore_path, False)
                if end_node.hash() in close_set:
                    break
            # 回溯
            result = []
            curNode = close_set[end_node.hash()]
            info = {}
            info['total_length'] = curNode.g_score
            while curNode:
                result.append([curNode.x, curNode.y])
                curNode = curNode.father
            return np.array(result), np.array(explore_path), info


def report(final_path, explore_path, info, my_map, case_number, algorithm, time0):
    filename = os.path.join(sys.path[0], 'Results\%s.txt' % algorithm)
    text = '任务%d,总时间：%ds,路径总长度：%d,探索点个数：%d\n' % \
           (case_number, int(time.time() - time0), int(info['total_length']), int(len(explore_path)))
    file_mode = 'a+'
    if case_number == 1:
        file_mode = 'w+'
    # with open(filename, file_mode) as f:
    #     f.write(text)

    fig = plt.figure()
    my_map.plot_map(final_path)
    plt.savefig(os.path.join(sys.path[0], 'Results\%s_Case%s.png' % (algorithm, case_number)))
    plt.close()
    return text


def plot_explore(map_data, final_path, explore_path):
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


if __name__ == '__main__':
    # 选择number m深度的海域进行测试
    number = 4650
    data_dir = os.path.join(sys.path[0], 'Data')
    map_data = np.load(os.path.join(data_dir, str(number) + 'm_small.npy'))
    map_temp = Map(map_data, None, None)
    read_position = [[500, 500, 200, 600], [1100, 460, 1150, 360], [500, 500, 500, 2500],
                     [2355, 2430, 2000, 4000], [1140, 1870, 820, 3200], [1500, 20, 2355, 2430]]
    # read_position = [[500, 500, 200, 600], [1100, 460, 1150, 360], [500, 500, 500, 2500]]
    sns.set()
    read = None  # 规划数据，选择对那一组测试或者从外面读取
    mode = 0  # 正向、反向或者多任务处理
    for algorithm in ('A', 'B', 'D'):
        print('Algorithm:', algorithm, 'Planning Mode:', mode)
        if read is None:
            # 全部输出上面的图像
            pass
        else:
            if read == -1:
                # 显示当前图像
                map_temp.plot_map()
                plt.show()
                # 读入起始和结束点位置并且显示图像
                while True:
                    print("起始点设置")
                    start_x = int(input('输入x点坐标：'))
                    start_y = int(input('输入y点坐标：'))
                    if map_temp.is_valid([start_x, start_y]):
                        start_position = [start_x, start_y]
                        break
                    else:
                        print("输入点无效，重新输入")
                while True:
                    print("目标点设置")
                    end_x = int(input('输入x点坐标：'))
                    end_y = int(input('输入y点坐标：'))
                    if map_temp.is_valid([end_x, end_y]):
                        end_position = [end_x, end_y]
                        break
                    else:
                        print("输入点无效，重新输入")
            else:
                start_position = read_position[read][: 2]
                end_position = read_position[read][2:]
            # 将所要规划的路径保存下来
            read_position = [[*start_position, *end_position]]
        # 定义map并且找到最优路径
        for read_tem in range(0, len(read_position)):
            start_position = read_position[read_tem][: 2]
            end_position = read_position[read_tem][2:]
            print('任务%d，起始点(%d,%d)，目标点(%d,%d)，开始规划：' %
                  (read_tem + 1, *start_position, *end_position))
            time0 = time.time()
            my_map = Map(map_data, start_position, end_position, alg=algorithm)
            # 进行规划
            final_path, explore_path, info = my_map.find_path(mode=mode)

            # 对每个情况进行画图保存
            text = report(final_path, explore_path, info, my_map, read_tem + 1, algorithm, time0)
            print(text)

            # 将规划结果进行保存
            # dir = sys.path[0]
            # np.savez(os.path.join(dir, 'result.npz'),
            #          final=final_path, explore=explore_path)
            # plot_explore(map_data, final_path, explore_path)
            # plt.savefig(os.path.join(sys.path[0], 'Results\%s_Case%s_explor.png' % (algorithm, read_tem+1)))
            # plt.close()
            # plt.show()
