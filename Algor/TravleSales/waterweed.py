import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from tsp import TSP


class WwTsp:
    '''
    定义为TSP优化的水草算法
    '''
    def __init__(self,
                num_ww=20,  # 水草种群数量
                max_age=10,  # 每株水草的存活年限
                num_seeds=20,  # 水草生产种子最大数量
                change_rate=0.3,  # 变异概率
                data_type = 'small' # TSP数据类型
                ):
        self.num_ww = num_ww
        self.change_rate = change_rate
        self.max_age = max_age
        self.num_seeds = num_seeds
        self.num_var = num_var
        self.bound_var = np.array(bound_var)