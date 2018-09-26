import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from dnn import DNN

'''
构建神经网络来预测协态变量和时间
'''


def main():
    # 读取数据
    path = os.path.join(sys.path[0], 'Data')
    path_data = os.path.join(path, 'all_samples_original.npy')
    data = np.load(path_data)
    # 使用X和Y
    X = data[:, :2].copy()
    Y = data[:, 2:].copy()
    



if __name__ == '__main__':
    main()
