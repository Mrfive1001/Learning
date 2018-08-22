import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


# 读取数据并且进行显示
def load(name):
    # 输入文件名称例如4650.npy
    dir = sys.path[0]
    data_path = os.path.join(dir, name)
    data = np.load(data_path)
    print(data)
    # 得到的是二维的数据
    fig = plt.figure()
    plt.imshow(data)
    # 显示图片

if __name__ == '__main__':
    load('4650_part.npy')
    plt.show()
