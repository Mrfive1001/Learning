import numpy as np
import math
import multiprocessing as mp
import time


def job(q):
    res = 0
    for i in range(10000000):
        res += i + i ** 2 + i ** 3
    return res + q


if __name__ == '__main__':
    time.clock()
    num = 10
    for i in range(num):
        job(i)
    t1 = time.clock()
    print(t1)
    num_core = mp.cpu_count() - 1  # 多核操作
    pool = mp.Pool(processes=num_core)
    jobs = [pool.apply_async(job, (ww,)) for ww in range(num)]  # 并行运算
    wws_fitness = np.array([j.get() for j in jobs])
    t2 = time.clock()-t1
    print(t2)