import math

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from model import MountainCar

sns.set() 

train = 0
env = MountainCar(train=train, name='Goodone')

record_tru = []
record_pre = []
record_mu = []
record_mu_tru = []


my_state = env.reset()
for epi in range(500):
    if epi %1 ==0:
        u = env.action_sample()
    my_state, _, _, _ = env.step(u)
    state, _, _, _ = env.env.step(u)
    # 保存得到的结果
    print(epi, (my_state-state)/state)
    record_tru.append(state)
    record_pre.append(my_state)
record_tru = np.array(record_tru)
record_pre = np.array(record_pre)
x_ = np.arange(len(record_pre))
fig = plt.figure()
axis = 0
plt.plot(x_, record_tru[:, axis], label='x_tru')
plt.plot(x_, record_pre[:, axis], label='x_pre')
plt.legend()
fig = plt.figure()
axis = 1
plt.plot(x_, record_tru[:, axis], label='v_tru')
plt.plot(x_, record_pre[:, axis], label='v_pre')
plt.legend()

a = 0.0025
x_ = np.arange(-1.1,0.6,0.01)
y_tru = 3*a*np.sin(3*x_)
y_pre = env.net.predict_dot(x_.reshape((-1,1)))[:,0]

# 验证对所有的x的拟合情况
fig = plt.figure()
plt.plot(x_, y_tru, label='x_tru')
plt.plot(x_, y_pre, label='x_pre')
plt.legend()

plt.show()
