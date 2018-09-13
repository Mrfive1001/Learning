import numpy as np
import gym
from dnn import DNN
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()
ratio = 1e4
simulation_step = 0.1  # 仿真步长


def step_my(X, u, net):
    '''
    输入状态量、控制量和网络
    返回下一个状态
    '''

    u = min(max(u, -1.0), 1.0)
    x, v = X

    dot = get_dot(X)/ratio
    # 预测导数
    v_dot = 0.0015 * u + dot[0]
    v = v + v_dot * simulation_step
    v = min(max(v, -0.07), 0.07)

    x = x + simulation_step * v
    x = min(max(x, -1.2), 0.6)
    X = np.array([x, v])
    if X.ndim == 2:
        X = X.reshape((2,))
    return X


def get_dot(X):
    return net.predict(X)[0]


def get_target(X, X_new, u):
    return (((X_new-X)/simulation_step)[1]-u*0.0015)*ratio


# 定义环境和要保存画图的变量
env = gym.make("MountainCarContinuous-v0")
train = 0
net = DNN(2, 1, 20, name='MyModel', train=train,
          memory_size=1000, batch_size=200)
record_tru = []
record_pre = []
record_mu = []
record_mu_tru = []
# 初始化
observation = env.reset()
if train:
    for big_epi in range(50):
        observation = env.reset()
        for epi in range(50000):
            u = env.action_space.sample()
            observation_old = observation.copy()
            observation, reward, done, info = env.step(u)
            target = get_target(observation_old, observation, u)
            net.store_sample(observation_old, target)
            if epi % 100 == 0:
                result = net.learn()
                if result:
                    print(big_epi, epi, result)
    net.store_net()
observation = env.reset()
my_observation = observation
for epi in range(20000):
    u = env.action_space.sample()
    my_observation = step_my(my_observation, u, net)
    observation, reward, done, info = env.step(u)
    # 保存得到的结果
    record_mu_tru.append(-math.cos(observation[0] * 3) * 0.0025)
    record_mu.append(get_dot(my_observation))
    record_tru.append(observation)
    record_pre.append(my_observation)
record_tru = np.array(record_tru)
record_pre = np.array(record_pre)
record_mu = np.array(record_mu)
record_mu_tru = np.array(record_mu_tru)
x_ = np.arange(len(record_pre))
fig = plt.figure()
axis = 0
plt.plot(x_, record_tru[:, axis], label='x_tru')
plt.plot(x_, record_pre[:, axis], label='x_pre')
fig = plt.figure()
axis = 1
plt.plot(x_, record_tru[:, axis], label='v_tru')
plt.plot(x_, record_pre[:, axis], label='v_pre')
fig = plt.figure()
plt.plot(x_, record_mu, label='mu_pre')
plt.plot(x_, record_mu_tru, label='mu_tru')
plt.legend()
plt.show()
