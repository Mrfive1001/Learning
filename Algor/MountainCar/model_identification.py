import numpy as np
import gym
from dnn import DNN
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def step_my(X, u, net):
    '''
    :param X: 状态变量
    :param net:  神经网络，可以进行预测
    :return: 返回下个状态，和神经网络预测结果
    '''
    simulation_step = 0.1  # 仿真步长
    u = min(max(u, -1.0), 1.0)
    x, v = X
    dot = net.predict(X)[0]

    # 预测导数
    v_dot = 0.0015 * u + dot[0]
    v = v + v_dot * simulation_step
    v = min(max(v, -0.07), 0.07)

    x = x + simulation_step * v
    x = min(max(x, -1.2), 0.6)
    X = np.array([x,v])
    if X.ndim == 2:
        X = X.reshape((2,))
    return X,dot


def get_error(error, X1, X2, k=0.1):
    '''
    通过两个状态得到导数之间的误差
    '''
    dot = k * (X1 - X2)
    simulation_step = 0.1  # 仿真步长
    error = error + simulation_step * dot
    return error[1]


env = gym.make("MountainCarContinuous-v0")
train = 1
net = DNN(2, 1, 20, name='MyModel', train=train, memory_size=200)
true_x = []
pre_x = []
if train:
    record_error = []
    for w in range(2):
        error = 0
        observation = env.reset()
        for epi in range(5000):
            u = env.action_space.sample()
            my_observation, d = step_my(observation, u, net)
            observation, reward, done, info = env.step(u)
            error = get_error(error, observation, my_observation)
            y = d + error
            record_error.append(error)
            net.store_sample(observation, y)
            result = net.learn()
            if result:
                print(w, result)
    net.store_net()
true_x = []
pre_x = []

observation = env.reset()
for epi in range(5000):
    u = env.action_space.sample()
    my_observation, d = step_my(observation, u, net)
    observation, reward, done, info = env.step(u)
    true_x.append(observation[1])
    pre_x.append(my_observation[1])
    result = net.learn()
    if result:
        print(w, result)
plt.plot(true_x,'r')
plt.plot(pre_x,'g')
x_ = np.arange(len(record_error))
fig = plt.figure()

plt.plot(x_, record_error, label='v')
plt.legend()
plt.show()