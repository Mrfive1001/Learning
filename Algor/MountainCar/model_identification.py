import numpy as np
import gym
from dnn import DNN
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def step_my(X, u, net, mu = 0):
    '''
    输入状态量和控制量，误差和网络
    返回下一个状态
    '''
    simulation_step = 0.1  # 仿真步长
    u = min(max(u, -1.0), 1.0)
    x, v = X
    dot = net.predict(X)[0]

    # 预测导数
    v_dot = 0.0015 * u + dot[0] + mu
    target = dot[0] + mu
    v = v + v_dot * simulation_step
    v = min(max(v, -0.07), 0.07)

    x = x + simulation_step * v
    x = min(max(x, -1.2), 0.6)
    X = np.array([x, v])
    if X.ndim == 2:
        X = X.reshape((2,))
    return X, target


def get_mu(mu, X1, X2, k=0.1):
    '''
    通过两个状态得到导数之间的误差
    '''
    dot = k * (X1 - X2)
    simulation_step = 0.1  # 仿真步长
    mu = mu + simulation_step * dot
    return mu[1]


env = gym.make("MountainCarContinuous-v0")
train = 1
net = DNN(2, 1, 20, name='MyModel', train=train, memory_size=200)
true_x = []
pre_x = []
if train:
    record_mu = []
    for w in range(2):
        mu = 0
        observation = env.reset()
        for epi in range(5000):
            u = env.action_space.sample()
            my_observation, target = step_my(observation, u, net,mu)
            observation, reward, done, info = env.step(u)
            mu = get_mu(mu, observation, my_observation)
            net.store_sample(observation, target)
            record_mu.append(mu)
            result = net.learn()
            if result:
                print(epi, result)
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
plt.plot(true_x, 'r')
plt.plot(pre_x, 'g')
x_ = np.arange(len(record_mu))
fig = plt.figure()

plt.plot(x_, record_mu, label='v')
plt.legend()
plt.show()
