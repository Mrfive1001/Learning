import math

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import odeint
from scipy.optimize import minimize, root

from dnn import DNN
from model import MountainCar

np.random.seed(10)
sns.set()

"""
MountainCar间接法来求解
1 直接对原系统使用间接法来求解
    1.1 将求解结果放到原系统中 done
    1.2 将求解结果放到神经网络辨识得到的系统中 done
2 对神经网络辨识得到的结果使用间接法
    1.1 将求解结果放到原系统中 done
    1.2 将求解结果放到神经网络辨识得到的系统中 done
"""

class MountainCarIndirect:
    def __init__(self):
        self.env = MountainCar()
        self.state = None   # 系统状态变量
        self.reset()
        self.state_dim = len(self.state)

    def render(self):
        pass

    def reset(self):
        """
        利用原始问题的初始化，随机初始化
        """
        self.state = self.env.reset()
        return self.state

    def get_reward(self, action):
        """
        看结果的符合情况
        """
        _, ceq, _, _ = self.step(action)
        return ceq

    def get_result(self, action_ini):
        """
        得到某个初始动作优化的结果
        """
        res = root(self.get_reward, action_ini)
        return res

    def step(self, action):
        """
        进行一次初值的求解，从lambda0到lambda_n
        action：lambda_0 , t_f
        return: 返回state，ceq, done, info
        """
        lambda_0 = action[:-1]
        t_f = action[-1]
        x_goal = 0.45
        t_f = abs(t_f)

        # 微分方程
        X0 = np.hstack([self.state, lambda_0])
        t = np.linspace(0, t_f, 101)
        # X = odeint(self.motionequation, X0, t, rtol=1e-12, atol=1e-12)
        X = self.odeint(self.motionequation, X0, t)

        # 末端Hamilton函数值
        X_end = X[-1, :]
        H_end = self.motionequation(X_end, 0, end=True)
        ceq = np.array([H_end,X_end[0]-x_goal,X_end[3]])
        
        done = True
        info = {}
        info['X'] = X.copy()
        info['t'] = t.copy()
        info['ceq'] = ceq


        return self.state.copy(), ceq, done, info

    def motionequation(self, X, t, end=False):
        """
        状态方程的转移
        输入当前X，输出X_dot
        输入t
        输入end 是否求最终的几个值 
        """
        x = X[0]
        v = X[1]
        lambda_x = X[2]
        lambda_v = X[3]

        a = 0.0025
        b = 0.0015
        u = 1
        if lambda_v > 0:
            # 最优控制
            u = -u

        original = 1  # 是否使用原系统进行求解
        pred_ = -a * math.cos(3 * x)
        pred_dot = 3 * a * math.sin(3 * x)
        if original == 0:
            pred_ = float(self.env.get_dot(X[:1]))
            pred_dot = float(self.env.get_dot2(X[:1]))

        # 动态方程
        x_dot = v
        v_dot = b*u+pred_
        lambda_x_dot = -lambda_v*pred_dot
        lambda_v_dot = -lambda_x
        X_dot = [x_dot, v_dot, lambda_x_dot, lambda_v_dot]
        if end:
            H_end = 1 + lambda_x*v + lambda_v*(b*u+pred_)
            return H_end
        return np.array(X_dot)

    def odeint(self,dot_fun,X0,t_list):
        """
        直接对dot_fun进行积分
        :param dot_fun: 输入X得到X_dot
        :param X0: 积分初值
        :param t_list: 积分的中间值
        :return: X_all
        """
        result = []
        X = X0.copy()
        result.append(X)
        for t_index in range(1,len(t_list)):
            t = t_list[t_index]-t_list[t_index-1]
            dot = dot_fun(X,0)
            X += dot*t
            result.append(X)
        return np.array(result)


    def choose_action(self,result_by_indirect,observation):
        """
        根据间接法计算得到的lambda来选择动作，并返回下一个lambda
        """
        try:
            lambda_x,lambda_v,t_f = result_by_indirect
        except Exception:
            lambda_x, lambda_v = result_by_indirect
        # 选择动作
        u = 1
        if lambda_v > 0:
            u = -u
        X = np.hstack((observation,lambda_x,lambda_v))
        X_dot = self.motionequation(X,0)
        X += X_dot*self.env.simulation_step
        return np.array([u]),X[2:]


if __name__ == '__main__':
    """
    测试下模型辨识结合间接法得到的效果
    """
    env = MountainCarIndirect()
    observation = env.reset()
    # 求出间接法的结果
    for i in range(100):
        lambda_n = np.random.randn(2)*10
        t_f = np.random.rand(1) * 100
        action = np.hstack([lambda_n, t_f])
        res = env.get_result(action)
        print('step', i, 'fun', res.fun, '\n', 'action', res.x)
        if res.success:
            print('sucess')
            break

    # 应用到当前初始化的小车控制上
    original_action = res.x
    result_indirect = res.x

    observation_record = []
    observation_record_net = []

    observation_net = observation # 神经网络系统
    env.env.state = observation # 神经网络系统
    while True:
        observation_record.append(observation)
        observation_record_net.append(observation_net)
        action,result_indirect = env.choose_action(result_indirect,observation)
        observation, _, done, info = env.env.env.step(action)
        observation_net, _, done_net, info_net = env.env.step(action)
        print(observation,observation_net)
        if done:
            break
    observation_record = np.array(observation_record)
    observation_record_net = np.array(observation_record_net)

    # 显示x曲线和v曲线
    plt.figure(1)
    plt.plot(observation_record[:,0],label = 'x_ture')
    plt.plot(observation_record_net[:,0],label = 'x_pre')

    plt.xlabel('Time(s)')
    plt.ylabel('Xposition')
    plt.plot(0.45*np.ones(len(observation_record)),'r')
    plt.legend()

    plt.figure(2)
    plt.plot((observation_record[:,1]),label = 'v_ture')
    plt.plot((observation_record_net[:,1]),label = 'v_pre')
    plt.xlabel('Time(s)')
    plt.ylabel('Vspeed')
    plt.legend()
    plt.show()
