import math

import gym
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize, root
import seaborn as sns

sns.set()

class MountainCarIndirect:
    """
    MountainCar间接法来求解
    """

    def __init__(self, random=False):
        self.t = None
        self.env = gym.make("MountainCarContinuous-v0")
        self.state = None  # 与外界交互的变量
        self.x = None  # 内部积分变量
        self.simulation_step = 0.1  # 积分步长
        self.random = random  # 是否随机初始化
        self.reset()
        self.state_dim = len(self.state)

    def render(self):
        pass

    def action_sample(self):
        """
        随机选取符合环境的动作
        """
        return self.env.action_space.sample()

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
        X = odeint(self.motionequation, X0, t, rtol=1e-12, atol=1e-12)

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
            u = -u

        # dynamic equation
        x_dot = v
        v_dot = b*u-a*math.cos(3*x)
        lambda_x_dot = -3*a*lambda_v*math.sin(3*x)
        lambda_v_dot = -lambda_x

        X_dot = [x_dot, v_dot, lambda_x_dot, lambda_v_dot]
        if end:
            H_end = 1 + lambda_x*v + lambda_v*(b*u-a*math.cos(3*x))
            return H_end
        return np.array(X_dot)


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
        X += X_dot*self.simulation_step
        return np.array([u]),X[2:]


if __name__ == '__main__':
    """
    测试下直接间接法的效果
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
    t = 0
    while True:
        env.env.render()
        action,result_indirect = env.choose_action(result_indirect,observation)
        t += env.simulation_step
        observation,_,done,info = env.env.step(action)
        if done:
            break

    observation, ceq, done, info = env.step(original_action)
    print('Result:',ceq,'Time',t)

    # 展示结果
    plt.figure(1)
    plt.plot(info['t'], info['X'][:, 0])
    plt.xlabel('Time(s)')
    plt.ylabel('Xposition')
    plt.plot(0.45*np.ones(int(info['t'].max()+2)),'r')
    plt.figure(2)
    plt.plot(info['t'], info['X'][:, 1])
    plt.xlabel('Time(s)')
    plt.ylabel('Vspeed')
    plt.show()
