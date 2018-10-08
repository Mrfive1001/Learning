import math
import os
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.integrate import RK45, odeint
from scipy.optimize import minimize, root

from dnn import DNN
from model import MountainCar

# np.random.seed(10)
sns.set()
original = 1  # 是否使用原系统进行求解
data_name = 'all_samples_original.npy' if original else 'all_samples_net.npy'
# 是否使用gym来验证
use_gym = 0
"""
MountainCar间接法来求解
1 直接对原系统使用间接法来求解
    1.1 将求解结果放到原系统中 done
    1.2 将求解结果放到神经网络辨识得到的系统中 done
2 对神经网络辨识得到的结果使用间接法
    1.1 将求解结果放到原系统中 done
    1.2 将求解结果放到神经网络辨识得到的系统中 done
3 保存通过打靶法得到的数据
4 验证得到的数据是否有效
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
        # Old API
        # X = odeint(self.motionequation, X0, t, rtol=1e-12, atol=1e-12)

        # New API
        ode = RK45(self.motionequation,t[0],X0,t_f)
        X = [X0]
        while True:
            ode.step()
            X.append(ode.y)
            if ode.status != 'running':
                break
        X = np.array(X)

        # X = self.odeint(self.motionequation, X0, t)

        # 末端Hamilton函数值
        X_end = X[-1, :]
        H_end = self.motionequation(0,X_end, end=True)
        ceq = np.array([H_end,X_end[0]-x_goal,X_end[3]])
        
        done = True
        info = {}
        info['X'] = X.copy()
        info['t'] = t.copy()
        info['ceq'] = ceq


        return self.state.copy(), ceq, done, info

    def motionequation(self, t, X, end=False):
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
        result.append(X.copy())
        for t_index in range(1,len(t_list)):
            t = t_list[t_index]-t_list[t_index-1]
            dot = dot_fun(X,0)
            X += dot*t
            result.append(X.copy())
        return np.array(result)

    def hit_target(self,X0):
        """
        间接法打靶方程
        :param X0: 初始状态,赋值给self.state
        :return: 打靶结果、中间状态
        """
        self.state = X0
        for i in range(100):
            lambda_n = np.random.randn(2) * 10
            t_f = np.random.rand(1) * 100
            coor = np.hstack([lambda_n, t_f])
            res = self.get_result(coor)
            print('step', i, 'fun', res.fun, '\n', 'coor', res.x)
            if res.success:
                print('sucess')
                break
        info = self.step(res.x)[-1]
        data_x = info['X'][:,0]
        success = res.success
        if data_x.min() < -1.2 or data_x.max() > 0.6:
            success = False
        return res.x,info,success

    def verity_cor(self,X0,coor,show = True):
        """
        对神经网络系统和原始系统进行协态变量验证
        :param X0: 初始状态
        :param action: 协态变量
        :return: 进行结果显示和返回打靶过程量
        """
        observation_record = []
        observation_record_net = []
        corr_record = []
        time_record = []
        # self.env.verity_net_2()

        observation_net = X0  # 神经网络系统
        self.env.state = X0  # 神经网络系统
        observation = X0 # 原系统
        time = 0
        while True:
            # self.env.env.render()
            observation_record.append(observation)
            observation_record_net.append(observation_net)
            time_record.append(time)
            corr_record.append(coor[:2])
            action, coor = self.choose_action(coor, observation_net)
            observation, _, done, info = self.env.env.step(action)
            observation_net, _, done_net, info_net = self.env.step(action)
            time += self.env.simulation_step
            # print(observation, observation_net)
            if done_net:
                break
        observation_record.append(observation)
        observation_record_net.append(observation_net)
        time_record.append(time)
        corr_record.append(coor[:2])

        observation_record = np.array(observation_record)
        observation_record_net = np.array(observation_record_net)
        corr_record = np.array(corr_record)
        time_record = np.array(time_record)
        if show:
            # 显示x曲线和v曲线
            plt.figure(1)
            if use_gym:
                plt.plot(observation_record[:, 0], label='x_ture')
            plt.plot(observation_record_net[:, 0], label='x_pre')

            plt.xlabel('Time(s)')
            plt.ylabel('Xposition')
            plt.plot(0.45 * np.ones(len(observation_record)), 'r')
            plt.legend()

            plt.figure(2)
            if use_gym:
                plt.plot((observation_record[:, 1]), label='v_ture')
            plt.plot((observation_record_net[:, 1]), label='v_pre')
            plt.xlabel('Time(s)')
            plt.ylabel('Vspeed')
            plt.legend()

        return observation_record_net,corr_record,time_record

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
        X_dot = self.motionequation(0,X)
        X += X_dot*self.env.simulation_step
        return np.array([u]),X[2:]

    def get_samples(self,epis):
        """
        保存epis轮的打靶数据
        :return: 保存到数据中
        数据中 X,V,lambda0,lambda1,tf
        """
        path = os.path.join(sys.path[0],'Data')
        record_all = None
        success_times = 0
        for epi in range(epis):
            observation = self.reset()
            print('-'*10,'Epi:',epi+1,',Begin!')
            coor, info, success = self.hit_target(observation.copy())
            print('-'*10,'Epi:',epi+1,',End!','Success:',success)
            if success:
                success_times += 1
                observation_record, coor_record, time_record = self.verity_cor(observation, coor,show=False)
                X_record = np.hstack((observation_record,coor_record))
                record = np.hstack((X_record, (time_record).reshape((-1, 1))))
                if record_all is None:
                    record_all = record
                else:
                    record_all = np.vstack((record_all, record))
        print('End!Successful target times:%d,successful rate:%f'%(success_times,success_times/epis))
        np.save(os.path.join(path,data_name),record_all)

    def verity_sample(self,name,num):
        """
        验证Data中name文件的数据是否有效
        :param name: 文件名称
        """
        path = os.path.join(sys.path[0], 'Data')
        data = np.load(os.path.join(path,name))
        index = np.random.choice(len(data),size=num)
        samples = data[index,:]
        for sample in samples:
            self.verity_cor(sample[:2],sample[2:])


if __name__ == '__main__':
    """
    测试下模型辨识结合间接法得到的效果
    """
    env = MountainCarIndirect()
    observation = env.reset()
    # 求出间接法的结果
    # coor,info,success = env.hit_target(observation.copy())
    # 应用到当前初始化的小车控制上
    # observation_record,coor_record,time_record = env.verity_cor(observation,coor)
    # 保存打靶法得到的结果
    env.get_samples(50)
    # 验证样本数据的有效性
    # env.verity_sample(data_name,num = 10)
    # plt.show()