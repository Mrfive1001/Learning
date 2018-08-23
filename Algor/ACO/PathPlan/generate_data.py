import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='dark')

def get_data(m=10, n=10,rate = 0.1):
    data = np.ones((m,n))*255
    obstacle_num = int(rate*m*n)
    for _ in range(obstacle_num):
        data[np.random.randint(m),np.random.randint(n)] = 0
    return data

def main():
    data = get_data(20,20,0.05)
    plt.imshow(data)
    plt.xticks([])  #去掉横坐标值
    plt.yticks([])  #去掉纵坐标值  
    plt.show()
if __name__ == '__main__':
    main()