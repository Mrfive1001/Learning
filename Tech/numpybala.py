import numpy as np

# 生成数据
data1 = np.arange(0,10,0.01)
data2 = np.linspace(0,10,100)
data3 = np.array([1,0,0,4])
data4 = np.random.rand(10,3)
print(data1,data2,data3,data4)

# 数据类型
print(data1.shape)
print(data2.dtype)

# 转换数据类型，统计
data1 = data1.astype(np.float32)
print(data3.mean())
print(data4.min())

# 索引和切片
print(data4[3,1])
print(data4[data4>0.5])