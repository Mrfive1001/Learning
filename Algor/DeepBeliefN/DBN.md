## RBM受限玻尔兹曼机
### 用途
* 当作一种降维的方法，将数据进行编码
* 得到权重矩阵和偏移量，供BP神经网络进行训练(预训练)
### 使用方法
* RBM拥有两层网络
* 假设RBM网络有m个可视化节点，总体成为A(输入节点x)和n个隐藏节点(输出节点y)，总体称为B
* 参数：A到B的权重矩阵 A到B的偏移矩阵 B到A的偏移矩阵
* 正向传播：算出B层的值，然后利用sigmoid算出概率，之后进行采样，得到y
* 反向传播：利用得到的y同理来求出x_
* 我们的损失函数就是有三个来分别优化三个变量矩阵()
## DBN深度信念网络
### 用途
* 使训练过程更加迅速
* 神经网络能够跳出局部极值点
### 使用方法
* 逐层进行RBM训练
* 最后输出层利用bp神经网络，将最后一层的RBM训练的特征输入，得到输出
* 进行梯度求解