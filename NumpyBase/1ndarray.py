# -*- coding: utf-8 -*-
"""
@File   : 1ndarray.py
@Author : Pengy
@Date   : 2020/10/12
@Description : Numpy的数据类型ndarray(N-dimensional Array)
"""
import numpy as np

'''
生成ndarray的几种方式：
1.从已有数据中创建
2.利用random创建
3.创建特定形状的多维数组
4.利用arange、linspace函数生成等
'''

# 1.从已有数据中创建
list1 = [3.14, 2.17, .0, 1, 2]
nd1 = np.array(list1)
print(type(nd1))
list2 = [[3.14, 2.17, .0, 1, 2], [1, 2, 3, 4, 5]]
nd2 = np.array(list2)
print(type(nd2))

# 2.利用random创建
nd3 = np.random.random([3, 4])
print(nd3)
print("nd3的形状:", nd3.shape)
# 为了每次生成同一份数据,可以指定一个随机种子,使用shuffle函数打乱生成的随机数
np.random.seed(123)
nd4 = np.random.randn(2, 3)
print(nd4)
np.random.shuffle(nd4)
print("打乱后的数据:")
print(nd4)

# 3.创建特定形状的多维数组
nd5 = np.zeros([3, 4])  # 生成全为0的矩阵
nd5_like = np.zeros_like(nd5)  # 生成与nd5形状一样的全0矩阵
nd6 = np.ones([3, 4])  # 生成全为1的矩阵
nd7 = np.eye(3)  # 生成3阶的单位矩阵
nd8 = np.diag([1, 2, 3])  # 生成3阶对角矩阵

# 4.利用arange、linspace函数生成等
