# -*- coding: utf-8 -*-

' a module of common function in EM Algorithm '

__author__ = 'William Chen'

import numpy as np
import numpy.linalg as la
import scipy.optimize as op

# 提供高斯与泊松分布的相关函数
def f0(x):
    return 1 / (1 + np.exp(-x))
    
def f0_inv(y):
    # 从概率得到线性预测子的值
    return -np.log(1/y - 1)
    
def isPositiveDefinite(A):
    (m, n) = A.shape
    if m != n: return False
    
    for i in np.arange(m):
        det = la.det(A[:i, :i])
        # print(det)
        if det < 1e-10: return False
    
    return True

def assign(Theta_p, r, s):
    return (Theta_p[:r].reshape(-1,1), 
        Theta_p[r:2*r].reshape(-1,1), 
        Theta_p[2*r:2*r+s].reshape(-1,1))
        
def getShape(data):
    (X, Y, Z) = data
    (n, s) = X.shape
    (n, r) = Z.shape
    return (n, s, r)
