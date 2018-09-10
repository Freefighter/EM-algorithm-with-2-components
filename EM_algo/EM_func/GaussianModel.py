# -*- coding: utf-8 -*-

' a module of Gaussian Distribution in EM Algorithm '

__author__ = 'William Chen'

import numpy as np
import numpy.linalg as la
import scipy.optimize as op
from EM_algo.EM_func import *

def EStep(Theta_p, data):
    
    (X, Y, Z) = data
    (n, s, r) = getShape(data)
    (beta_1_p, beta_2_p, gamma_p) = assign(Theta_p, r, s)
    sigma_p = Theta_p[-1]
    
    d0 = (Y - Z @ beta_1_p); d0 = d0 * d0 / (-2*sigma_p**2)
    d1 = (Y - Z @ (beta_1_p + beta_2_p)); d1 = d1 * d1 / (-2*sigma_p**2)
    
    p0 = np.exp(d0)
    p1 = np.exp(d1 + X @ gamma_p)
    denominator = p0+p1
    if np.any(denominator == 0.0):
        denominator[np.where(denominator == 0.0)] = 1
    
    return p1 / denominator
    
def MStep(W, Theta_p, data):
    
    global X, Y, Z
    (X, Y, Z) = data
    (n, s, r) = getShape(data)
    (beta_1_p, beta_2_p, gamma_p) = assign(Theta_p, r, s)
    sigma_p = Theta_p[-1]
    
    #############################更新gamma
    def obj_gamma(gamma_p):
        """
        Digging into the code a bit.  minimize calls optimize._minimize._minimize_slsqp. One of the first things it does is:

        x = asfarray(x0).flatten()

        So you need to design your objFunc to work with the flattened version of w. It may be enough to reshape it at the start of that function.
        """
        gamma_p = gamma_p.flatten().reshape(-1, 1)
        arg = X @ gamma_p
        p = f0(arg)
        return -(np.log(1-p) + W*arg).sum()

    res_gamma = op.minimize(obj_gamma, gamma_p)
            
    #############################更新beta
    # stack 会保留维度，变成nx2x3;
    # hstack 则直接合并，变成nx6
    Z_tilde = np.hstack([Z, W * Z])
    A = np.sqrt(W - W*W) * np.hstack([np.zeros([n, r]), Z])
    beta_tilde = la.inv(Z_tilde.T @ Z_tilde + A.T @ A) @ Z_tilde.T @ Y
    
    #############################更新sigma
    def obj_beta(A, beta_tilde, Z_tilde):
        tmp_a = Y - Z_tilde @ beta_tilde; tmp_b = A @ beta_tilde
        return (tmp_a.T @ tmp_a + tmp_b.T @ tmp_b)[0, 0]
    
    return np.vstack((beta_tilde[:r], beta_tilde[r:], 
      res_gamma.x.reshape(-1,1), np.sqrt(obj_beta(A, beta_tilde, Z_tilde)/n)))

      
def infoMatrix(W, Theta_p, data):

    (X, Y, Z) = data
    (n, s, r) = getShape(data)
    (beta_1_p, beta_2_p, gamma_p) = assign(Theta_p, r, s)
    sigma_p = Theta_p[-1]

    #############################计算X的二阶阵
    I_Y = np.zeros((2*r + s + 1, 2*r + s + 1))
    
    beta_tilde = np.vstack([beta_1_p, beta_2_p])
    Z_tilde = np.hstack([Z, W * Z])
    Z_square = Z.T @ Z
    Z_square_W = Z.T @ (W*Z)
    
    P = f0(X @ gamma_p) # delta为1的概率
    
    A = np.sqrt(W - W*W) * np.hstack([np.zeros([n, r]), Z])
    tmp_a = Y - Z_tilde @ beta_tilde; tmp_b = A @ beta_tilde
    # print((tmp_a * tmp_a + tmp_b * tmp_b).sum()/sigma_p**2)
    # 二阶阵的期望
    Hessian_beta_beta = np.vstack([np.hstack([Z_square, Z_square_W]), \
                            np.hstack([Z_square_W, Z_square_W])]) / -sigma_p**2
    
    Hessian_beta_sigma = -2*((Y * Z_tilde).sum(axis=0) / sigma_p**3).reshape(-1,1) - 2 * Hessian_beta_beta @ beta_tilde / sigma_p
                            
    Hessian_gamma_gamma = - (P * (1 - P) * X).T @ X
    
    Hessian_sigma_sigma = -2*n / (sigma_p**2)

    I_Y[0:2*r, 0:2*r] = -Hessian_beta_beta; I_Y[2*r:2*r+s, 2*r:2*r+s] = -Hessian_gamma_gamma
    I_Y[2*r+s:, 2*r+s:] = -Hessian_sigma_sigma
    I_Y[0:2*r, 2*r+s:] = -Hessian_beta_sigma; I_Y[2*r+s:, 0:2*r] = -Hessian_beta_sigma.T
    
    # 导向量的期望
    # S_beta = Hessian_beta_sigma * (-2*sigma_p) # 求和形式
    S_beta = np.hstack([tmp_a * Z, (W*Z) * (Y - np.hstack([Z, Z]) @ beta_tilde)]) / sigma_p**2
    
    S_gamma = (-P + W) * X
    
    # S_sigma = ((1/n - 1) / sigma_p).reshape(-1, 1) # 求和形式
    S_sigma = (tmp_a * tmp_a + tmp_b * tmp_b)/(sigma_p**3) - 1/sigma_p
    
    S_theta = np.hstack([S_beta, S_gamma, S_sigma])
    S_theta_sum = S_theta.sum(axis=0).reshape((-1,1))
    
    I_Y -= ((S_theta_sum @ S_theta_sum.T) - S_theta.T @ S_theta) # 第三部分
    # I_Y -= ((S_theta_sum @ S_theta_sum.T)) # 第三部分
    
    # 导向量乘积的期望
    a = Y - Z @ beta_1_p; b = Z @ beta_2_p
    d = tmp_a * tmp_a + tmp_b * tmp_b
    Z_square_d = (d*Z).T @ Z
    Z_square_d_W = ((a-b)**2 * Z).T @ (W*Z)
    
    S_beta_beta = np.vstack([np.hstack([Z_square_d, Z_square_d_W]), \
                            np.hstack([Z_square_d_W, Z_square_d_W])]) / (sigma_p**4)                        

    S_beta_gamma = np.vstack([(((W - P)*a - (W - W*P)*b) * Z).T @ X, \
                                (((W - W*P)*(a - b)) * Z).T @ X])/sigma_p**2
                                
    S_beta_sigma = np.hstack([(((b*W - a) / sigma_p**3 + (W*(a - b)**3 + (1-W) * a**3)/sigma_p**5) * Z), \
                            (W*(a-b)*(-1/sigma_p**3 + (a-b)**2/sigma_p**5) * Z)]).sum(axis=0).reshape((-1,1))
                                
    S_gamma_gamma = ((W - 2*W*P + P**2) * X).T @ X
    
    S_gamma_sigma = ((-(W-P) / sigma_p + (W * (a-b)**2 - P*d) / (sigma_p**3)) * X).sum(axis=0).reshape((-1,1))
    
    S_sigma_sigma = ((W * (a - b)**4 + (1-W) * a**4)).sum()/sigma_p**6 \
                        - n/sigma_p**2 # 注意最后不是矩阵
    
    I_Y[0:2*r, 0:2*r] -= S_beta_beta;
    I_Y[2*r:2*r+s, 2*r:2*r+s] -= S_gamma_gamma
    I_Y[2*r+s:, 2*r+s:] -= S_sigma_sigma
    
    I_Y[0:2*r, 2*r:2*r+s] -= S_beta_gamma; I_Y[2*r:2*r+s, 0:2*r] -= S_beta_gamma.T
    I_Y[0:2*r, 2*r+s:] -= S_beta_sigma; I_Y[2*r+s:, 0:2*r] -= S_beta_sigma.T
    I_Y[2*r:2*r+s, 2*r+s:] -= S_gamma_sigma; I_Y[2*r+s:, 2*r:2*r+s] -= S_gamma_sigma.T
    
    # I_Y += S_theta.T @ S_theta
    
    return I_Y
    
def likelihood(Theta_p, data):
    (X, Y, Z) = data
    (n, s, r) = getShape(data)
    (beta_1_p, beta_2_p, gamma_p) = assign(Theta_p, r, s)
    sigma_p = Theta_p[-1]
    
    d0 = (Y - Z @ beta_1_p)**2 / (-2*sigma_p**2)
    d1 = (Y - Z @ (beta_1_p + beta_2_p))**2 / (-2*sigma_p**2)
    
    p0 = np.exp(d0)
    p1 = np.exp(d1 + X @ gamma_p)
    p = (p0+p1)/(1+np.exp(X @ gamma_p))
    # print(d0.shape, p0.shape, p1.shape)
    
    return np.log(p).sum() - n*np.log(sigma_p)
