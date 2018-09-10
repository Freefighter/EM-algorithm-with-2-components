# -*- coding: utf-8 -*-

' a module of Poisson Distribution in EM Algorithm '

__author__ = 'William Chen'

import numpy as np
import numpy.linalg as la
import scipy.optimize as op
from EM_algo.EM_func import *

def obj_gamma(gamma_p):
    gamma_p = gamma_p.flatten().reshape(-1,1)
    arg = X @ gamma_p;
    P = f0(arg)
    # P[P > 1 - 1e-8] = 1 - 1e-8            
    return -(np.log(1-P) + W * arg).sum()
    
def gradient_gamma_obs(gamma_p):
    gamma_p = gamma_p.flatten().reshape(-1,1)
    arg = X @ gamma_p;
    P = f0(arg)
    
    return ((P-W) * X)
def gradient_gamma(gamma_p):
    return gradient_gamma_obs(gamma_p).sum(axis=0)
    
def Hessian_gamma_gamma(gamma_p):
    gamma_p = gamma_p.flatten().reshape(-1,1)
    arg = X @ gamma_p
    P = f0(arg)
    
    return (P * X).T @ ((1-P) * X)
    
def obj_beta(beta_p):
    beta_p = beta_p.flatten().reshape(-1, 1)
    beta_1_p, beta_2_p = beta_p[:r], beta_p[r:]
        
    expression = Y * (Z @ beta_1_p + W * Z @ beta_2_p) \
      - W * np.exp(Z @ (beta_1_p + beta_2_p)) - (1 - W) * np.exp(Z @ beta_1_p)
    
    return -expression.sum()
    
def gradient_beta_obs(beta_p):
    Z_tilde_W = np.hstack([Z, W * Z])
    beta_p = beta_p.flatten().reshape(-1, 1)
    beta_1_p, beta_2_p = beta_p[:r], beta_p[r:]
    gradient = (Y*Z_tilde_W - W*np.exp(Z_tilde @ beta_p)*Z_tilde 
                                - (1-W)*np.exp(Z @ beta_1_p)*Z_zero)
    return -gradient#.reshape(-1, 1)
def gradient_beta(beta_p):
    return gradient_beta_obs(beta_p).sum(axis=0)
    
def Hessian_beta_beta(beta_p):
    beta_p = beta_p.flatten().reshape(-1, 1)
    beta_1_p, beta_2_p = beta_p[:r], beta_p[r:]
    
    return (Z_tilde.T @ (W*np.exp(Z_tilde @ beta_p)*Z_tilde)
            + Z_zero.T @ ((1-W)*np.exp(Z @ beta_1_p)*Z_zero))

def EStep(Theta_p, data):

    global X, Y, Z, n, s, r
    (X, Y, Z) = data
    (n, s, r) = getShape(data)
    (beta_1_p, beta_2_p, gamma_p) = assign(Theta_p, r, s)
    
    exponent = (np.exp(Z @ beta_1_p) * (np.exp(Z @ beta_2_p) - 1)
                - ((Z @ beta_2_p) * Y + X @ gamma_p)).ravel()
    result = np.zeros(exponent.size)
    result[exponent < 100] = 1 / (1 + np.exp(exponent[exponent < 100]))
    result[exponent >= 100] = 1 - 1 / (1 + np.exp(-exponent[exponent >= 100]))
    
    return result.reshape(-1,1)
    
def MStep(W_para, Theta_p, data):
    
    global X, Y, Z, n, s, r, W, Z_tilde, Z_zero
    W = W_para
    (beta_1_p, beta_2_p, gamma_p) = assign(Theta_p, r, s)
    Z_tilde =  np.hstack([Z, Z])
    Z_zero = np.hstack([Z, np.zeros([n, r])])
    
    #############################更新gamma
    try:
        res_gamma = op.minimize(obj_gamma, gamma_p, method="Newton-CG",
                        jac=gradient_gamma, hess=Hessian_gamma_gamma)
        # res_gamma = op.minimize(obj_gamma, gamma_p, method="Nelder-Mead")
    except:
        print(gamma_p); raise
    
    gamma_p = res_gamma.x.reshape(-1,1)
    
    #############################更新beta
    # stack 会保留维度，变成nx2x3;
    # hstack 则直接合并，变成nx6
    
    beta_p = np.vstack((beta_1_p, beta_2_p))
    # if beta_p[r] >= 0: beta_p = -beta_p
    

    temp = beta_p
    # res_beta = Newton_Raphson(beta_p, gradient_beta, Hessian_beta_beta, round=c)
    
    # beta_p = res_beta.reshape(-1,1)
    res_beta = op.minimize(obj_beta, beta_p, method="Newton-CG",
                        jac=gradient_beta, hess=Hessian_beta_beta)
    beta_p = res_beta.x.reshape(-1,1)
    # res_beta = op.minimize(obj_beta, beta_p, method="Nelder-Mead")
    # beta_p = res_beta.x.reshape(-1,1)

    return np.vstack((beta_p[:r], beta_p[r:], gamma_p))
    # 改维度s!!!!!
    
def infoMatrix(W_para, Theta_p, data):

    global X, Y, Z, n, s, r, W, Z_tilde, Z_zero
    W = W_para
    (beta_1_p, beta_2_p, gamma_p) = assign(Theta_p, r, s)

    I_Y = np.zeros((2*r + s, 2*r + s))
    beta_p = np.vstack((beta_1_p, beta_2_p))
    arg = X @ gamma_p
    P = f0(arg)
    
    #############################计算X的Hessian阵，要注意正负号
    I_Y[0:2*r, 0:2*r] = Hessian_beta_beta(beta_p)
    I_Y[2*r:, 2*r:] = Hessian_gamma_gamma(gamma_p)
    
    # print(I_Y)
    #############################导向量的期望（不汇总形式），这里不用注意正负号
    S_beta = gradient_beta_obs(beta_p)
    S_gamma = gradient_gamma_obs(gamma_p)
    
    S_theta = np.hstack([S_beta, S_gamma])
    S_theta_sum = S_theta.sum(axis=0).reshape((-1,1))
    
    I_Y -= ((S_theta_sum @ S_theta_sum.T) - (S_theta.T @ S_theta)) # 第三部分
    # I_Y -= (S_theta_sum @ S_theta_sum.T) # 第三部分
    # print(I_Y, '\n', S_theta_sum, '\n', (S_theta.T @ S_theta))
    
    #############################导向量乘积的期望
    S_beta_beta = (Z_tilde.T @ (W*(Y-np.exp(Z_tilde @ beta_p))**2 * Z_tilde)
            + Z_zero.T @ ((1-W)*(Y-np.exp(Z @ beta_1_p))**2 * Z_zero))

    S_beta_gamma = (W * (Y-np.exp(Z_tilde @ beta_p)) *
                    (1-P) * Z_tilde  + (1-W) * 
                    (Y-np.exp(Z @ beta_1_p)) * -P * Z_zero).T @ X
                                
    S_gamma_gamma = ((W - 2*W*P + P**2) * X).T @ X
    
    I_Y[0:2*r, 0:2*r] -= S_beta_beta
    I_Y[0:2*r, 2*r:] -= S_beta_gamma
    I_Y[2*r:, 0:2*r] -= S_beta_gamma.T
    I_Y[2*r:, 2*r:] -= S_gamma_gamma
    
    # I_Y += S_theta.T @ S_theta
    
    return I_Y

def likelihood(Theta_p, data):
    
    (X, Y, Z) = data
    (n, s, r) = getShape(data)
    (beta_1_p, beta_2_p, gamma_p) = assign(Theta_p, r, s)
    
    arg = X @ gamma_p
    P = f0(arg)
    
    lambda0 = np.exp(Z @ beta_1_p)
    lambda1 = np.exp(Z @ (beta_1_p + beta_2_p))
    d0 = lambda0 ** Y * np.exp(-lambda0) * (1-P)
    d1 = lambda1 ** Y * np.exp(-lambda1) * P
    
    return np.log(d0+d1).sum()