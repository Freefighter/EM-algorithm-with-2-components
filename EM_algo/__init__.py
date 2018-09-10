import numpy as np
import numpy.linalg as la
import EM_algo.EM_func
from EM_algo.EM_func import f0
from EM_algo.EM_func import GaussianModel
from EM_algo.EM_func import PoissonModel
    
class EM_Algorithm(object):
    # 1. 读取数据或自己产生数据
    # 2. 设定E步，M步与似然函数 f(Y|\Theta)
    # 3. 执行模拟
    
    def __init__(self, distribution="Gaussian", data=None):
        # 如果分布为其他，需要自定义
        
        np.random.seed(2018)
        self.distribution = distribution
         
        if distribution in ["Gaussian", "Poisson"]:
            if self.distribution == "Gaussian": 
                func_module = GaussianModel
            else:
                func_module = PoissonModel
            
            self.setEStep(func_module.EStep)
            self.setMStep(func_module.MStep)
            self.setInfoMatrix(func_module.infoMatrix)
            self.setLikelihood(func_module.likelihood)
        
        else:
            print("EStep(), MStep(), infoMatrix() and likelihood() are still needed.")
            print("Please use setXXX() method.")
            
            
        if data:
            (self.X, self.Y, self.Z) = data
            (self.n, self.s) = self.X.shape
            (self.n, self.r) = self.Z.shape
        else:
            print("Data are still needed.")
            print("Please use generateData() method to experience EM Algorithm.")
        
    def setEStep(self, EStep):
        self.EStep = EStep
    
    def setMStep(self, MStep):
        self.MStep = MStep
    
    def setInfoMatrix(self, infoMatrix):
        self.infoMatrix = infoMatrix
        
    def setLikelihood(self, likelihood):
        self.likelihood = likelihood
    
    def generateData(self, shape, parameter):
        (self.n, self.s, self.r) = shape
        (beta_1, beta_2, gamma) = parameter[:3]
        
        if self.distribution == "Gaussian":
            sigma = parameter[3]
            self.X = np.random.rand(self.n, self.s)
            self.Z = np.random.rand(self.n, self.r) * 10
            
            # @ 语法适用于3.5+版本
            Delta = (np.random.rand(self.n, 1) < f0(self.X @ gamma)).astype(int)
            self.Y = np.random.normal((self.Z * (beta_1 + beta_2 @ Delta.T).T).sum(1), sigma, self.n).reshape(-1, 1)
        
        if self.distribution == "Poisson":
            
            self.X = np.random.rand(self.n, self.s) - 0.5;
            self.Z = np.random.rand(self.n, self.r)/2 + np.array([[0.2, -0.6, 0.8]]);
            
            # @ 语法适用于3.5+版本
            Delta = (np.random.rand(self.n, 1) < f0(self.X @ gamma)).astype(int)
            
            Lambda = np.exp((self.Z * (beta_1+beta_2 @ Delta.T).T).sum(1)).reshape(-1,1)
            self.Y = np.random.poisson(Lambda, (self.n, 1))

    def simulate(self, Theta_init, times=3, L=1e-6):
        # Theta_init 为待估参数的生成函数，可以随机可以常值，应由使用者提供；
        # times 模拟次数，L 收敛精度
        Theta_p = Theta_init()
        data = (self.X, self.Y, self.Z)        
        res = np.zeros((times, Theta_p.size))

        # 记录每轮不同初值产生的结果!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        for k in np.arange(times):
            Theta_p = Theta_init()
            c = 0
            while c < 1000: # 有时候不收敛，强制退出；100次太小
                
                last = Theta_p.flatten()
            
                W = self.EStep(Theta_p, data)
                # print(c, W)
                Theta_p = self.MStep(W, Theta_p, data)
                
                # print(Theta_p)
                res[k, :] = Theta_p.flatten()
                if np.abs(res[k, :] - last).max() < L: break; # 无穷范数
                
                c += 1
            # print(Theta_p)
            # 计算完成后，控制beta_2首元素的正负以区分两类形式

            if Theta_p[self.r] < 0:
                Theta_p[self.r:2*self.r]= -Theta_p[self.r:2*self.r]
                Theta_p[:self.r] -= Theta_p[self.r:2*self.r]
                Theta_p[2*self.r:2*self.r+self.s] = -Theta_p[2*self.r:2*self.r+self.s]
            
            res[k, :] = Theta_p.flatten()

        def chooseGuess(guesses, likelihood):
            # 不同初值收敛不同，需要比较f(Y|\Theta)似然函数选择一个
            val = np.apply_along_axis(likelihood, 1, guesses, data)
            # 返回似然函数最大的那一组参数估计
            return val.argmax()
            
        if np.any(res.std(axis=0) > 1e-2):
            print("Multiple Local Minimum!")
            result = res[chooseGuess(res, self.likelihood)]
        else:
            result = res.mean(axis=0)
        
        ## 比较似然函数值，确认找到的解优于真值
        # print(self.likelihood(result, data), 
                # self.likelihood(self.ans, data))
        
        # 计算信息阵得方差
        W = self.EStep(Theta_p, data)
        I_Y = self.infoMatrix(W, Theta_p, data)
        var = np.diag(la.inv(I_Y))
            
        print(result)
        print(res.var(axis=0))
        print(var)
        
        # 记录数据
        self.res = res
        self.result = result
        self.var = var