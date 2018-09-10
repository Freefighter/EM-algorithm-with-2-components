import numpy as np
import EM_algo
from importlib import reload
reload(EM_algo)

if __name__ == "__main__":

    # def Theta_init():
        # beta_1_p = (np.random.random([3, 1]) - 0.5) * 10; 
        # beta_2_p = (np.random.random([3, 1]) - 0.5) * 10
        # gamma_p = (np.random.random([3, 1]) - 0.5) * 10;
        # sigma_p = 2
        
        # return np.vstack([beta_1_p, beta_2_p, gamma_p, sigma_p])
        
    # beta_1 = np.array([-1, 2, 3]).reshape(-1,1); 
    # beta_2 = np.array([2, -3, 4]).reshape(-1,1);
    # gamma = np.array([2, -1, -1]).reshape(-1,1); 
    # sigma = 0.5
    
    def Theta_init():
        beta_1_p = ((np.random.random([3, 1])) - 0.5)/10;
        beta_2_p = ((np.random.random([3, 1])) - 0.5)/10;
        gamma_p = ((np.random.random([3, 1])) - 0.5)/10;
        
        return np.vstack([beta_1_p, beta_2_p, gamma_p])
    
    beta_1 = np.array([-1, 1, 1]).reshape(-1,1);
    beta_2 = np.array([1.5, -1, 0.5]).reshape(-1,1);
    gamma = np.array([2, -1.5, -0.5]).reshape(-1,1)
    
    proc = EM_algo.EM_Algorithm(distribution="Poisson")
    
    # proc.ans = np.vstack((beta_1, beta_2, gamma, sigma))
    # proc.generateData((500, 3, 3), (beta_1, beta_2, gamma, sigma))
    proc.ans = np.vstack((beta_1, beta_2, gamma))
    proc.generateData((1000, 3, 3), (beta_1, beta_2, gamma))
    
    proc.simulate(Theta_init, times=5)