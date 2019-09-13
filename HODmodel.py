import numpy as np
from scipy.special import erfc
 
 
 
class Occupy:
    def __init__(self,M,HODpar):
        self.M = M
        self.M_cut = HODpar["M_cut"]
        self.sigma = HODpar["sigma"]
        self.kappa = HODpar["kappa"]
        self.M1 = HODpar["M1"]
        self.alpha = HODpar["alpha"]
         
    def central(self):
        N_cen = erfc(np.log(self.M_cut/self.M)/(np.sqrt(2)*self.sigma))/2.0
        return N_cen
         
    def satellite(self):
        N_sat = Occupy.central(self)*((self.M - self.kappa*self.M_cut)/self.M1)**self.alpha
        return N_sat
         
par = {"M_cut": 13., "sigma": 0.98, "kappa": 1.13 , "M1": 14., "alpha" : .9}

def main():
    occupy = Occupy(10,par)
    print (occupy.central())
    print (occupy.satellite())
    
    
if __name__ == "__main__":
    main()    
