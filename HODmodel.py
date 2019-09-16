import numpy as np
from scipy.special import erfc

filename = "/mnt/data1/MDhalos.npy"

def load(fin):
    """
    file has 6 columns: Mass of Halo, Radius1 (scale radius), Radius 2 (virial radius), x, y, z
    
    Output:
    a dictionary containing mass,r1,r2,x,y,z
    """
    fout = {}
    _file = np.load(fin)
    fout["mass"] = _file[:,0]
    fout["r1"] = _file[:,1]
    fout["r2"] = _file[:,2]
    fout["x"] = _file[:,3]
    fout["y"] = _file[:,4]
    fout["z"] = _file[:,5]
    
    return fout
 
 
class Occupy:
    def __init__(self,M,HODpar):
        self.M = M
        self.M_cut = HODpar["M_cut"]
        self.sigma = HODpar["sigma"]
        self.kappa = HODpar["kappa"]
        self.M1 = HODpar["M1"]
        self.alpha = HODpar["alpha"]
         
    def central(self):
        _Ncen = 0.5*erfc(np.log(10**self.M_cut/self.M)/(np.sqrt(2)*self.sigma))
        return np.random.binomial(1,_Ncen)
        
    def satellite(self):
        _Nsat = Occupy.central(self)*((self.M - self.kappa*(10**self.M_cut))/10**self.M1)**self.alpha
        return np.random.poisson(_Nsat)
         
par = {"M_cut": 13., "sigma": 0.98, "kappa": 1.13 , "M1": 14., "alpha" : .9}

def main():
    occupy = Occupy(1e15,par)
    print (occupy.central())
    print (occupy.satellite())
    fout = (load(filename))
    print (fout.keys())
    
if __name__ == "__main__":
    main()    
