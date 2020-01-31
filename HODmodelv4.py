# This code is used to generate catalogs using uniform satellite distribution

import numpy as np
from scipy.special import erfc
import gc
import warnings
import random
from numba import jit
warnings.filterwarnings("ignore")

np.random.seed(42)
filename = "/mnt/data1/MDhalos.npy"
 
class Occupy:
    def __init__(self,HODpar,fin):
        self.fout = self.load(fin)
        self.M = self.fout["mass"]
        self.M_cut = HODpar["M_cut"]
        self.sigma = HODpar["sigma"]
        self.kappa = HODpar["kappa"]
        self.M1 = HODpar["M1"]
        self.alpha = HODpar["alpha"]
    
    def load(self,fin):
        """
        File has 6 columns: Mass of Halo, Radius1 (scale radius), Radius2 (virial radius), x, y, z
    
        Output:
        A dictionary containing "mass", "r1", "r2", "x", "y", "z"
        """
        out = {}
        __file = np.load(fin)
        out["mass"] = __file[:,0]
        out["r1"] = __file[:,1]
        out["r2"] = __file[:,2]
        out["xyz"] = __file[:,(3,4,5)]
        __file = [None]
        return out

    def central(self):
        """
        Returns 1 if there's a central galaxy. 
        Distribution found using eq.12 of https://iopscience.iop.org/article/10.1088/0004-637X/728/2/126/pdf
        """
        _Ncen = 0.5*erfc(np.log10(10**self.M_cut/self.M)/(np.sqrt(2)*self.sigma))
        return np.random.binomial(1,_Ncen)

    def satellite(self):
        """
        Returns Poisson distribution with mean _Nsat.
        Distribution found using eq.13 of https://iopscience.iop.org/article/10.1088/0004-637X/728/2/126/pdf
        """
        _Nsat = Occupy.central(self)*((self.M - self.kappa*(10**self.M_cut))/10**self.M1)**self.alpha
        _Nsat[np.where(np.isnan(_Nsat))] = 0
        _Nsat[np.where(_Nsat < 1)] = 0
        return np.random.poisson(_Nsat)

class Coordinates(Occupy):
    def __init__(self,HODpar,fin): 
        super().__init__(HODpar,fin)
        
    def cen_coord(self):
        """
        Returns the coordinates of the central galaxies
        """
        _cen = Occupy.central(self)
        __nonzero = _cen.nonzero()[0]
        _cen = self.fout["xyz"][__nonzero]
        __nonzero = None
        print (_cen)
        return _cen


def mock(path = "/home/ajana/mockHOD"):
    global HODpar
    global key
    global filename

    rows = HODpar.shape[0]
    for i in range(1):
        par = {key[j]:HODpar[i][j] for j in range(len(key))}
        print (par)
        print ('Loading file...')
        occupy = Coordinates(par,filename)
        print ('File loaded!')
        tic = time.time()
        print ('Calculating coordinates...')
        coordinates = occupy.cen_coord()
        np.save(os.path.join(path,f'galaxies_{i:04d}.npy'),coordinates.astype('float16'))
        print ('Done!')
        print (f'Total number of galaxies = {coordinates.shape[0]}')
        print (f'Total time = {time.time()-tic}')
    gc.collect()

key = ["M_cut","M1" ,"sigma", "kappa", "alpha"]
HODpar = np.loadtxt("parameters.txt")
path = '/home/ajana/mockHOD'
import time
import os.path
def main():

    mock()

if __name__ == "__main__":
    main()    
