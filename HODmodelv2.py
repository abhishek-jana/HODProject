### NOT WORKING ###


import numpy as np
from scipy.special import erfc
import multiprocessing as mp
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

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
        out["x"] = __file[:,3]
        out["y"] = __file[:,4]
        out["z"] = __file[:,5]
        __file = [None]
        return out

    def central(self):
        """
        Returns 1 if there's a central galaxy. 
        Distribution found using eq.12 of https://iopscience.iop.org/article/10.1088/0004-637X/728/2/126/pdf
        """
        _Ncen = 0.5*erfc(np.log(10**self.M_cut/self.M)/(np.sqrt(2)*self.sigma))
        return np.random.binomial(1,_Ncen)
        
    def satellite(self):
        """
        Returns Poisson distribution with mean _Nsat.
        Distribution found using eq.13 of https://iopscience.iop.org/article/10.1088/0004-637X/728/2/126/pdf
        """
        _Nsat = Occupy.central(self)*((self.M - self.kappa*(10**self.M_cut))/10**self.M1)**self.alpha
        return np.random.poisson(_Nsat)

class Coordinates(Occupy):
    def __init__(self,HODpar,fin): 
        super().__init__(HODpar,fin)

    def sphere_coordinates(self,number_of_particles,R):
        """
        Given the number of particles this will generate random uniform points inside a sphere of radius R
        """
        u = np.random.uniform(0.,1., (number_of_particles,1))
        theta = np.arccos(1-2*np.random.uniform(0.,1.,(number_of_particles,1)))
        phi = np.random.uniform(0.0,1.,(number_of_particles,1))*2*np.pi
        x = np.cbrt(u)*R*np.sin( theta ) * np.cos( phi )
        y = np.cbrt(u)*R*np.sin( theta ) * np.sin( phi )
        z = np.cbrt(u)*R*np.cos( theta)
        u,theta,phi = [None,None,None]
        return np.c_[x,y,z]

    def cen_coord(self):
        """
        Returns the coordinates of the central galaxies
        """
        _cen = Occupy.central(self)
        __nonzero = _cen.nonzero()
        xcen = np.take(self.fout["x"], __nonzero)
        ycen = np.take(self.fout["y"], __nonzero)
        zcen = np.take(self.fout["z"], __nonzero)
        _cen = np.vstack([xcen,ycen,zcen]).T
        xcen,ycen,zcen,__nonzero = [None,None,None,None]
        return _cen
    
    def sat_coord(self,n_jobs = 1):
        """
        Returns the coordinates of the satellite galaxies.
        Change the radius to Mpc
        """
        backend = 'loky'
        _sat = Occupy.satellite(self)
        __nonzero = _sat.nonzero()
        radius = np.take(self.fout["r1"],__nonzero)/1000.
        xsat = np.take(self.fout["x"],__nonzero)
        ysat = np.take(self.fout["y"],__nonzero)
        zsat = np.take(self.fout["z"],__nonzero)
        _sat = np.take(_sat,__nonzero)
        xyz_sat = np.vstack([xsat,ysat,zsat]).T
        xyz_sat = np.repeat(xyz_sat,_sat[0],axis=0)
        xsat,ysat,zsat = [None,None,None]
        results = Parallel(n_jobs,backend=backend)(delayed(Coordinates.sphere_coordinates)(self,i,j) for i,j in zip(_sat[0],radius[0]))
        radius,_sat,__nonzero = [None,None,None]
        return np.vstack((results)) + xyz_sat
    
    def galaxy_coordinates(self,n_jobs = 1):
        """
        Returns the combined galaxy coordinates of satellite and central galaxies
        """
        return np.vstack((Coordinates.cen_coord(self),Coordinates.sat_coord(self,n_jobs = 1)))
         
par = {"M_cut": 13., "sigma": 0.98, "kappa": 1.13 , "M1": 14., "alpha" : .9}
import time
def main():
    np.random.seed(42)
    occupy = Coordinates(par,filename)
    print (np.max(occupy.fout["x"]))
    #sat = occupy.satellite()
    #r = np.take(occupy.fout["r1"],sat.nonzero())
    #sat = np.take(sat,sat.nonzero())
    tic = time.time()
    #xyz_sat = Parallel(n_jobs=-1)(delayed(occupy.sphere_coordinates)(i,j) for i,j in zip(sat[0],r[0])) 
    print (occupy.sat_coord(1)[0])
    #print (xyz_sat.shape)
    print (f'Total time = {time.time()-tic}')
if __name__ == "__main__":
    main()    
