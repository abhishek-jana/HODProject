import numpy as np
from scipy.special import erfc

filename = "/mnt/data1/MDhalos.npy"
 
class Occupy:
    def __init__(self,HODpar,fin):
        self.fout = {}
        self.load(fin)
        self.M = self.fout["mass"][:1000]
        self.M_cut = HODpar["M_cut"]
        self.sigma = HODpar["sigma"]
        self.kappa = HODpar["kappa"]
        self.M1 = HODpar["M1"]
        self.alpha = HODpar["alpha"]
    
    def load(self,fin):
        """
    file has 6 columns: Mass of Halo, Radius1 (scale radius), Radius 2 (virial radius), x, y, z
    
    Output:
    a dictionary containing mass,r1,r2,x,y,z
        """
        __file = np.load(fin)
        self.fout["mass"] = __file[:,0]
        self.fout["r1"] = __file[:,1]
        self.fout["r2"] = __file[:,2]
        self.fout["x"] = __file[:,3]
        self.fout["y"] = __file[:,4]
        self.fout["z"] = __file[:,5]
         
    def central(self):
        _Ncen = 0.5*erfc(np.log(10**self.M_cut/self.M)/(np.sqrt(2)*self.sigma))
        return np.random.binomial(1,_Ncen)
        
    def satellite(self):
        _Nsat = Occupy.central(self)*((self.M - self.kappa*(10**self.M_cut))/10**self.M1)**self.alpha
        return np.random.poisson(_Nsat)

class Coordinates(Occupy):
    def __init__(self,HODpar,fin):
        super().__init__(HODpar,fin)

    def cen_coord(self):
        _cen = Occupy.central(self)
        __nonzero = _cen.nonzero()
        x = np.take(self.fout["x"], __nonzero)
        y = np.take(self.fout["y"], __nonzero)
        z = np.take(self.fout["z"], __nonzero)
        _cen = np.vstack([x,y,z]).T
        x,y,z = [None,None,None]
        return _cen
    def sat_coord(self):
        pass
class sphere:
    def __init__(self,number_of_particles):
        self.number_of_particles = number_of_particles
    def new_position(self):
        radius = np.random.uniform(0.0,1.0, (self.number_of_particles,1))
        theta = np.random.uniform(0.,1.,(self.number_of_particles,1))*np.pi
        phi = np.arccos(1-2*np.random.uniform(0.0,1.,(self.number_of_particles,1)))
        x = radius * np.sin( theta ) * np.cos( phi )
        y = radius * np.sin( theta ) * np.sin( phi )
        z = radius * np.cos( theta )
        return (x,y,z)
         
par = {"M_cut": 13., "sigma": 0.98, "kappa": 1.13 , "M1": 14., "alpha" : .9}
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
def main():
    np.random.seed(42)
    #occupy = Coordinates(par,filename)
    #print (occupy.cen_coord())
    #print (occupy.satellite())
    sp = sphere(1000)
    plt.plot(sp.new_position()[0],sp.new_position()[2],'o')
    plt.savefig('sphere.png')
    print (sp.new_position()[0])


if __name__ == "__main__":
    main()    
