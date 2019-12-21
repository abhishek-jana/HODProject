#calculating VPF using halotools
from __future__ import print_function,division
from halotools.mock_observables import void_prob_func
import numpy as np
import time
import os
import multiprocessing as mp

#start = time.time()

N = int(1e5)
L = 2500.

radius = np.logspace(-1,1.2,30)

path = '/home/ajana/mockHOD/'

def voidprobfunc(rbins, n_ran, period, filename):
    global path
    if filename.endswith(".npy"):
        sample = np.load(os.path.join(path,filename))
        vpf = void_prob_func(sample,rbins=rbins,n_ran=n_ran,period=period)
        np.save(os.path.join('/home/ajana/VPF/','vpf_'+str(filename)),(rbins.astype('float64'),vpf.astype('float64')))
    else:
        raise TypeError("File should be in .npy format")
    #return None

def parallel_vpf(rbins,n_ran,period,path):
    pool = mp.Pool()
    filenames = [files for files in os.listdir(path) if files.endswith('.npy')]
    results = [pool.apply_async(voidprobfunc, args=(rbins,n_ran,period,i,)) for i in filenames]
    pool.close()
    pool.join()
    return results

def main():
    start = time.time()
    parallel_vpf(rbins = radius, n_ran = N, period = L, path = path)
    #voidprobfunc(rbins = radius,n_ran = N, period = L, filename = 'MDgalaxies_0100.npy')
    print (f'Total time taken: {time.time() - start}')
if __name__ == "__main__":
    main()

'''
for file in os.listdir(path)[:5]:
    if file.endswith(".npy"):
        start = time.time()
        gal = np.load(os.path.join(path,file)) # coordinates of the galaxies
    
        a = np.min(gal)
        b = np.max(gal)

        print (f'minimum = {a}\nmax = {b}\nNumber of particles = {len(gal)}')

        vpf = void_prob_func(gal,rbins = radius,n_ran = N,period = L)

        #print (f'Void Probability = {vpf}')
        #print (f'Radius of the sphere = {radius} Mpc')
        #print (f'Void Probability = {nvoid/N}')
        np.save('test_'+ str(file),(radius,vpf))
        print (f'Total time = {time.time()-start} sec')
        #print (sp)

'''
