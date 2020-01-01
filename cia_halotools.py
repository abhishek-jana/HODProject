#calculating CIA using halotools
from __future__ import print_function,division
import numpy as np
import time
import os
from halotools.mock_observables import counts_in_cylinders
import multiprocessing as mp
#from halotools.mock_observables import return_xyz_formatted_array

#NEED to convert sample to 'float64' for cic to work

path = '/mnt/data4/Abhishek/mockHOD/'

inner_radius = 2.0
outer_radius = 5.0
cylinder_half_length = 10.0
L = 2500.

def CountsInAnnuli(inner_radius,outer_radius, cylinder_half_length, period, filename):
    global path
    if filename.endswith(".npy"):
        sample = np.load(os.path.join(path,filename))
        if (sample.dtype != 'float64'):
            sample = sample.astype('float64')
            outer = counts_in_cylinders(sample,sample,proj_search_radius=outer_radius,cylinder_half_length=cylinder_half_length,period=period)
            inner = counts_in_cylinders(sample,sample,proj_search_radius=inner_radius,cylinder_half_length=cylinder_half_length,period=period)
            np.save(os.path.join('/mnt/data4/Abhishek/CIA/random','cia_'+str(filename)),(outer-inner).astype('int8'))
            del outer
            del inner
        else:
            outer = counts_in_cylinders(sample,sample,proj_search_radius=outer_radius,cylinder_half_length=cylinder_half_length,period=period)
            inner = counts_in_cylinders(sample,sample,proj_search_radius=inner_radius,cylinder_half_length=cylinder_half_length,period=period)
            np.save(os.path.join('/mnt/data4/Abhishek/CIA/random','cia_'+str(filename)),(outer-inner).astype('int8'))
            del outer
            del inner
    else:
        raise TypeError("File should be in .npy format")
    #return None

def parallel_cia(inner_radius,outer_radius,cylinder_half_length,period,path):
    pool = mp.Pool()
    filenames = ['galaxies_'+str(files)+'.npy' for files in range(10000)]
    #filenames = [files for files in os.listdir(path) if files.endswith('.npy')]
    results = [pool.apply_async(CountsInAnnuli, args=(inner_radius,outer_radius,cylinder_half_length,period,i,)) for i in filenames]
    pool.close()
    pool.join()
    return results

def main():
    start = time.time()
    parallel_cia(inner_radius=inner_radius,outer_radius=outer_radius,cylinder_half_length = cylinder_half_length, period = L, path = path)
    #CountsInCylinders( proj_search_radius = proj_search_radius,cylinder_half_length = cylinder_half_length, period = L, filename = 'galaxies_0100.npy')
    print (f'Total time taken: {time.time() - start}')
if __name__ == "__main__":
    main() 
       


