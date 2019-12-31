#calculating CIC using halotools
from __future__ import print_function,division
import numpy as np
import time
import os
from halotools.mock_observables import counts_in_cylinders
import multiprocessing as mp
#from halotools.mock_observables import return_xyz_formatted_array

#NEED to convert sample to 'float64' for cic to work

path = '/mnt/data4/Abhishek/mockHOD/'

proj_search_radius = 2.0
cylinder_half_length = 10.0
L = 2500.

def CountsInCylinders(proj_search_radius, cylinder_half_length, period, filename):
    global path
    if filename.endswith(".npy"):
        sample = np.load(os.path.join(path,filename))
        if (sample.dtype != 'float64'):
            sample = sample.astype('float64')
            cic = counts_in_cylinders(sample,sample,proj_search_radius=proj_search_radius,cylinder_half_length=cylinder_half_length,period=period)
            np.save(os.path.join('/mnt/data4/Abhishek/CIC/','cic_'+str(filename)),cic.astype('int8'))
        else:
            cic = counts_in_cylinders(sample,sample,proj_search_radius=proj_search_radius,cylinder_half_length=cylinder_half_length,period=period)
            np.save(os.path.join('/mnt/data4/Abhishek/CIC/','cic_'+str(filename)),cic.astype('int8'))

    else:
        raise TypeError("File should be in .npy format")
    #return None

def parallel_cic(proj_search_radius,cylinder_half_length,period,path):
    pool = mp.Pool()
    filenames = ['galaxies_'+str(files)+'.npy' for files in range(5000,10000)]
    #filenames = [files for files in os.listdir(path) if files.endswith('.npy')]
    results = [pool.apply_async(CountsInCylinders, args=( proj_search_radius,cylinder_half_length,period,i,)) for i in filenames]
    pool.close()
    pool.join()
    return results

def main():
    start = time.time()
    parallel_cic( proj_search_radius = proj_search_radius,cylinder_half_length = cylinder_half_length, period = L, path = path)
    #CountsInCylinders( proj_search_radius = proj_search_radius,cylinder_half_length = cylinder_half_length, period = L, filename = 'galaxies_0100.npy')
    print (f'Total time taken: {time.time() - start}')
if __name__ == "__main__":
    main() 
       





'''
for filename in os.listdir(path)[:1]:
    if filename.endswith(".npy"):
        print (filename)
        start = time.time()
        gal = np.load(os.path.join(path,filename)) # coordinates of the galaxies
        print (np.shape(gal))
        gal = gal.astype('float64')
        cic = counts_in_cylinders(gal, gal, proj_search_radius = 2.0 ,cylinder_half_length = 10.0 ,period = L,num_threads = 'max')
        print (len(cic))
        print (f'max number of galaxies in a cylinder = {np.max(cic)}')
        #np.save(os.path.join('/home/ajana/CIC/','test_'+str(filename)),cic.astype('int8'))
        print (f'Total time = {time.time()-start} sec')
'''
