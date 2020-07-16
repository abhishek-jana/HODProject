#calculating CIA using halotools
from __future__ import print_function,division
import numpy as np
import time
import os
from halotools.mock_observables import counts_in_cylinders
import multiprocessing as mp
from Corrfunc.theory.wp import wp
#from halotools.mock_observables import return_xyz_formatted_array

#NEED to convert sample to 'float64' for cic to work

path = '/mnt/data4/Abhishek/mockHOD/'
#path = '/mnt/data4/Abhishek/fidmock'

bins = np.logspace(-1,1.5,31)
L = 2500.
pi_max = 60

def ProjectedCorrFunc(boxsize, pimax, nthreads, binfile,filename):
    global path
    if filename.endswith(".npy"):
        sample = np.load(os.path.join(path,filename))
        if (sample.dtype != 'float64'):
            sample = sample.astype('float64')
            wp_counts = wp(boxsize, pimax, nthreads, binfile, X = sample[:,0], Y = sample[:,1], Z = sample[:,2])
            corr = [i[3] for i in wp_counts]
            np.save(os.path.join('/mnt/data4/Abhishek/WP','wp_'+str(filename)),[corr,bins[:-1]])
            #np.save(os.path.join('/mnt/data4/Abhishek/fidmock/wp','wp_'+str(filename)),[corr,bins[:-1]])
            del wp_counts
            del corr
            del sample
        else:
            wp_counts = wp(boxsize, pimax, nthreads, binfile, X = sample[:,0], Y = sample[:,1], Z = sample[:,2])
            corr = [i[3] for i in wp_counts]
            np.save(os.path.join('/mnt/data4/Abhishek/WP','wp_'+str(filename)),[corr,bins[:-1]])
            #np.save(os.path.join('/mnt/data4/Abhishek/wp','wp_'+str(filename)),[corr,bins[:-1]])
            del wp_counts
            del corr
            del sample
    else:
        raise TypeError("File should be in .npy format")
    #return None

def cal_wp(boxsize, pimax, nthreads, binfile,path):
    filenames = [files for files in os.listdir(path) if files.endswith('.npy')]
    results = [ProjectedCorrFunc(boxsize, pimax, nthreads, binfile, i) for i in filenames]
    #return results

def main():
    start = time.time()
    cal_wp(boxsize = L, pimax=pi_max, nthreads=50, binfile=bins,path=path)
    #CountsInCylinders( proj_search_radius = proj_search_radius,cylinder_half_length = cylinder_half_length, period = L, filename = 'galaxies_0100.npy')
    print (f'Total time taken: {time.time() - start}')
if __name__ == "__main__":
    main() 
       


