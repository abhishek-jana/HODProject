# script for dividing box into 10X10 sample size and 1 at a time

import numpy as np
import os

# load the file
path = '/mnt/data4/Abhishek/' # fiducial mock

mockfile = np.load(path + 'mockHOD/galaxies_0000.npy')
bin_width = 250
num_bins = 10
b = [(i*bin_width, (i+1)*bin_width) for i in range(num_bins)]

import itertools
binsize = list(itertools.product(b, b)) # x and y coordinates



for j,bins in enumerate(binsize):
    arr = [i for i in range(mockfile.shape[0]) if ((mockfile[i,:-2]>=bins[0][0]) & (mockfile[i,:-2]<=bins[0][1]) & (mockfile[i,2:]>=bins[1][0]) & (mockfile[i,2:]<=bins[1][1]))]
    np.save(os.path.join(path+'fidmock',f'subsample_{j+200:04d}.npy'),np.delete(mockfile,arr,0))


#arr = [i for i in range(mockfile.shape[0]) if ((mockfile[i,:-2]>=2250) & (mockfile[i,:-2]<=2500) & (mockfile[i,1:-1]>=2250) & (mockfile[i,1:-1]<=2500))]
#np.save(os.path.join(path+'fidmock',f'subsample_0199.npy'),np.delete(mockfile,arr,0))

