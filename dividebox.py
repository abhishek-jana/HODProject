# script for dividing box into 10X10 sample size and 1 at a time

import numpy as np
import os

# load the file
path = '/mnt/data4/Abhishek/' # fiducial mock

mockfile = np.load(path + 'mockHOD/galaxies_0000.npy')
bin_width = 250
num_bins = 10
bins = [(i*bin_width, (i+1)*bin_width) for i in range(num_bins)]

import itertools
binsize = list(itertools.product(bins, bins)) # x and y coordinates



for i in binsize:
    arr = [i for i in range(mockfile.shape[0]) if ((mockfile[i,:-2]>=binsize[0][0]) & (mockfile[i,:-2]<=binsize[0][1]) & (mockfile[i,1:-1]>=binsize[1][0]) & (mockfile[i,1:-1]<=binsize[1][1]))]
      np.save(os.path.join(path+'fidmock',f'MDgalaxies_{i:04d}.npy'))


print (mockfile.shape[0])



